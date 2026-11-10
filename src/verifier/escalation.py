
# Import dataclass to store the escalation result.
from dataclasses import dataclass

# Import the existing intelligent routing function.
from src.router.intelligent_router import route_prompt

# Import our existing provider client.
from src.providers.client import send_request

# Import the standardized provider response.
from src.providers.response import ModelResponse

# Import the model registry.
from src.providers.model_registry import load_model_registry

# Import our configured spending and token limits.
from config.budget import (
    MAX_REQUEST_COST_USD,
    MAX_OUTPUT_TOKENS,
)

# Import the model-specific cost estimator.
from src.router.model_cost_estimator import estimate_model_cost

# Import our prompt-based token estimator.
from src.router.token_estimator import estimate_input_tokens

# Import our shared prompt validation function.
from src.router.prompt_validator import validate_prompt

# Import our existing workflow budget estimator.
from src.router.budget_policy import estimate_routed_workflow_cost

# Import our asynchronous quality checker and result type.
from src.verifier.quality_verifier import (
    QualityResult,
    verify_quality_async,
)


# Store the result of our escalation process.
@dataclass(frozen=True)
class EscalationResult:

    # Store the final AI response.
    response: ModelResponse

    # Indicate whether we tried a stronger model.
    escalated: bool

    # Record how many AI requests were made.
    attempts: int

    # Explain what happened.
    reason: str

    # Store the verification report for the final AI answer.
    quality: QualityResult

    # Store every AI response made during this workflow.
    # This lets us account for the cost of both attempts.
    responses: tuple[ModelResponse, ...] = ()



# Send a question and escalate only if the first answer is empty.

# Route a prompt, verify the response, and escalate if empty.
async def route_with_escalation(prompt: str) -> EscalationResult:

     
    # Validate and clean the customer's question.
    prompt = validate_prompt(prompt)

    # Estimate the maximum planned cost of this routed workflow.
    # Include the first model and any possible GPT-4o fallback.
    estimated_workflow_cost = estimate_routed_workflow_cost(
        prompt=prompt,
    )

    # Reject the workflow before any paid provider call
    # when its estimated cost exceeds our configured limit.
    if estimated_workflow_cost > MAX_REQUEST_COST_USD:
        raise ValueError(
            "Estimated workflow cost exceeds the configured budget."
        )

    # Choose the first model only after the budget check passes.
    decision = route_prompt(prompt)


    # Send the question to the selected model.
    first_response = await send_request(
        prompt=prompt,
        model_config=decision.model_config,
    )

    # Verify the first AI response asynchronously.
    first_quality = await verify_quality_async(
        answer=first_response.output_text,
    )

    # Return immediately if the answer contains text.
    # Do not escalate solely because its word-count score is low.
    if first_response.output_text.strip():
        return EscalationResult(
            response=first_response,
            escalated=False,
            attempts=1,
            reason="Initial model returned a nonempty answer.",
            quality=first_quality,
            responses=(first_response,),
        )

    # Avoid escalation if the first model is already gpt-4o.
    if decision.model_name == "gpt-4o":
        return EscalationResult(
            response=first_response,
            escalated=False,
            attempts=1,
            reason="Initial answer was empty; already using fallback model.",
            quality=first_quality,
            responses=(first_response,),
        )

    # Load the existing model configurations.
    registry = load_model_registry()

    # Select the stronger fallback model.
    stronger_model = registry["gpt-4o"]

    
    # Calculate how much money remains after the first request.
    remaining_budget = (
        MAX_REQUEST_COST_USD - first_response.cost_usd
    )

    # Estimate the fallback's input tokens from the prompt.
    estimated_fallback_input_tokens = estimate_input_tokens(prompt)

    # Estimate the possible GPT-4o fallback cost.
    estimated_fallback_cost = estimate_model_cost(
        model_name="gpt-4o",
        input_tokens=estimated_fallback_input_tokens,
        output_tokens=MAX_OUTPUT_TOKENS,
    )

    # Check whether the fallback fits the remaining budget.
    if estimated_fallback_cost > remaining_budget:

        # Return the first response without making another API call.
        return EscalationResult(
            response=first_response,
            escalated=False,
            attempts=1,
            reason=(
                "Initial answer was empty; fallback blocked "
                "because its estimated cost exceeds the remaining budget."
            ),
            quality=first_quality,
            responses=(first_response,),
        )


    # Make one additional AI request.
    second_response = await send_request(
        prompt=prompt,
        model_config=stronger_model,
    )

    # Verify the second AI response.
    second_quality = await verify_quality_async(
        answer=second_response.output_text,
    )

    # Return the final answer and its verification report.
    
    # Return the final answer and its verification report.
    return EscalationResult(

        # Store the final answer from the fallback model.
        response=second_response,

        # Store both AI responses for accurate cost tracking.
        responses=(first_response, second_response),

        # Confirm that model escalation occurred.
        escalated=True,

        # Record the total number of AI requests.
        attempts=2,

        # Explain why escalation happened.
        reason="Initial answer was empty; escalated once to gpt-4o.",

        # Store the quality report for the final answer.
        quality=second_quality,
    )

