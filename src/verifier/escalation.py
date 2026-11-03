
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


# Send a question and escalate only if the first answer is empty.

# Route a prompt, verify the response, and escalate if empty.
async def route_with_escalation(prompt: str) -> EscalationResult:

    # Ask our intelligent router to select the first model.
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
        )

    # Avoid escalation if the first model is already gpt-4o.
    if decision.model_name == "gpt-4o":
        return EscalationResult(
            response=first_response,
            escalated=False,
            attempts=1,
            reason="Initial answer was empty; already using fallback model.",
            quality=first_quality,
        )

    # Load the existing model configurations.
    registry = load_model_registry()

    # Select the stronger fallback model.
    stronger_model = registry["gpt-4o"]

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
    return EscalationResult(
        response=second_response,
        escalated=True,
        attempts=2,
        reason="Initial answer was empty; escalated once to gpt-4o.",
        quality=second_quality,
    )
