
# Import dataclass to store routing decisions neatly.
from dataclasses import dataclass

# Import our existing complexity classifier.
from src.classifier.complexity_classifier import (
    Complexity,
    classify_complexity,
)

# Import our trained V2 complexity classifier.
from src.classifier.v2_adapter import predict_v2_complexity

# Import our reusable risk-aware routing policy.
from src.router.risk_policy import apply_policy

# Import the model registry and its configuration type.
from src.providers.model_registry import (
    ModelConfig,
    load_model_registry,
)


# Import our existing unified AI provider client.
from src.providers.client import send_request

# Import the standardized response type.
from src.providers.response import ModelResponse

# Import our reusable prompt validator.
from src.router.prompt_validator import validate_prompt

# Define which model handles each difficulty level.
# These are initial choices that we will evaluate later.
MODEL_BY_COMPLEXITY = {
    "simple": "gpt-4o-mini",
    "medium": "claude-haiku",
    "complex": "gpt-4o",
}


# Create a structure to describe a routing decision.


@dataclass(frozen=True)
class RoutingDecision:

    # Store the predicted question difficulty.
    complexity: Complexity

    # Store the selected model's name.
    model_name: str

    # Store the selected model's configuration.
    model_config: ModelConfig

    # Explain why the model was selected.
    reason: str

# Store both the routing decision and the AI response.
@dataclass(frozen=True)
class RoutedResponse:

    # Explain which model was chosen and why.
    decision: RoutingDecision

    # Store the actual answer and usage information.
    response: ModelResponse


# Create a function that chooses a model for a question.
# Allow callers to choose the V2 classifier explicitly.
def route_prompt(
    prompt: str,
    use_v2: bool = False,
    risk_threshold: float = 0.40,
) -> RoutingDecision:

        # Validate and clean the prompt before classifying it.
    prompt = validate_prompt(prompt)

    
    # Use the trained V2 classifier only when requested.
    if use_v2:

        # Predict complexity and category probabilities.
        prediction = predict_v2_complexity(prompt)

        # Prepare the prediction for our routing policy.
        policy_input = {
            "predicted": prediction.complexity,
            "probabilities": prediction.probabilities,
        }

        # Apply our candidate risk-aware escalation rule.
        complexity = apply_policy(
            policy_input,
            threshold=risk_threshold,
        )

    # Otherwise preserve our existing classifier.
    else:

        # Classify the prompt using the original method.
        complexity = classify_complexity(prompt)

    # Select the model assigned to the final complexity.
    model_name = MODEL_BY_COMPLEXITY[complexity]


    # Load all available models from our existing YAML registry.
    registry = load_model_registry()

    # Stop if our routing policy selects an unregistered model.
    if model_name not in registry:
        raise ValueError(
            f"Selected model '{model_name}' is missing from the registry."
        )

    # Retrieve the selected model's configuration.
    model_config = registry[model_name]


    # Identify which classifier produced the decision.
    classifier_name = "V2 ML classifier" if use_v2 else "original classifier"

    # Explain the final model selection.
    reason = (
        f"Question routed as '{complexity}' using {classifier_name}. "
        f"Selected model '{model_name}'."
    )

    # Return the decision to the calling program.
    return RoutingDecision(
        complexity=complexity,
        model_name=model_name,
        model_config=model_config,
        reason=reason,
    )


# Store the configuration needed to call the selected model.
model_config: ModelConfig



# Create an asynchronous function for automatic AI routing.
async def route_and_send(
    prompt: str,
    use_v2: bool = False,
    risk_threshold: float = 0.40,
) -> RoutedResponse:

    # Validate the user's question before sending it.
    prompt = validate_prompt(prompt)

    # Choose the AI model using the requested classifier.
    decision = route_prompt(
        prompt,
        use_v2=use_v2,
        risk_threshold=risk_threshold,
    )

    # Send the question to the selected AI provider.
    response = await send_request(
        prompt=prompt,
        model_config=decision.model_config,
    )

    # Return the model decision and the AI's response.
    return RoutedResponse(
        decision=decision,
        response=response,
    )

