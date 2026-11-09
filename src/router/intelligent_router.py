
# Import dataclass to store routing decisions neatly.
from dataclasses import dataclass

# Import our existing complexity classifier.
from src.classifier.complexity_classifier import (
    Complexity,
    classify_complexity,
)

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
def route_prompt(prompt: str) -> RoutingDecision:

        # Validate and clean the prompt before classifying it.
    prompt = validate_prompt(prompt)

    # Ask our classifier to determine question difficulty.
    complexity = classify_complexity(prompt)

    # Look up the model assigned to this difficulty.
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


    # Create an explanation for the routing decision.
    reason = (
        f"Question classified as '{complexity}'. "
        f"Selected model '{model_name}' using the initial "
        "complexity-based routing policy."
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
async def route_and_send(prompt: str) -> RoutedResponse:

    # Validate the prompt before routing and sending it.
    prompt = validate_prompt(prompt)

    # Classify the question and select a registered model.
    decision = route_prompt(prompt)

    # Send the question using our existing unified client.
    response = await send_request(
        prompt=prompt,
        model_config=decision.model_config,
    )

    # Return both the routing decision and the AI answer.
    return RoutedResponse(
        decision=decision,
        response=response,
    )
