
# Import our existing model registry loader.
from src.providers.model_registry import load_model_registry


# Estimate the cost of a selected AI model.
def estimate_model_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:

    # Load the models from config/models.yaml.
    registry = load_model_registry()

    # Make sure the requested model exists.
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")

    # Retrieve the selected model's ModelConfig object.
    model = registry[model_name]

    # Reuse the cost calculator we already built in Phase 1.
    # It also checks that token counts are not negative.
    return model.calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
