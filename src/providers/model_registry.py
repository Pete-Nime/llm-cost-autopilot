
from dataclasses import dataclass
from typing import Literal


# These are the three levels of task difficulty.
QualityTier = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelConfig:
    """
    Store information about one AI model.

    Think of this as an employee information card.
    """

    provider: str
    model_id: str

    # Prices are in USD per 1 million tokens.
    input_cost_per_million: float
    output_cost_per_million: float

    # Expected response time in milliseconds.
    average_latency_ms: float

    # Our initial estimate of model capability.
    quality_tier: QualityTier

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate the estimated API cost of a request.
        """

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        # Calculate input token cost.
        input_cost = (
            input_tokens / 1_000_000
        ) * self.input_cost_per_million

        # Calculate output token cost.
        output_cost = (
            output_tokens / 1_000_000
        ) * self.output_cost_per_million

        return input_cost + output_cost



# Import Path to locate our configuration file.
from pathlib import Path

# Import YAML to read the model configuration.
import yaml


# Define a function to load registered AI models.
def load_model_registry(
    config_path: str | Path | None = None
) -> dict[str, ModelConfig]:

    # Find the project root using this file's location.
    project_root = Path(__file__).resolve().parents[2]

    # Use the default YAML file unless another path is supplied.
    path = (
        Path(config_path)
        if config_path is not None
        else project_root / "config" / "models.yaml"
    )

    # Open the YAML file for reading.
    with path.open("r", encoding="utf-8") as file:

        # Convert the YAML contents into a Python dictionary.
        configuration = yaml.safe_load(file)

    # Ensure that the configuration has a models section.
    if not isinstance(configuration, dict):
        raise ValueError("Model configuration must be a dictionary")

    # Read the models section.
    models = configuration.get("models")

    # Reject missing or incorrectly formatted model entries.
    if not isinstance(models, dict):
        raise ValueError("Configuration must contain a models dictionary")

    # Create an empty dictionary for registered models.
    registry = {}

    # Visit every model in the configuration.
    for name, settings in models.items():

        # Ensure each model has a dictionary of settings.
        if not isinstance(settings, dict):
            raise ValueError(f"Invalid configuration for {name}")

        # Convert the model settings into a ModelConfig object.
        registry[name] = ModelConfig(**settings)

    # Return the completed model registry.
    return registry
