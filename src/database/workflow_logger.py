
# Import Path to specify a database location.
from pathlib import Path

# Import our default SQLite database location.
from src.database.db import DATABASE_PATH

# Import the function that saves one request.
from src.database.logger import log_request

# Import our existing escalation workflow and result type.
from src.verifier.escalation import (
    EscalationResult,
    route_with_escalation,
)

# Import the existing classifier.
from src.classifier.complexity_classifier import classify_complexity


# Run an AI workflow and log its provider attempts.
async def run_and_log(
    prompt: str,
    db_path: Path = DATABASE_PATH,
    decision=None,
) -> EscalationResult:
    
    
    # Use the original classifier if no decision was supplied.
    if decision is None:

        # Classify the prompt for our existing database records.
        complexity = classify_complexity(prompt)

        # Run the original workflow to preserve existing behavior.
        result = await route_with_escalation(prompt)

    # Use the preselected routing decision when available.
    else:

        # Record the same complexity selected by the router.
        complexity = decision.complexity

        # Execute the workflow with that exact routing decision.
        result = await route_with_escalation(
            prompt,
            decision=decision,
        )


    # Process every response generated during this workflow.
    for attempt_number, response in enumerate(
        result.responses,
        start=1,
    ):

        # Save this provider attempt into SQLite.
        log_request(
            model_name=response.model_id,
            provider=response.provider,
            complexity=complexity,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            estimated_cost_usd=response.cost_usd,
            escalated=attempt_number > 1,
            attempts=attempt_number,
            db_path=db_path,
        )

    # Return the original result to the caller.
    return result
