
# Import FastAPI to create our API application.
from fastapi import FastAPI

# Import our existing database analytics function.
from src.database.analytics import get_usage_summary

from fastapi import FastAPI

# Import Pydantic to validate incoming request data.
from pydantic import BaseModel, Field

# Import our existing intelligent routing function.
from src.router.intelligent_router import route_prompt



# Create our FastAPI application.
app = FastAPI(
    title="LLM Cost Autopilot API",
    description="AI model routing and cost monitoring",
    version="1.0.0",
)

# Define the information a client must send to our API.
class RoutingRequest(BaseModel):

    # Require a prompt containing between 1 and 2000 characters.
    prompt: str = Field(
        min_length=1,
        max_length=2000,
    )



# Create a simple endpoint to check whether the API works.
@app.get("/health")
def health_check():

    # Return a simple confirmation message.
    return {
        "status": "healthy",
        "service": "LLM Cost Autopilot",
    }


# Create an endpoint that returns our usage statistics.
@app.get("/analytics/summary")
def analytics_summary():

    # Read the stored usage statistics from SQLite.
    summary = get_usage_summary()

    # Return the statistics as JSON.
    return summary

# Create an endpoint that previews intelligent model routing.
@app.post("/route/preview")
def route_preview(request: RoutingRequest):

    # Remove whitespace from the beginning and end.
    cleaned_prompt = request.prompt.strip()

    # Reject prompts containing only spaces.
    if not cleaned_prompt:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=422,
            detail="Prompt cannot contain only whitespace.",
        )

    # Ask our existing router to choose an AI model.
    decision = route_prompt(cleaned_prompt)

    # Return the decision without calling an AI provider.
    return {
        "complexity": decision.complexity,
        "selected_model": decision.model_name,
        "reason": decision.reason,
        "executed": False,
    }

