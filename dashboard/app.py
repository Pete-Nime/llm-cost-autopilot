
# Import Streamlit to create our dashboard.
import streamlit as st

# Import our existing database analytics functions.
from src.database.analytics import (
    get_usage_summary,
    get_model_breakdown,
)

# Import our existing default database location.
from src.database.db import DATABASE_PATH


# Configure the browser tab and page layout.
st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="📊",
    layout="wide",
)


# Display the dashboard title.
st.title("📊 LLM Cost Autopilot")

# Explain the purpose of the dashboard.
st.caption("AI usage and estimated API cost monitoring")



# Import Path to locate our demonstration database.
from pathlib import Path

# Find the project root using this dashboard file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the location of our demo database.
DEMO_DB = PROJECT_ROOT / "data" / "demo_autopilot.db"

# Allow the user to switch between real and demo records.
use_demo = st.toggle(
    "Show demonstration data",
    value=False,
)

# Choose which database the dashboard should read.
selected_db = DEMO_DB if use_demo else DATABASE_PATH

# Read summary statistics from the selected database.
summary = get_usage_summary(db_path=selected_db)

# Read model spending from the same database.
models = get_model_breakdown(db_path=selected_db)

# Clearly identify demonstration data when selected.
if use_demo:
    st.info("DEMO MODE — These records are simulated, not actual API spending.")



# Create four columns for our main metrics.
col1, col2, col3, col4 = st.columns(4)

# Display the number of provider attempts.
col1.metric(
    "AI Provider Attempts",
    summary["total_attempts"],
)

# Display the estimated API spending.
col2.metric(
    "Estimated API Cost",
    f'${summary["total_cost_usd"]:.6f}',
)

# Display the average response time.
col3.metric(
    "Average Latency",
    f'{summary["average_latency_ms"]:.0f} ms',
)

# Display the number of fallback attempts.
col4.metric(
    "Fallback Attempts",
    summary["fallback_attempts"],
)


# Add a divider between the metrics and charts.
st.divider()

# Display the token usage heading.
st.subheader("Token Usage")

# Create two columns for token metrics.
token_col1, token_col2 = st.columns(2)

# Display the total input tokens.
token_col1.metric(
    "Input Tokens",
    summary["total_input_tokens"],
)

# Display the total output tokens.
token_col2.metric(
    "Output Tokens",
    summary["total_output_tokens"],
)


# Display the model spending section.
st.subheader("Estimated Cost by AI Model")

# Check whether our database contains model records.
if models:

    # Create a dictionary containing model names and costs.
    cost_data = {
        item["model_name"]: item["total_cost_usd"]
        for item in models
    }

    # Display a bar chart comparing estimated spending.
    st.bar_chart(cost_data)

    # Show the detailed model breakdown as a table.
    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True,
    )

# Handle the case where no AI usage has been recorded.
else:

    # Explain why the dashboard has no chart yet.
    st.info(
        "No AI requests have been logged yet. "
        "Run the AI workflow to generate usage records."
    )


# Explain how dashboard values are calculated.
st.caption(
    "Costs are estimates based on configured model pricing. "
    "Each provider attempt is counted separately, "
    "including fallback attempts."
)
