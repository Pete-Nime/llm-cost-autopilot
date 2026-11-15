
# LLM Cost Autopilot

An AI engineering project that routes LLM requests to suitable models based on task complexity, risk-aware policies, and estimated API costs.

## Project Overview

LLM Cost Autopilot demonstrates how to build a cost-aware AI routing system using Python.

Instead of sending every request to the same expensive model, the system can select from configured AI models according to the predicted complexity of the prompt.

The project includes:

- AI provider integration
- Model configuration and pricing
- Prompt complexity classification
- V2 machine-learning classifier
- Risk-aware model routing
- Request budget enforcement
- Response verification and fallback
- SQLite usage logging
- FastAPI endpoints
- Automated Python tests

## Architecture

```text
Client Request
      |
      v
FastAPI
      |
      v
Complexity Classifier
      |
      v
Risk-Aware Router
      |
      v
Budget Policy
      |
      v
AI Provider
      |
      v
Quality Verification
      |
      +----> Optional GPT-4o fallback
      |
      v
SQLite Logging
```

## Project Structure

```text
llm-cost-autopilot/
├── config/
├── data/
├── models/
├── src/
│   ├── api/
│   ├── classifier/
│   ├── database/
│   ├── providers/
│   ├── router/
│   └── verifier/
├── tests/
├── requirements.txt
├── pytest.ini
└── README.md
```

## Technology Stack

- Python
- FastAPI
- Pydantic
- OpenAI API
- Anthropic API
- Ollama
- scikit-learn
- SQLite
- pytest
- Git and GitHub

## Installation

Clone the repository:

```bash
git clone https://github.com/Pete-Nime/llm-cost-autopilot.git
cd llm-cost-autopilot
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```


## Running the API

Start the FastAPI development server from the project root:

```bash
python -m uvicorn src.api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

The API provides endpoints for health checks, routing previews, request execution, and usage analytics.

AI execution is disabled by default. Configure provider credentials, budget limits, and the execution environment variable before enabling paid requests.


## Configuration

Provider API credentials should be stored in a local `.env` file.

Never commit API keys or secrets to GitHub.

Real AI execution is disabled unless the following environment variable is set:

```text
ENABLE_AI_EXECUTION=true
```

Only enable execution when your provider credentials and spending limits are configured.

## Running Tests

Run all tests:

```bash
python -m pytest tests/ -q
```

Latest locally verified result:

**133 tests passed.**

## API Endpoints

| Endpoint | Purpose |
|---|---|
| GET /health | Check API health |
| GET /analytics/summary | View usage statistics |
| POST /route/preview | Preview model selection |
| POST /route/execute | Execute and log an AI request |

### Example V2 Preview Request

```json
{
  "prompt": "Design a secure distributed authentication system.",
  "use_v2": true,
  "risk_threshold": 0.40
}
```

## Budget Protection

The system estimates workflow costs before sending requests to paid providers.

If a workflow exceeds the configured budget, execution is blocked.

Unexpected routing or budget-estimation errors are handled without proceeding to paid execution.

## Testing Strategy

Tests cover:

- Model registry and pricing
- Provider integration
- Complexity classification
- Intelligent routing
- Budget policies
- Response verification and escalation
- SQLite logging
- FastAPI validation
- V2 routing integration

Provider calls are mocked in automated integration tests to avoid API charges.


## Classifier Evaluation Results

The V2 classifier uses TF-IDF text features and seven numerical prompt features to predict three complexity categories: simple, medium, and complex.

### Benchmark Evaluation

Evaluated on 90 labeled benchmark prompts:

| Metric | Result |
|---|---|
| Classifier accuracy | 81.11% |
| Routing accuracy at 40% risk threshold | 85.56% |
| Complex tasks routed below complex | 3 of 30 |
| Non-complex tasks routed as complex | 3 |
| Benchmark prompts flagged for training similarity | 0 at threshold 0.65 |

The 40% threshold achieved the highest routing accuracy among the six thresholds tested on this benchmark.

### Challenge Evaluation

Evaluated on 15 challenge prompts:

| Metric | Result |
|---|---|
| Classifier accuracy | 73.33% |
| Incorrect predictions | 4 |
| Complex prompts classified as simple | 2 |

These challenge results expose an important limitation: the current risk policy only escalates medium predictions to complex.

The benchmark and challenge datasets have already been reviewed during development, so these results should not be treated as independent, untouched test-set performance.


## Current Limitations


- The V2 classifier needs evaluation on a fresh, untouched test dataset before making generalization claims.
- The current risk policy escalates medium predictions when the complex-class probability exceeds the configured threshold. Simple predictions are not escalated.
- The response verifier uses a basic word-count score. It does not measure factual correctness, reasoning quality, or semantic completeness.
- The GPT-4o fallback is triggered by an empty initial response, not by a low word-count score or an incorrect answer.
- Model routing depends on classifier accuracy and manually configured model-selection policies.
- Cost estimates may differ from actual provider billing.
- The project is a portfolio prototype, not a production-hardened service.


## Author

Peter Nime

GitHub: https://github.com/Pete-Nime
```
