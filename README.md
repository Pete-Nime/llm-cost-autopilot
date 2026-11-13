
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

## Current Limitations

- The V2 classifier requires independent evaluation on unseen prompts.
- Routing quality depends on the classifier and configured model policies.
- Cost estimates may differ from actual provider billing.
- The project is a portfolio implementation, not a production-hardened service.

## Author

Peter Nime

GitHub: https://github.com/Pete-Nime
```
