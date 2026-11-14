# atlas-ai-service

Python gRPC AI service for ATLAS (`atlas.ai.v1.AIService`), called by the Go backend
([atlas-backend-core](https://github.com/Elioth-Gray/atlas-backend-core)):

- **ProcessFeedback (Path A):** LLM opinion extraction, local E5 embeddings, and issue
  matching with an LLM merge judge.
- **ProcessConsultationTurn (Path B):** guided consultation with grounded, validated solutions.

The LLM provider is OpenAI, accessed through LangChain with strict JSON-schema structured
output (`src/atlas_ai/llm.py`).

| Document | Purpose |
|---|---|
| [docs/local-testing.md](docs/local-testing.md) | Run AI + Go + frontend locally with Docker and test manually |
| [docs/deploy-checklist.md](docs/deploy-checklist.md) | Pilot/staging deploy checklist (secrets, network, limits, data) |
| [docs/prd_path_a_feedback_intelligence.md](docs/prd_path_a_feedback_intelligence.md) | Path A PRD |
| [docs/prd_python_grpc_service_nlp_ml_ai_consultation.md](docs/prd_python_grpc_service_nlp_ml_ai_consultation.md) | Service PRD |
| [docs/go-integration-audit.md](docs/go-integration-audit.md) | Go contract audit and integration runs |

Quick start (service only): copy `.env.example` to `.env`, fill `GRPC_AUTH_TOKEN`,
`OPENAI_API_KEY`, `CONSULTATION_MODEL` and `FEEDBACK_MODEL`, then run `docker compose up --build`.

Tests run inside the dev image:

```
docker compose build atlas-ai
docker run --rm --network none -v "${PWD}:/app" -w /app atlas-ai-service:dev python -m pytest -q
```
