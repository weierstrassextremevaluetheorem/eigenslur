# EigenSlur Backend MVP

Backend service that computes an `EigenSlur Score` for a term based on:

- Context covariance largest eigenvalue
- Co-occurrence graph spectral radius
- Context labels (targetedness, severity, reclaimed usage)
- Fusion scoring model
- Persistent score history and feedback ingestion

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Run the API:

```bash
uvicorn app.main:app --reload
```

4. Open frontend:

`http://127.0.0.1:8000/`

5. Open docs:

`http://127.0.0.1:8000/docs`

## Configuration

Set these env vars as needed:

- `EIGENSLUR_ENABLE_PERSISTENCE=true` (default)
- `EIGENSLUR_DATABASE_PATH=data/eigenslur.db`
- `EIGENSLUR_USE_LLM_LABELER=true` (default)
- `EIGENSLUR_OPENAI_API_KEY=...` (required when LLM labeler is enabled)
- `EIGENSLUR_OPENAI_MODEL=gpt-4.1-mini`

When `EIGENSLUR_USE_LLM_LABELER=true` but the API key or optional dependency is unavailable,
the service automatically falls back to the heuristic labeler.

Install optional LLM dependency when enabling LLM labeling:

```bash
pip install -e .[llm]
```

## Example

```bash
curl -X POST http://127.0.0.1:8000/score/term \
  -H "Content-Type: application/json" \
  -d '{
    "term":"example",
    "contexts":[
      "You are an example and nobody wants you here.",
      "They quoted the word example in a documentary.",
      "We reclaimed example in our own community."
    ]
  }'
```

## Additional Endpoints

- `GET /term/{term}/history?limit=50`
- `POST /feedback`

## Notes

- The labeling engine is heuristic by default and designed for replacement with a strict JSON-schema LLM workflow.
- This project is for research and moderation safety tooling, not censorship automation.
