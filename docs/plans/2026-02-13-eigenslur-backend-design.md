# EigenSlur Backend Design

## Objective

Ship an MVP backend that computes a term-level risk score for moderation research:

- Spectral context signal (`lambda_ctx`)
- Spectral graph signal (`lambda_graph`)
- Context-level harm labels
- Fusion score and confidence band (`monitor`, `review`, `block`)

## MVP Components

1. API layer (`app/main.py`)
2. Spectral engine (`app/services/spectral.py`)
3. Labeling engine (`app/services/labeler.py`)
4. Fusion engine (`app/services/fusion.py`)
5. Scoring orchestrator (`app/services/scoring.py`)
6. Prompt templates (`app/prompts/*.md`)

## Data Contracts

Primary API contracts are defined in `app/schemas.py`:

- `TermScoreRequest` / `TermScoreResponse`
- `TextScoreRequest` / `TextScoreResponse`
- `HealthResponse`

## Scoring Formula

`score = sigmoid(b0 + b1*f(lambda_graph) + b2*f(lambda_ctx) + b3*severity + b4*targetedness - b5*reclaimed + b6*trend)`

where `f(x) = tanh(log(1+x))` for nonnegative compression.

## Prompt Plan

Prompt templates are versioned under `app/prompts/`:

1. `prompt_a_usage.md` for targetedness / quotation / reclamation
2. `prompt_b_severity.md` for harm severity
3. `prompt_c_disambiguation.md` for classifier conflicts
4. `prompt_d_drift.md` for semantic drift and variant detection

## Next Build Items

1. Replace heuristic labeler with strict JSON-schema LLM client.
2. Persist scoring history and drift windows in PostgreSQL + pgvector.
3. Add review queue and human feedback endpoint.
4. Calibrate fusion weights using labeled evaluation data.
