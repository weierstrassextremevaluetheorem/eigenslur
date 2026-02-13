# EigenSlur Improvement Roadmap

## Phase 1: Testing & Quality (Estimated: 2-3 days)

### 1.1 Add Type Checking
- Add `mypy>=1.10.0` to dev dependencies
- Create `mypy.ini` or add to `pyproject.toml`:
  ```
  [tool.mypy]
  python_version = "3.12"
  strict = true
  warn_return_any = true
  ```
- Add `# type: ignore` only where necessary (e.g., third-party libs)

### 1.2 Expand Test Coverage
- Add `pytest-cov` to dev dependencies
- Create test files:
  - `tests/test_labeler.py` - heuristic + LLM labeler edge cases
  - `tests/test_scoring.py` - ScoreService with mocked storage
  - `tests/test_storage.py` - SQLite operations, thread safety
  - `tests/test_text.py` - tokenization edge cases (unicode, emojis)
- Add coverage configuration to `pyproject.toml`:
  ```
  [tool.coverage.run]
  source = ["app"]
  branch = true
  
  [tool.coverage.report]
  fail_under = 80
  ```
- Test edge cases: empty contexts, unicode terms, very long inputs, concurrent requests

### 1.3 Structured Logging
- Add `structlog` or use stdlib `logging`
- Create `app/logging_config.py`:
  - JSON output in production, colored console in dev
  - Log levels configurable via `EIGENSLUR_LOG_LEVEL`
- Instrument key operations: scoring requests, LLM calls, storage ops
- Add request ID tracing for debugging

### 1.4 API Security
- Add rate limiting with `slowapi`:
  ```python
  from slowapi import Limiter
  limiter = Limiter(key_func=get_remote_address)
  ```
- Add API key authentication:
  - `EIGENSLUR_API_KEYS` env var (comma-separated)
  - `X-API-Key` header validation middleware
  - Optional: make auth configurable (off by default for dev)
- Add input validation: max context length, max contexts per request

---

## Phase 2: Feature Enhancements (Estimated: 3-4 days)

### 2.1 Async LLM Labeler
- Convert `OpenAIJSONLabeler` to async:
  ```python
  async def label_batch_async(self, term: str, contexts: list[str], locale: str) -> list[ContextLabel]:
      tasks = [self.label_context_async(term, ctx, locale) for ctx in contexts]
      return await asyncio.gather(*tasks)
  ```
- Add `asyncio` batch semaphore to limit concurrent API calls
- Add retry with exponential backoff for transient failures

### 2.2 Response Caching
- Add caching layer for repeated term scoring:
  ```python
  from cachetools import TTLCache
  
  score_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL
  ```
- Cache key: hash of (term, contexts, locale, model_version)
- Add `/cache/clear` admin endpoint
- Future: Redis backend for distributed caching

### 2.3 Semantic Drift Detection
- Implement `prompt_d_drift.md` logic:
  - Store embedding vectors for each term over time
  - Compare current context embeddings vs historical baseline
  - Detect semantic shift (e.g., term gaining new harmful connotations)
- Add `drift_score` to `TermScoreResponse`
- Add `/term/{term}/drift` endpoint returning drift history

### 2.4 Review Queue System
- Add `review_queue` table to storage:
  ```sql
  CREATE TABLE review_queue (
      id INTEGER PRIMARY KEY,
      term TEXT NOT NULL,
      feedback_id INTEGER REFERENCES feedback(id),
      current_band TEXT,
      proposed_band TEXT,
      status TEXT DEFAULT 'pending',
      resolved_by TEXT,
      resolved_at TEXT
  );
  ```
- Add endpoints:
  - `GET /review/pending?limit=50` - list pending reviews
  - `POST /review/{id}/resolve` - mark as resolved with decision
- Automatically create review items for high-confidence feedback

### 2.5 Weight Calibration System
- Create `app/services/calibration.py`:
  - Logistic regression on feedback-labeled data
  - Periodic recalibration (manual trigger or scheduled)
- Add endpoint: `POST /calibrate` to trigger retraining
- Store calibrated weights in database with version history
- Compare calibrated vs default weights in API response

---

## Phase 3: Infrastructure (Estimated: 2-3 days)

### 3.1 Docker Support
- Create `Dockerfile`:
  ```dockerfile
  FROM python:3.12-slim
  WORKDIR /app
  COPY . .
  RUN pip install -e .[dev,llm]
  EXPOSE 8000
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```
- Create `docker-compose.yml`:
  - API service
  - Redis service (for caching)
  - PostgreSQL service (future)

### 3.2 CI/CD Pipeline
- Create `.github/workflows/ci.yml`:
  ```yaml
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
        - run: pip install -e .[dev]
        - run: ruff check .
        - run: mypy app
        - run: pytest --cov=app --cov-report=xml
        - uses: codecov/codecov-action@v4
  ```

### 3.3 PostgreSQL + pgvector Migration
- Create `app/services/storage_pg.py`:
  - Replace SQLite with async PostgreSQL
  - Add vector column for context embeddings
  - Add similarity search capabilities
- Add migration scripts in `migrations/`
- Support both backends via config flag (transition period)

### 3.4 Observability
- Add `/metrics` endpoint (Prometheus format):
  ```
  eigenslur_scores_total{band="monitor"} 1234
  eigenslur_llm_latency_seconds{model="gpt-4.1-mini"} 0.85
  eigenslur_cache_hits_total 567
  ```
- Add health check details (database connectivity, LLM availability)
- Add structured access logs with request duration

---

## Phase 4: API Improvements (Estimated: 1-2 days)

### 4.1 OpenAPI Enhancements
- Add tags to endpoints:
  ```python
  @app.post("/score/term", tags=["Scoring"])
  @app.get("/health", tags=["System"])
  @app.post("/feedback", tags=["Feedback"])
  ```
- Add response examples in schemas
- Add deprecation warnings for future changes

### 4.2 Bulk Operations
- Add `POST /score/batch` for multiple terms:
  ```python
  class BatchScoreRequest(BaseModel):
      items: list[TermScoreRequest]
  ```
- Use asyncio for parallel processing

### 4.3 Webhook Support
- Add webhook configuration for score changes:
  ```python
  class WebhookConfig(BaseModel):
      url: str
      events: list[Literal["score_computed", "feedback_received"]]
  ```
- Fire webhooks on significant events

---

## Implementation Timeline

```
Week 1: Phase 1 (Testing & Quality)
  - mypy + expanded tests
  - logging
  - rate limiting + auth

Week 2: Phase 2.1-2.3 (Core features)
  - async LLM labeler
  - caching
  - drift detection

Week 3: Phase 2.4-2.5 + Phase 3
  - review queue
  - calibration
  - Docker + CI/CD

Week 4: Phase 3.3-3.4 + Phase 4
  - PostgreSQL migration
  - observability
  - API improvements
```

---

## Configuration Additions

Add to `app/config.py`:
```python
log_level: str = "INFO"
api_keys: list[str] = []
rate_limit_per_minute: int = 60
enable_cache: bool = True
cache_ttl_seconds: int = 300
database_backend: Literal["sqlite", "postgresql"] = "sqlite"
postgres_url: str | None = None
redis_url: str | None = None
```
