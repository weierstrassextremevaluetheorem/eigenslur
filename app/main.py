from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.prompt_templates import load_prompt_templates
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    TermHistoryResponse,
    TermScoreRequest,
    TermScoreResponse,
    TextScoreRequest,
    TextScoreResponse,
)
from app.services.fusion import FusionEngine
from app.services.labeler import HeuristicLabeler
from app.services.llm_labeler import OpenAIJSONLabeler
from app.services.prompting import load_labeler_prompts
from app.services.scoring import ScoreService
from app.services.storage import SQLiteStore

settings = get_settings()
static_dir = Path(__file__).resolve().parent / "static"

storage = SQLiteStore(settings.database_path) if settings.enable_persistence else None

llm_configured = bool(settings.openai_api_key)
llm_enabled = settings.use_llm_labeler and llm_configured
labeler_mode = "heuristic_v1"

if llm_enabled:
    try:
        labeler = OpenAIJSONLabeler(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
            prompts=load_labeler_prompts(),
        )
        labeler_mode = "openai_json_v1"
    except RuntimeError:
        labeler = HeuristicLabeler()
        labeler_mode = "heuristic_v1"
else:
    labeler = HeuristicLabeler()

score_service = ScoreService(
    embedding_dim=settings.embedding_dim,
    labeler=labeler,
    fusion_engine=FusionEngine(
        review_threshold=settings.fusion_threshold_review,
        block_threshold=settings.fusion_threshold_block,
    ),
    storage=storage,
)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for EigenSlur spectral risk scoring.",
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        labeler_mode=labeler_mode,
        llm_configured=llm_configured,
    )


@app.post("/score/term", response_model=TermScoreResponse)
def score_term(payload: TermScoreRequest) -> TermScoreResponse:
    return score_service.score_term(
        term=payload.term,
        contexts=payload.contexts,
        locale=payload.locale,
        trend_velocity=payload.trend_velocity,
    )


@app.post("/score/text", response_model=TextScoreResponse)
def score_text(payload: TextScoreRequest) -> TextScoreResponse:
    return score_service.score_text(
        text=payload.text,
        candidate_terms=payload.candidate_terms,
        locale=payload.locale,
    )


@app.get("/term/{term}/history", response_model=TermHistoryResponse)
def term_history(
    term: str, limit: int = Query(default=50, ge=1, le=200)
) -> TermHistoryResponse:
    return score_service.get_term_history(term=term, limit=limit)


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        return score_service.submit_feedback(payload)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/prompts")
def get_prompt_templates() -> dict[str, str]:
    return load_prompt_templates()
