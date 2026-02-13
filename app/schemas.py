from typing import Literal

from pydantic import BaseModel, Field


class ContextLabel(BaseModel):
    targetedness: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    reclaimed: bool = False
    is_quoted: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    rationale_code: str = "heuristic"


class TermScoreRequest(BaseModel):
    term: str = Field(min_length=1, max_length=128)
    contexts: list[str] = Field(default_factory=list, min_length=1)
    locale: str = Field(default="en-US", min_length=2, max_length=16)
    trend_velocity: float = 0.0


class TermScoreResponse(BaseModel):
    term: str
    locale: str
    sample_size: int = Field(ge=1)
    eigen_ctx: float = Field(ge=0.0)
    eigen_graph: float = Field(ge=0.0)
    severity_mean: float = Field(ge=0.0, le=1.0)
    targetedness_mean: float = Field(ge=0.0, le=1.0)
    reclaimed_rate: float = Field(ge=0.0, le=1.0)
    trend_velocity: float
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    band: Literal["monitor", "review", "block"]
    model_version: str
    warnings: list[str] = Field(default_factory=list)


class TermScoreHistoryItem(BaseModel):
    id: int = Field(ge=1)
    term: str
    locale: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    band: Literal["monitor", "review", "block"]
    model_version: str
    created_at: str


class TermHistoryResponse(BaseModel):
    term: str
    count: int = Field(ge=0)
    history: list[TermScoreHistoryItem] = Field(default_factory=list)


class TextScoreRequest(BaseModel):
    text: str = Field(min_length=1)
    candidate_terms: list[str] = Field(default_factory=list, min_length=1)
    locale: str = Field(default="en-US", min_length=2, max_length=16)


class TextTermScore(BaseModel):
    term: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    band: Literal["monitor", "review", "block"]


class TextScoreResponse(BaseModel):
    locale: str
    terms_found: int = Field(ge=0)
    results: list[TextTermScore] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    term: str = Field(min_length=1, max_length=128)
    locale: str = Field(default="en-US", min_length=2, max_length=16)
    feedback_type: Literal[
        "false_positive", "false_negative", "policy_override", "other"
    ]
    proposed_band: Literal["monitor", "review", "block"] | None = None
    proposed_score: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str = Field(default="", max_length=4000)


class FeedbackResponse(BaseModel):
    status: Literal["accepted"]
    feedback_id: int = Field(ge=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    version: str
    labeler_mode: str
    llm_configured: bool
    persistence_enabled: bool
    persistence_available: bool
    database_path: str
    persistence_error: str | None = None
