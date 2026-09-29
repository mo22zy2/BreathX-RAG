"""Application-facing contracts for the RAG answer pipeline.

Replaces the 10-argument `answer_rag_question(...)` call and its 10-tuple
return with intention-revealing objects. `AnswerRequest` validates its own
invariants (project id, limit bounds) so routes and repositories never see
illegal values — this is where the `info/-1` and `limit=-5` 500s die.
"""
from dataclasses import dataclass, field
from typing import Optional

from domain.exceptions import ValidationError

MIN_LIMIT = 1
MAX_LIMIT = 50


@dataclass
class AnswerRequest:
    project_id: int
    query: str
    limit: int = 5
    score_threshold: Optional[float] = None
    metadata_filter: Optional[dict] = None
    include_sources: bool = True
    retrieval_mode: str = "hybrid"
    rerank: Optional[bool] = None
    rerank_top_k: Optional[int] = None
    expand_query: Optional[bool] = None
    verify_claims: bool = True
    conversation_history: Optional[list] = field(default=None)

    def __post_init__(self):
        if self.project_id is None or int(self.project_id) < 1:
            raise ValidationError(f"invalid project_id: {self.project_id!r}")
        self.project_id = int(self.project_id)
        try:
            limit = int(self.limit)
        except (TypeError, ValueError):
            raise ValidationError(f"invalid limit: {self.limit!r}")
        self.limit = max(MIN_LIMIT, min(MAX_LIMIT, limit))
        self.query = self.query or ""


@dataclass
class AnswerResult:
    answer: Optional[str]
    full_prompt: Optional[str]
    chat_history: Optional[list]
    sources: list
    risk_assessment: dict
    confidence: dict
    quality: dict
    disclaimer: str
    evidence_panel: dict
    expanded_query: str
