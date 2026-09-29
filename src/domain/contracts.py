"""Application-facing contracts for the RAG answer pipeline.

Replaces the 10-argument `answer_rag_question(...)` call and its 10-tuple
return with intention-revealing objects. `AnswerRequest` validates its own
invariants (project id, limit bounds) so routes and repositories never see
illegal values — this is where the `info/-1` and `limit=-5` 500s die.
"""
from dataclasses import dataclass, field
from typing import Optional

from domain.confidence import ConfidenceScorer
from domain.exceptions import ValidationError

MIN_LIMIT = 1
MAX_LIMIT = 50


def clamp_limit(limit, default: int = 5) -> int:
    """Clamp retrieval limits into [1, 50].

    Out-of-range numbers are clamped (they previously caused 500s);
    non-numeric garbage is a client error and raises ValidationError.
    """
    if limit is None:
        return default
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise ValidationError(f"invalid limit: {limit!r}")
    return max(MIN_LIMIT, min(MAX_LIMIT, value))


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
        self.limit = clamp_limit(self.limit)
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

    def to_api_dict(self, *, signal: str, query: str, expanded_query: str,
                    query_expanded: bool, pipeline_metadata: dict) -> dict:
        """Wire format identical to the pre-refactor answer endpoints.

        Generation artefacts (prompt, history, retrieval flags) are only
        present when an answer was actually generated.
        """
        quality = self.quality or {}
        payload = {
            'signal': signal,
            'answer': self.answer or '',
            'sources': self.sources or [],
            'risk_assessment': self.risk_assessment,
            'confidence': self.confidence,
            'confidence_summary': ConfidenceScorer._build_confidence_summary(self.confidence or {}),
            'evidence_panel': self.evidence_panel,
            'pipeline_metadata': pipeline_metadata,
            'quality': quality,
            'citations': quality.get('citations', []),
            'unsupported_claims': quality.get('unsupported_claims', []),
            'unsupported_claim_rate': quality.get('unsupported_claim_rate', 0.0),
            'citation_faithfulness': quality.get('citation_faithfulness', 0.0),
            'disclaimer': self.disclaimer,
            'query': query,
            'expanded_query': expanded_query,
            'query_expanded': query_expanded,
        }
        if self.answer:
            payload['full_prompt'] = self.full_prompt
            payload['chat_history'] = self.chat_history
            payload['retrieval_mode'] = pipeline_metadata.get('retrieval_mode')
            payload['rerank'] = pipeline_metadata.get('rerank_enabled')
        return payload
