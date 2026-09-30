
from pydantic import BaseModel, Field


class PushRequest(BaseModel):
    do_reset: int | None=0


class ConversationTurn(BaseModel):
    question: str
    answer: str | None = None


class SearchRequest(BaseModel):
    text:str
    limit:int | None=5
    # Prior turns in this session, oldest first, so follow-up questions
    # ("what about someone older?") can be understood in context.
    conversation_history: list[ConversationTurn] | None = None
    score_threshold: float | None = None
    # Swagger UI auto-fills a bare `Optional[dict]` with a placeholder like
    # {"additionalProp1": {}} — an explicit null example stops that, since
    # submitting the placeholder as-is filters on a metadata key that no
    # chunk has and silently returns zero results.
    metadata_filter: dict | None = Field(default=None, examples=[None])
    include_sources: bool | None = True
    # "vector" | "keyword" | "hybrid"
    retrieval_mode: str | None = "hybrid"
    # Post-retrieval rerank. None = auto (enabled for /answer, disabled for /search)
    rerank: bool | None = None
    # Expand asthma abbreviations/synonyms before retrieval.
    expand_query: bool | None = None
    # Run deterministic citation and claim-support checks on generated answers.
    verify_claims: bool | None = True
