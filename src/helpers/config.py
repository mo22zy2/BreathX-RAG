import json
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings (BaseSettings):
    APP_NAME:str
    APP_VERSION:str
    ENVIRONMENT:str
    FILE_ALLOWED_TYPES:list
    FILE_MAX_SIZE:int
    FILE_DEFAULT_CHUNK_SIZE:int
    
    
    MONGODB_URL:str | None=None
    MONGODB_DATABASE:str | None=None
    
    POSTGRES_USERNAME:str
    POSTGRES_PASSWORD:str
    POSTGRES_HOST:str
    POSTGRES_PORT:int
    POSTGRES_MAIN_DATABASE:str
    
    GENERATION_BACKEND:str
    EMBEDDING_BACKEND:str

    OPENAI_API_KEY:str | None=None
    OPENAI_BASE_URL:str | None=None


    COHERE_API_KEY:str | None=None

    GENERATION_MODEL_ID_LIST:str | None=None
    GENERATION_MODEL_ID:str | None=None
    EMBEDDING_MODEL_ID:str | None=None
    EMBEDDING_MODEL_SIZE:int | None=None


    INPUT_DEFAULT_MAX_CHARS:int | None=None
    GENERATION_DEFAULT_MAX_TOKENS:int | None=None
    GENERATION_DEFAULT_TEMPERATURE:float=0.8

    RETRIEVAL_TOP_K:int=8
    ANSWER_TOP_K:int=8
    RETRIEVAL_SCORE_THRESHOLD:float=0.0
    ANSWER_MIN_TOP_SCORE:float=0.0
    ANSWER_MIN_EVIDENCE_COUNT:int=1
    ANSWER_MAX_VERIFICATION_REGENERATIONS:int=1
    MAX_CONTEXT_CHARS:int=12000
    EMBEDDING_BATCH_SIZE:int=64
    VECTOR_INSERT_BATCH_SIZE:int=64

    # Retrieval optimization
    RERANK_BACKEND:str="COHERE"
    RERANK_MODEL_ID:str="rerank-v3.5"
    RERANK_TOP_K:int=5
    HYBRID_FUSION:str="rrf"
    HYBRID_RRF_K:int=60

    # Clinical ingestion / section-aware chunking
    DEFAULT_CHUNKING_METHOD:str="section"
    CHUNK_MIN_TOKENS:int | None=400
    CHUNK_MAX_TOKENS:int | None=800
    CHUNK_CHARS_PER_TOKEN:float=4.0
    USE_TIKTOKEN_CHUNKING:bool=False
    TIKTOKEN_ENCODING:str="cl100k_base"
    RUNNING_HEADER_MIN_PAGE_FRACTION:float=0.5
    STRIP_PATTERNS:list[str] | None=None
    SECTION_HEADER_PATTERNS:list[str] | None=None

    @field_validator("STRIP_PATTERNS", "SECTION_HEADER_PATTERNS", mode="before")
    @classmethod
    def _parse_pattern_list(cls, value):
        # pydantic-settings tries JSON first; regex escapes like \s break that,
        # so fall back to a comma-separated list.
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return [
                item.strip().strip('"').strip("'")
                for item in value.split(",")
                if item.strip()
            ]
        return value


    VECTOR_DB_BACKEND_LITERAL:list[str] | None=None
    VECTOR_DB_BACKEND:str
    VECTOR_DB_PATH:str
    VECTOR_DB_DISTANCE_METHOD:str
    VECTOR_DB_PGVEC_INDEX_THRESHOLD:int
    VECTOR_DB_INDEX_TYPE:str
    
    DEFAULT_LANGUAGE:str

    ADMIN_USERNAME:str = "admin"
    ADMIN_PASSWORD_HASH:str | None = None

    SECRET_JWT_KEY:str



    model_config = SettingsConfigDict(env_file=Path(__file__).parent.parent / '.env')


@lru_cache
def get_settings():
    """Process-wide cached settings. src/.env edits require a restart
    (safety_config.json still hot-reloads independently)."""
    return Settings()
