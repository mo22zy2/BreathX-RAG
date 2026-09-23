APP_NAME="mini-RAG"
APP_VERSION="0.1"

# ============================================================
# File upload
# ============================================================

FILE_ALLOWED_TYPES=["text/plain","application/pdf"]
FILE_MAX_SIZE=25                         # MB
FILE_DEFAULT_CHUNK_SIZE=512000


# ============================================================
# MongoDB
# ============================================================

# MONGODB_URL="mongodb://admin:admin@localhost:27017"
# MONGODB_DATABASE="mini-rag"


# ============================================================
# PostgreSQL Database
# ============================================================

POSTGRES_USERNAME="postgres"
POSTGRES_PASSWORD=""
POSTGRES_HOST="pgvector"
POSTGRES_PORT=5432
POSTGRES_MAIN_DATABASE="breathX"
# ============================================================
# LLM Backends
# ============================================================

GENERATION_BACKEND="COHERE"
EMBEDDING_BACKEND="COHERE"


# ============================================================
# OpenAI / OpenAI-compatible API
# ============================================================

# OPENAI_API_KEY="<your-openai-or-compatible-api-key>"

OPENAI_API_KEY=""

# Your current OpenAI-compatible endpoint (example: ngrok tunnel)
# OPENAI_BASE_URL="https://<your-tunnel>.ngrok-free.dev/v1"

OPENAI_BASE_URL="https://api.jina.ai/v1"


# For direct OpenAI:
# OPENAI_BASE_URL="https://api.openai.com/v1"


# ============================================================
# Cohere
# ============================================================

COHERE_API_KEY=""


# ============================================================
# Generation Models
# ============================================================

GENERATION_MODEL_ID_LIST=["command-a-03-2025","gemma2:9b","command-r-plus","command-r7b-12-2024"]
GENERATION_MODEL_ID="command-a-03-2025"   # command-r-plus was +2-3s slower per question


# ============================================================
# Embedding Models
# ============================================================

EMBEDDING_MODEL_ID_LIST=["embed-multilingual-light-v3.0","qwen3-embedding:8b","mxbai-embed-large:335m","bge-m3","qwen3-embedding:0.6b","jina-embeddings-v5-omni-nano"]

EMBEDDING_MODEL_ID="embed-multilingual-light-v3.0"

# Dimensions:
# bge-m3=1024
# embed-multilingual-light-v3.0=384
# qwen3-embedding:8b=4096
# mxbai-embed-large:335m=1024

EMBEDDING_MODEL_SIZE=384


# ============================================================
# Generation Parameters
# ============================================================

INPUT_DEFAULT_MAX_CHARS=16000
GENERATION_DEFAULT_MAX_TOKENS=500
GENERATION_DEFAULT_TEMPERATURE=0.1

# Prevent the verification loop from triggering a full 2nd generation pass.
# Citation & claim checks still RUN and are reported in the response — only the
# retry/regeneration is skipped. This removes the unpredictable +3-4s tail.
ANSWER_MAX_VERIFICATION_REGENERATIONS=0


# ============================================================
# RAG Retrieval and Indexing
# ============================================================

RETRIEVAL_TOP_K=20
ANSWER_TOP_K=8
RETRIEVAL_SCORE_THRESHOLD=0.35
ANSWER_MIN_TOP_SCORE=0.45
ANSWER_MIN_EVIDENCE_COUNT=1
MAX_CONTEXT_CHARS=12000
EMBEDDING_BATCH_SIZE=8
VECTOR_INSERT_BATCH_SIZE=64

# Retrieval optimization
RERANK_BACKEND="COHERE"
RERANK_MODEL_ID="rerank-v3.5"
RERANK_TOP_K=10
HYBRID_FUSION="rrf"
HYBRID_RRF_K=30


# ============================================================
# Clinical Ingestion / Section-Aware Chunking
# ============================================================

# Default splitter used by /api/v1/data/process (recursive | simple | semantic | section)
DEFAULT_CHUNKING_METHOD="section"

# Token budget per chunk (section splitter)
CHUNK_MIN_TOKENS=400
CHUNK_MAX_TOKENS=800

# Fallback tokens estimator: chars per token (used unless USE_TIKTOKEN_CHUNKING=true)
CHUNK_CHARS_PER_TOKEN=4.0
USE_TIKTOKEN_CHUNKING=false
TIKTOKEN_ENCODING="cl100k_base"

# Lines repeated on >= this fraction of pages are treated as running headers/footers and stripped
RUNNING_HEADER_MIN_PAGE_FRACTION=0.5

# Extra regexes for stripping artifacts from page text (JSON list or comma-separated)
STRIP_PATTERNS=[]

# Extra regexes (in addition to built-in numbered / ALL-CAPS detection) for section headings.
# Avoid backslashes in regex (breaks JSON env parsing); use e.g. [0-9] instead of \d.
SECTION_HEADER_PATTERNS="^CHAPTER [0-9]+,^GINA,^NICE"

# ============================================================
# Vector Database
# ============================================================

VECTOR_DB_BACKEND_LITERAL=["QDRANT","PGVECTOR"]
VECTOR_DB_BACKEND="PGVECTOR"
VECTOR_DB_PATH="qdrant_db"
VECTOR_DB_DISTANCE_METHOD="cosine"
VECTOR_DB_PGVEC_INDEX_THRESHOLD=150
VECTOR_DB_INDEX_TYPE="HNSW"

# ============================================================
# Templates
# ============================================================

DEFAULT_LANGUAGE="en"
