FROM ghcr.io/astral-sh/uv:0.6.14-python3.10-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libavif-dev \
    libjpeg-dev \
    libpng-dev \
    gcc \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .

RUN uv pip install -r requirements.txt --system

COPY src/ .

COPY utils/ ./utils/

RUN mkdir -p /app/models/db_schemas/breathx_rag

COPY docker/breathX/alembic.ini /app/models/db_schemas/breathx_rag/alembic.ini

COPY docker/breathX/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]

CMD [ "uvicorn" ,"main:app","--host", "0.0.0.0","--port","8000","--workers","4"]

