from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from domain.exceptions import BreathXError
from helpers.config import get_settings
from models.db_schemas.startup_checks import ensure_chunks_indexed_column
from routes import base, data, login, nlp
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.rerank.RerankProviderFactory import RerankProviderFactory
from stores.templates.template_parser import Template_Parser
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from utils.metrices import setup_metrics

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    postgres_connection=f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    app.db_engine=create_async_engine(postgres_connection)
    app.db_client=sessionmaker(
        app.db_engine,class_=AsyncSession,expire_on_commit=False
    )

    # app.mongo_connection = AsyncIOMotorClient(settings.MONGODB_URL)
    # app.db_client = app.mongo_connection[settings.MONGODB_DATABASE]

    llm_provider_factory = LLMProviderFactory(config=settings)
    vectordb_provider_factory= VectorDBProviderFactory(config=settings,db_client=app.db_client)
    rerank_provider_factory = RerankProviderFactory(config=settings)

    app.generation_client=llm_provider_factory.create_provider(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    app.embedding_client=llm_provider_factory.create_provider(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )
    # Rerank client
    app.rerank_client=rerank_provider_factory.create_provider(provider=settings.RERANK_BACKEND)

    app.vectordb_client=vectordb_provider_factory.create(
        provider=settings.VECTOR_DB_BACKEND
    )

    await app.vectordb_client.connect()

    await ensure_chunks_indexed_column(app.db_engine)

    app.template_parser=Template_Parser(
        language=settings.DEFAULT_LANGUAGE,
        default_language=settings.DEFAULT_LANGUAGE
    )

    yield

    await app.db_engine.dispose()
    await app.vectordb_client.disconnect()


settings = get_settings()

is_production = settings.ENVIRONMENT.lower() == "prod"

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None if is_production else "/openapi.json",
)

setup_metrics(app)


@app.exception_handler(BreathXError)
async def breathx_error_handler(request, exc: BreathXError):
    content = {"signal": exc.signal}
    content.update(exc.extra)
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
    )


app.add_middleware(
    CORSMiddleware,
    # allow_origins takes literal origin strings, not regexes — the regex
    # belongs only in allow_origin_regex (any localhost/127.0.0.1 port).
    allow_origin_regex=r"^(https?://)?(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)
app.include_router(login.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if not is_production:
    # Self-hosted Swagger UI (vendored in src/static/docs/): the default
    # docs page pulls JS/CSS from a public CDN, which renders blank on
    # networks without CDN access. Same ENVIRONMENT gate as before.
    @app.get("/docs", include_in_schema=False)
    async def selfhosted_swagger_ui():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{settings.APP_NAME} - Swagger UI",
            swagger_js_url="/static/docs/swagger-ui-bundle.js",
            swagger_css_url="/static/docs/swagger-ui.css",
        )


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    return HTMLResponse("<h1>BreathX RAG</h1><p>Frontend not built yet. Use <a href='/docs'>Swagger</a>.</p>")