from fastapi import APIRouter, FastAPI, UploadFile, status, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from routes.schemes.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from controllers.NLPController import NLPController
from application.ingestion import IndexingService
from domain.contracts import AnswerRequest, clamp_limit
from auth.jwt import get_current_user

from models import Response
import logging

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(
    prefix='/api/v1/nlp',
    tags=["api_v1", 'nlp']
)


@nlp_router.post('/index/push/{project_id}')
async def index_project(request: Request, project_id: int, push_request: PushRequest, current_user: str = Depends(get_current_user)):

    project = await ProjectModel(db_client=request.app.db_client).get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": Response.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    result = await IndexingService.from_app(request.app).push_project(
        project, do_reset=push_request.do_reset
    )

    return JSONResponse(content=result)


@nlp_router.get('/index/info/{project_id}')
async def get_project_index_info(request: Request, project_id: int, current_user: str = Depends(get_current_user)):

    project = await ProjectModel(db_client=request.app.db_client).get_project_or_create_one(
        project_id=project_id
    )
    nlp_controller = NLPController.from_app(request.app)

    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": Response.VECTORDB_COLLECTION_RETREIVED.value,
            "collection_info": collection_info
        }
    )


@nlp_router.post('/index/search/{project_id}')
async def search_index_info(request: Request, project_id: int, search_request: SearchRequest):

    project = await ProjectModel(db_client=request.app.db_client).get_project_or_create_one(
        project_id=project_id
    )
    nlp_controller = NLPController.from_app(request.app)

    limit = clamp_limit(search_request.limit)

    results = await nlp_controller.search_vector_db_collection(
        project=project,
        text=search_request.text,
        limit=limit,
        score_threshold=search_request.score_threshold,
        metadata_filter=search_request.metadata_filter,
        retrieval_mode=search_request.retrieval_mode,
        rerank=bool(search_request.rerank),
        expand_query=search_request.expand_query is True,
    )

    # Use `is None` rather than falsy check so a legitimate empty
    # result set (e.g. []) isn't reported as an error.
    if results is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": Response.SEARCH_IN_VECTOR_DB_ERROR.value
            }
        )

    return JSONResponse(
        content={
            "signal": Response.SEARCH_IN_VECTOR_DB_SUCCESS.value,
            "query": search_request.text,
            "expanded_query": nlp_controller.expand_query(search_request.text) if search_request.expand_query is True else search_request.text,
            "query_expanded": search_request.expand_query is True,
            "results": results
        }
    )


@nlp_router.post('/index/answer/{project_id}')
async def answer_index_info(request: Request, project_id: int, search_request: SearchRequest):

    project = await ProjectModel(db_client=request.app.db_client).get_project_or_create_one(
        project_id=project_id
    )
    nlp_controller = NLPController.from_app(request.app)

    use_rerank = search_request.rerank if search_request.rerank is not None else True
    use_query_expansion = search_request.expand_query if search_request.expand_query is not None else True

    answer_request = AnswerRequest(
        project_id=project_id,
        query=search_request.text,
        limit=search_request.limit,
        score_threshold=search_request.score_threshold,
        metadata_filter=search_request.metadata_filter,
        include_sources=search_request.include_sources,
        retrieval_mode=search_request.retrieval_mode,
        rerank=use_rerank,
        expand_query=use_query_expansion,
        verify_claims=bool(search_request.verify_claims),
        conversation_history=[turn.model_dump() for turn in (search_request.conversation_history or [])],
    )

    result = await nlp_controller.answer(project, answer_request)

    pipeline_metadata = {
        'retrieval_mode': search_request.retrieval_mode,
        'rerank_enabled': use_rerank,
        'query_expansion_enabled': use_query_expansion,
        'verify_claims_enabled': bool(search_request.verify_claims),
    }

    # `None` means the generation call itself failed.
    if result.answer is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": Response.RAG_ANSWER_ERROR.value
            }
        )

    # Empty answer means retrieval returned no relevant documents.
    if not result.answer:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.to_api_dict(
                signal=Response.RAG_NO_DOCUMENTS_FOUND.value,
                query=search_request.text,
                expanded_query=result.expanded_query,
                query_expanded=use_query_expansion,
                pipeline_metadata=pipeline_metadata,
            )
        )

    return JSONResponse(
        content=result.to_api_dict(
            signal=Response.RAG_ANSWER_SUCCEED.value,
            query=search_request.text,
            expanded_query=result.expanded_query,
            query_expanded=use_query_expansion,
            pipeline_metadata=pipeline_metadata,
        )
    )


@nlp_router.post('/index/answer/{project_id}/stream')
async def answer_index_info_stream(request: Request, project_id: int, search_request: SearchRequest):

    project = await ProjectModel(db_client=request.app.db_client).get_project_or_create_one(
        project_id=project_id
    )
    nlp_controller = NLPController.from_app(request.app)

    use_rerank = search_request.rerank if search_request.rerank is not None else True
    use_query_expansion = search_request.expand_query if search_request.expand_query is not None else True
    conversation_history = [turn.model_dump() for turn in (search_request.conversation_history or [])]

    async def event_generator():
        async for event in nlp_controller.answer_rag_question_stream(
            project=project,
            query=search_request.text,
            limit=clamp_limit(search_request.limit),
            score_threshold=search_request.score_threshold,
            metadata_filter=search_request.metadata_filter,
            include_sources=search_request.include_sources,
            retrieval_mode=search_request.retrieval_mode,
            rerank=use_rerank,
            expand_query=use_query_expansion,
            verify_claims=bool(search_request.verify_claims),
            conversation_history=conversation_history,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
