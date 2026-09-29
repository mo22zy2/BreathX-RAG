"""Application services: upload / process / push use-cases.

Moved verbatim out of routes/data.py and routes/nlp.py so routes only do
request validation, service calls, and response mapping. Wire behavior
(status codes, signals, payload shapes) is preserved exactly — failures
raise domain exceptions that the API exception handler translates.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import aiofiles
from tqdm.auto import tqdm

from controllers import DataController, ProjectController, NLPController
from controllers.ProcessController import ProcessController
from domain.exceptions import ValidationError
from helpers.config import get_settings
from models.AssetModel import AssetModel
from models.ChunkModel import ChunkModel
from models import Response
from models.db_schemas import Asset, DataChunk
from models.enums.AssetTypeEnum import AssetTypeEnum

logger = logging.getLogger("uvicorn.error")


@dataclass
class ProcessOptions:
    """Service-layer copy of the process request (decoupled from the API schema)."""
    file_id: Optional[str] = None
    chunk_size: Optional[int] = None
    overlap_size: Optional[int] = None
    do_reset: Optional[int] = 0
    chunking_method: Optional[str] = None
    document_name: Optional[str] = None
    source_url: Optional[str] = None
    org: Optional[str] = None


class UploadService:
    """File upload: validation, unique paths, streaming save, asset record."""

    def __init__(self, asset_model: AssetModel,
                 data_controller: DataController = None,
                 project_controller: ProjectController = None):
        self.asset_model = asset_model
        self.data_controller = data_controller or DataController()
        self.project_controller = project_controller or ProjectController()

    @classmethod
    def from_app(cls, app) -> "UploadService":
        return cls(asset_model=AssetModel(db_client=app.db_client))

    async def validate_upload(self, file):
        return await self.data_controller.validate_upload_file(file=file)

    def build_filepath(self, original_file_name: str, project_id: int):
        return self.data_controller.generate_unique_filepath(
            original_file_name=original_file_name,
            project_id=project_id,
        )

    async def save_upload(self, file, file_path: str, max_size_bytes: int, chunk_size: int) -> int:
        total_bytes = 0
        try:
            async with aiofiles.open(file_path, "wb") as f:
                while chunk := await file.read(chunk_size):
                    total_bytes += len(chunk)
                    if total_bytes > max_size_bytes:
                        await f.close()
                        os.remove(file_path)
                        raise ValidationError(Response.FILE_SIZE_EXCEEDED.value)
                    await f.write(chunk)
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error while uploading file : {e}")
            raise ValidationError(Response.FILE_VALIDATED_FALIED.value)
        return total_bytes

    async def register_asset(self, project_id: int, file_id: str, file_path: str,
                             document_name: str = None, source_url: str = None,
                             org: str = None):
        asset_config = {}
        if document_name:
            asset_config["document_name"] = document_name
        if source_url:
            asset_config["source_url"] = source_url
        if org:
            asset_config["org"] = org

        asset_resource = Asset(
            asset_project_id=project_id,
            asset_name=file_id,
            asset_type=AssetTypeEnum.FILE.value,
            asset_size=str(os.path.getsize(file_path)),
            asset_config=asset_config or None,
        )
        return await self.asset_model.create_asset(asset=asset_resource)


class ProcessService:
    """Document processing: chunking files into the relational store."""

    def __init__(self, chunk_model: ChunkModel, asset_model: AssetModel,
                 answering: NLPController, vectordb_client, settings=None):
        self.chunk_model = chunk_model
        self.asset_model = asset_model
        self.answering = answering
        self.vectordb_client = vectordb_client
        self.settings = settings or get_settings()

    @classmethod
    def from_app(cls, app) -> "ProcessService":
        return cls(
            chunk_model=ChunkModel(db_client=app.db_client),
            asset_model=AssetModel(db_client=app.db_client),
            answering=NLPController.from_app(app),
            vectordb_client=app.vectordb_client,
        )

    async def process_project(self, project, options: ProcessOptions) -> dict:
        project_file_ids = {}
        asset_configs = {}

        if options.file_id:
            asset_record = await self.asset_model.get_asset_record(
                asset_project_id=project.project_id,
                asset_name=options.file_id
            )

            if asset_record is None:
                raise ValidationError(Response.FILE_ID_ERROR.value)
            project_file_ids = {
                asset_record.asset_id: asset_record.asset_name
            }
            asset_configs = {
                asset_record.asset_id: asset_record.asset_config or {}
            }
        else:
            project_files = await self.asset_model.get_all_project_assets(
                asset_project_id=project.project_id,
                asset_type=AssetTypeEnum.FILE.value
            )
            project_file_ids = {
                record.asset_id: record.asset_name
                for record in project_files
            }
            asset_configs = {
                record.asset_id: record.asset_config or {}
                for record in project_files
            }

        if len(project_file_ids) == 0:
            raise ValidationError(Response.NO_FILES_ERROR.value)

        process_controller = ProcessController(project.project_id)

        if options.do_reset == 1:
            collection_name = self.answering.create_collection_name(project_id=project.project_id)
            _ = await self.vectordb_client.delete_collection(collection_name=collection_name)
            _ = await self.chunk_model.delete_chunk_by_project_id(project_id=project.project_id)

        no_records = 0
        no_files = 0

        for asset_id, file_id in project_file_ids.items():
            file_content = process_controller.get_file_content(file_id=file_id)

            if file_content is None:
                logger.error(f"Error while Processing File {file_id}")
                continue

            file_chunks = process_controller.process_file_content(
                file_content=file_content,
                file_id=file_id,
                chunk_size=options.chunk_size,
                chunk_overlap=options.overlap_size,
                method=options.chunking_method or self.settings.DEFAULT_CHUNKING_METHOD
            )

            if file_chunks is None or len(file_chunks) == 0:
                raise ValidationError(Response.FILE_PROCESSING_FALIED.value)

            asset_config = asset_configs.get(asset_id, {})
            provenance = {
                "document_name": asset_config.get("document_name")
                or options.document_name
                or file_id,
                "source_url": asset_config.get("source_url")
                or options.source_url
                or "",
                "org": asset_config.get("org") or options.org or "",
            }

            file_chunks_records = [
                DataChunk(
                    chunk_text=chunk.page_content,
                    chunk_metadata=json.dumps({
                        **chunk.metadata,
                        "asset_id": asset_id,
                        "file_name": file_id,
                        "project_id": project.project_id,
                        "chunk_order": i + 1,
                        **provenance,
                    }),
                    chunk_order=i + 1,
                    chunk_project_id=project.project_id,
                    chunk_asset_id=asset_id
                )
                for i, chunk in enumerate(file_chunks)]

            no_records += await self.chunk_model.insert_many_chunks(chunks=file_chunks_records)
            no_files += 1

        return {
            "signal": Response.FILE_PROCESSING_SUCCEED.value,
            "inserted_chunks": no_records,
            "processed_files": no_files
        }


class IndexingService:
    """Vector indexing: pushing stored chunks into the vector database."""

    def __init__(self, chunk_model: ChunkModel, vectordb_client, answering: NLPController):
        self.chunk_model = chunk_model
        self.vectordb_client = vectordb_client
        self.answering = answering

    @classmethod
    def from_app(cls, app) -> "IndexingService":
        return cls(
            chunk_model=ChunkModel(db_client=app.db_client),
            vectordb_client=app.vectordb_client,
            answering=NLPController.from_app(app),
        )

    async def push_project(self, project, do_reset: int = 0) -> dict:
        collection_name = self.answering.create_collection_name(project_id=project.project_id)

        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.answering.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        if do_reset:
            await self.chunk_model.reset_chunk_indexed(project_id=project.project_id)

        unindexed_count = await self.chunk_model.get_unindexed_chunk_count(project_id=project.project_id)

        if unindexed_count == 0:
            return {
                "signal": Response.INSERT_INTO_VECTOR_DB_SUCCESS.value,
                "inserted_items_count": 0,
                "message": "All chunks already indexed"
            }

        pbar = tqdm(
            total=unindexed_count,
            desc="Vector Indexing",
            position=0
        )

        has_records = True
        page_no = 1
        inserted_items_count = 0

        try:
            while has_records:
                page_chunk = await self.chunk_model.get_unindexed_chunks(
                    project_id=project.project_id,
                    page_no=page_no
                )

                if not page_chunk:
                    has_records = False
                    break

                page_no += 1

                chunk_ids = [c.chunk_id for c in page_chunk]

                is_inserted = await self.answering.index_into_vector_db(
                    project=project,
                    chunks=page_chunk,
                    chunk_ids=chunk_ids
                )

                if not is_inserted:
                    raise ValidationError(
                        Response.INSERT_INTO_VECTOR_DB_ERROR.value,
                        extra={
                            "inserted_items_count": inserted_items_count,
                            "remaining_items": unindexed_count - inserted_items_count,
                        },
                    )

                await self.chunk_model.mark_chunks_indexed(chunk_ids)

                pbar.update(len(page_chunk))
                inserted_items_count += len(page_chunk)
        finally:
            pbar.close()

        return {
            "signal": Response.INSERT_INTO_VECTOR_DB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
