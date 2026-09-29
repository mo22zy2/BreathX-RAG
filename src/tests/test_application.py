"""
Unit tests for application services (upload / process / push).

All externals are fakes — no database, no vector DB, no LLM.
Run from src/:
    pytest tests/test_application.py -v
"""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from application.ingestion import IndexingService, ProcessService, UploadService
from domain.contracts import AnswerResult
from domain.exceptions import ValidationError
from models import Response


def _project(pid=1):
    return SimpleNamespace(project_id=pid)


class TestIndexingService:

    def _service(self, unindexed_count=0, chunks=None, inserted=True):
        chunk_model = MagicMock()
        chunk_model.get_unindexed_chunk_count = AsyncMock(return_value=unindexed_count)
        chunk_model.get_unindexed_chunks = AsyncMock(return_value=chunks or [])
        chunk_model.reset_chunk_indexed = AsyncMock()
        chunk_model.mark_chunks_indexed = AsyncMock()
        vectordb = MagicMock()
        vectordb.create_collection = AsyncMock()
        answering = MagicMock()
        answering.create_collection_name.return_value = "collection_1"
        answering.embedding_client.embedding_size = 1024
        answering.index_into_vector_db = AsyncMock(return_value=inserted)
        return IndexingService(chunk_model=chunk_model, vectordb_client=vectordb, answering=answering)

    def test_all_indexed_short_circuits(self):
        svc = self._service(unindexed_count=0)
        result = asyncio.run(svc.push_project(_project(), do_reset=0))
        assert result["signal"] == Response.INSERT_INTO_VECTOR_DB_SUCCESS.value
        assert result["inserted_items_count"] == 0
        assert result["message"] == "All chunks already indexed"

    def test_failed_insert_raises_with_counts(self):
        chunk = SimpleNamespace(chunk_id=7)
        svc = self._service(unindexed_count=1, chunks=[chunk], inserted=False)
        with pytest.raises(ValidationError) as exc_info:
            asyncio.run(svc.push_project(_project(), do_reset=0))
        assert exc_info.value.signal == Response.INSERT_INTO_VECTOR_DB_ERROR.value
        assert exc_info.value.extra["inserted_items_count"] == 0
        assert exc_info.value.extra["remaining_items"] == 1

    def test_do_reset_resets_index_flags(self):
        svc = self._service(unindexed_count=0)
        asyncio.run(svc.push_project(_project(), do_reset=1))
        svc.chunk_model.reset_chunk_indexed.assert_called_once()


class TestUploadService:

    def _upload_file(self, payload: bytes):
        data = [payload]

        class _FakeUpload:
            content_type = "application/pdf"

            async def read(self, size=-1):
                if data:
                    return data.pop(0)
                return b""

        return _FakeUpload()

    def test_save_upload_writes_bytes(self, tmp_path):
        svc = UploadService(asset_model=MagicMock())
        target = str(tmp_path / "f.pdf")
        total = asyncio.run(svc.save_upload(self._upload_file(b"hello"), target, 100, 2))
        assert total == 5
        assert Path(target).read_bytes() == b"hello"

    def test_save_upload_rejects_oversize_and_cleans_up(self, tmp_path):
        svc = UploadService(asset_model=MagicMock())
        target = str(tmp_path / "big.pdf")
        with pytest.raises(ValidationError) as exc_info:
            asyncio.run(svc.save_upload(self._upload_file(b"123456"), target, 3, 10))
        assert exc_info.value.signal == Response.FILE_SIZE_EXCEEDED.value
        assert not Path(target).exists()

    def test_register_asset_shapes_record(self, tmp_path):
        asset_model = MagicMock()
        created = {}

        async def _create(asset):
            created.update(asset_name=asset.asset_name, asset_config=asset.asset_config,
                           asset_size=asset.asset_size, asset_type=asset.asset_type,
                           asset_project_id=asset.asset_project_id)
            return SimpleNamespace(asset_id=9)

        asset_model.create_asset.side_effect = _create
        svc = UploadService(asset_model=asset_model)
        target = tmp_path / "doc.pdf"
        target.write_bytes(b"1234")
        record = asyncio.run(svc.register_asset(1, "abc_doc.pdf", str(target),
                                                document_name="Doc", source_url="http://x", org="gina"))
        assert record.asset_id == 9
        assert created["asset_name"] == "abc_doc.pdf"
        assert created["asset_size"] == "4"
        assert created["asset_config"] == {"document_name": "Doc", "source_url": "http://x", "org": "gina"}


class TestProcessService:

    def _service(self):
        chunk_model = MagicMock()
        inserted = {}

        async def _insert(chunks):
            inserted["chunks"] = chunks
            return len(chunks)

        chunk_model.insert_many_chunks.side_effect = _insert
        asset_model = MagicMock()

        async def _one(asset_project_id, asset_name):
            return SimpleNamespace(asset_id=3, asset_name="f.pdf",
                                   asset_config={"document_name": "Doc", "source_url": "u", "org": "gina"})

        asset_model.get_asset_record.side_effect = _one
        svc = ProcessService(chunk_model=chunk_model, asset_model=asset_model,
                             answering=MagicMock(), vectordb_client=MagicMock())
        return svc, inserted

    def test_process_single_file_counts_and_shapes_chunks(self):
        from application.ingestion import ProcessOptions
        svc, inserted = self._service()
        fake_chunks = [SimpleNamespace(page_content="hello world asthma",
                                       metadata={"page": 2})]
        with patch("application.ingestion.ProcessController") as factory:
            factory.return_value.get_file_content.return_value = [{"page_content": "x"}]
            factory.return_value.process_file_content.return_value = fake_chunks
            result = asyncio.run(svc.process_project(
                _project(), ProcessOptions(file_id="f.pdf", chunk_size=100, overlap_size=10)))
        assert result["signal"] == Response.FILE_PROCESSING_SUCCEED.value
        assert result == {**result, "inserted_chunks": 1, "processed_files": 1}
        record = inserted["chunks"][0]
        assert record.chunk_text == "hello world asthma"
        assert record.chunk_order == 1
        assert record.chunk_project_id == 1
        assert record.chunk_asset_id == 3
        meta = json.loads(record.chunk_metadata)
        assert meta["asset_id"] == 3
        assert meta["file_name"] == "f.pdf"
        assert meta["document_name"] == "Doc"

    def test_missing_file_id_raises(self):
        from application.ingestion import ProcessOptions
        svc, _ = self._service()
        svc.asset_model.get_asset_record.side_effect = self._none_record
        with pytest.raises(ValidationError) as exc_info:
            asyncio.run(svc.process_project(_project(), ProcessOptions(file_id="nope.pdf")))
        assert exc_info.value.signal == Response.FILE_ID_ERROR.value

    @staticmethod
    async def _none_record(asset_project_id, asset_name):
        return None


class TestAnswerResultPresenter:

    def _result(self, answer):
        return AnswerResult(
            answer=answer, full_prompt="fp", chat_history=[], sources=[{"a": 1}],
            risk_assessment={"risk_level": "allowed"}, confidence={"generation_allowed": True},
            quality={"citations": [], "unsupported_claims": [], "unsupported_claim_rate": 0.0,
                     "citation_faithfulness": 1.0},
            disclaimer="d", evidence_panel={}, expanded_query="q expanded",
        )

    def _meta(self):
        return {"retrieval_mode": "hybrid", "rerank_enabled": True,
                "query_expansion_enabled": True, "verify_claims_enabled": True}

    def test_full_answer_carries_generation_artefacts(self):
        payload = self._result("an answer").to_api_dict(
            signal="S", query="q", expanded_query="qe", query_expanded=True,
            pipeline_metadata=self._meta())
        assert payload["full_prompt"] == "fp"
        assert payload["retrieval_mode"] == "hybrid"
        assert payload["rerank"] is True
        assert payload["answer"] == "an answer"

    def test_empty_answer_omits_generation_artefacts(self):
        payload = self._result("").to_api_dict(
            signal="S", query="q", expanded_query="qe", query_expanded=True,
            pipeline_metadata=self._meta())
        assert payload["answer"] == ""
        assert "full_prompt" not in payload
        assert "chat_history" not in payload
        assert "retrieval_mode" not in payload
