"""
Unit tests for repository input guards (no live database — fakes only).

Run from src/:
    pytest tests/test_repositories.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domain.exceptions import ValidationError
from models.BaseDataModel import ensure_valid_project_id
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from models.AssetModel import AssetModel


class TestEnsureValidProjectId:

    @pytest.mark.parametrize("good", [1, 5, "3"])
    def test_accepts_positive(self, good):
        assert ensure_valid_project_id(good) == int(good)

    @pytest.mark.parametrize("bad", [-1, 0, None, "abc"])
    def test_rejects_non_positive(self, bad):
        with pytest.raises(ValidationError):
            ensure_valid_project_id(bad)


class TestRepositoryGuardsFailFast:
    """Guards raise before any database session is opened."""

    def _failing_db(self):
        db = MagicMock()
        return db

    def test_project_model_rejects_bad_id_without_db(self):
        model = ProjectModel(db_client=self._failing_db())
        with pytest.raises(ValidationError):
            import asyncio
            asyncio.run(model.get_project_or_create_one(-1))
        model.db_client.assert_not_called()

    def test_chunk_model_rejects_bad_id_without_db(self):
        model = ChunkModel(db_client=self._failing_db())
        with pytest.raises(ValidationError):
            import asyncio
            asyncio.run(model.get_unindexed_chunk_count(0))
        model.db_client.assert_not_called()

    def test_asset_model_rejects_bad_id_without_db(self):
        model = AssetModel(db_client=self._failing_db())
        with pytest.raises(ValidationError):
            import asyncio
            asyncio.run(model.get_asset_record(asset_project_id=-5, asset_name="x.pdf"))
        model.db_client.assert_not_called()
