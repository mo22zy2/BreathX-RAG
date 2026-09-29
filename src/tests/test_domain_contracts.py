"""
Unit tests for domain contracts (AnswerRequest validation).

Run from src/:
    pytest tests/test_domain_contracts.py -v
"""
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from domain.contracts import AnswerRequest
from domain.exceptions import ValidationError


class TestAnswerRequest:

    def test_defaults(self):
        r = AnswerRequest(project_id=1, query="What is asthma?")
        assert r.limit == 5
        assert r.retrieval_mode == "hybrid"
        assert r.verify_claims is True

    def test_limit_clamped_to_range(self):
        assert AnswerRequest(project_id=1, query="q", limit=100000).limit == 50
        assert AnswerRequest(project_id=1, query="q", limit=-5).limit == 1
        assert AnswerRequest(project_id=1, query="q", limit=0).limit == 1

    def test_limit_passthrough_in_range(self):
        assert AnswerRequest(project_id=1, query="q", limit=7).limit == 7

    def test_limit_garbage_rejected(self):
        with pytest.raises(ValidationError):
            AnswerRequest(project_id=1, query="q", limit="many")

    def test_bad_project_id_rejected(self):
        for bad in (-1, 0, None):
            with pytest.raises(ValidationError):
                AnswerRequest(project_id=bad, query="q")

    def test_none_query_becomes_empty(self):
        assert AnswerRequest(project_id=1, query=None).query == ""
