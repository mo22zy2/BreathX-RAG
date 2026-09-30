"""Answering application service (package).

Import path preserved: `from controllers.NLPController import NLPController`
keeps working for routes, the package facade, and tests. `get_settings` is
re-exported so the test patch-point `controllers.NLPController.get_settings`
keeps resolving (the controller reads it lazily from this namespace).
"""
from helpers.config import get_settings  # noqa: F401  (re-export: test patch-point)

from .controller import NLPController  # noqa: F401

__all__ = ["NLPController", "get_settings"]
