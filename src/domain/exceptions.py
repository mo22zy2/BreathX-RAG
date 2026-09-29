"""Domain exceptions: meaningful errors raised by business logic.

Each error carries an HTTP status and a response payload. The API layer
translates these into responses; infrastructure details (tracebacks,
connection strings) never leave the process.
"""


class BreathXError(Exception):
    status_code: int = 500
    signal: str = "Unexpected error"

    def __init__(self, signal: str = None, extra: dict = None):
        if signal is not None:
            self.signal = signal
        # Extra response fields (e.g. partial counts); merged by the handler.
        self.extra = extra or {}
        super().__init__(self.signal)


class ValidationError(BreathXError):
    status_code = 400


class ProjectNotFoundError(BreathXError):
    status_code = 400

    def __init__(self, project_id: int = None):
        detail = "project_was_not_found" if project_id is None else f"project_was_not_found: {project_id}"
        super().__init__(detail)


class UpstreamError(BreathXError):
    """Vector DB, LLM, rerank, or embedding backend failure."""
    status_code = 502


class InsufficientEvidenceError(BreathXError):
    status_code = 200

    def __init__(self, signal: str = "No relevant documents found to answer the question"):
        super().__init__(signal)
