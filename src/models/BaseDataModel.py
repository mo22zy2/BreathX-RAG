from sqlalchemy.orm import sessionmaker

from domain.exceptions import ValidationError
from helpers.config import get_settings


def ensure_valid_project_id(project_id: int) -> int:
    """Guard every project-scoped query: ids below 1 used to create junk
    rows (e.g. -1, 0) and 500s. Pure function, unit-tested with fakes."""
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        raise ValidationError(f"invalid project_id: {project_id!r}")
    if pid < 1:
        raise ValidationError(f"invalid project_id: {project_id!r}")
    return pid


class BaseDataModel:
    def __init__(self, db_client: sessionmaker):
        self.db_client = db_client
        self.app_settings = get_settings()
