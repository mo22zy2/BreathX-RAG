"""Startup data checks that must run before serving traffic.

Kept separate from the Alembic migration path on purpose: these are
idempotent guards for columns the application depends on, safe to run on
every boot (including concurrent uvicorn workers).
"""
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger("uvicorn.error")


async def ensure_chunks_indexed_column(engine: AsyncEngine) -> None:
    async with engine.connect() as conn:
        await conn.execute(text(
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_indexed BOOLEAN NOT NULL DEFAULT FALSE"
        ))
        await conn.commit()
