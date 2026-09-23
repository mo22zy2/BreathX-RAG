#!/bin/bash
set -e
echo "Running DB migrations"
cd /app/models/db_schemas/breathx_rag/
alembic upgrade head
cd /app
exec "$@"