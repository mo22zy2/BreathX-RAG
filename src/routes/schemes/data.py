
from pydantic import BaseModel


class ProcessRequest(BaseModel):
    file_id:str=None
    chunk_size:int | None=100
    overlap_size:int | None=50
    do_reset:int | None=0
    chunking_method: str | None = None
    # Provenance fallback, applied when the asset has no asset_config
    document_name: str | None = None
    source_url: str | None = None
    org: str | None = None
    
