
from fastapi import APIRouter, Depends, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse

from application.ingestion import ProcessOptions, ProcessService, UploadService
from auth.jwt import get_current_user
from helpers.config import Settings, get_settings
from models import Response
from models.ProjectModel import ProjectModel

from .schemes.data import ProcessRequest

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1,data"]
)

@data_router.post("/upload/{project_id}")

async def upload_data(
    request:Request,
    project_id:int,
    file:UploadFile,
    document_name: str | None = Form(None),
    source_url: str | None = Form(None),
    org: str | None = Form(None),
    app_settings:Settings =Depends(get_settings), current_user: str = Depends(get_current_user)):
    
    project = await ProjectModel(db_client=request.app.db_client).get_project_or_create_one(project_id=project_id)
    
    service = UploadService.from_app(request.app)
    isValid,response_signal = await service.validate_upload(file=file)
    
    if not isValid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":response_signal}
        )
        
    file_path,file_id = service.build_filepath(
        original_file_name=file.filename,
        project_id=project_id,
    )
    max_size_bytes = app_settings.FILE_MAX_SIZE * 1048576
    await service.save_upload(
        file=file,
        file_path=file_path,
        max_size_bytes=max_size_bytes,
        chunk_size=app_settings.FILE_DEFAULT_CHUNK_SIZE,
    )
    asset_record = await service.register_asset(
        project_id=project.project_id,
        file_id=file_id,
        file_path=file_path,
        document_name=document_name,
        source_url=source_url,
        org=org,
    )
    
    return JSONResponse(
        content={
            "signal":Response.FILE_UPLOAD_SUCCED.value,
            "file_id":str(asset_record.asset_id),

        }
    )
    
    
    
@data_router.post("/process/{project_id}")

async def process_endpoint(request:Request,project_id:int,process_request:ProcessRequest, current_user: str = Depends(get_current_user)):
    
    project = await ProjectModel(db_client=request.app.db_client).get_project_or_create_one(project_id=project_id)
    
    options = ProcessOptions(
        file_id=process_request.file_id,
        chunk_size=process_request.chunk_size,
        overlap_size=process_request.overlap_size,
        do_reset=process_request.do_reset,
        chunking_method=process_request.chunking_method,
        document_name=process_request.document_name,
        source_url=process_request.source_url,
        org=process_request.org,
    )
    result = await ProcessService.from_app(request.app).process_project(project, options)
    
    return JSONResponse(content=result)
