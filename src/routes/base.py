from fastapi import APIRouter,FastAPI,Depends
from helpers.config import get_settings,Settings
from auth.jwt import get_current_user
from datetime import datetime

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)

@base_router.get("/")
async def welcome(app_settings:Settings =Depends(get_settings), current_user: str = Depends(get_current_user)):
    
    app_name=app_settings.APP_NAME
    
    app_version=app_settings.APP_VERSION
    return {
        "app_name":app_name,
        "app_version":app_version,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }