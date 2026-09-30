from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .BaseDataModel import BaseDataModel, ensure_valid_project_id
from .db_schemas import Asset


class AssetModel(BaseDataModel):
    def __init__(self, db_client: sessionmaker):
        super().__init__(db_client=db_client)

                
    async def create_asset(self, asset: Asset):
        async with self.db_client() as session:
            async with session.begin():
                session.add(asset)
            await session.refresh(asset)
        return asset
    
    
    
    
    
    async def get_all_project_assets(self, asset_project_id: str,asset_type: str):
        asset_project_id = ensure_valid_project_id(asset_project_id)
        async with self.db_client() as session:
            async with session.begin():
                get_all_project_asset_query=select(Asset).where(
                    Asset.asset_project_id==asset_project_id,
                    Asset.asset_type==asset_type
                )
                result=await session.execute(get_all_project_asset_query)
                record=result.scalars().all()
        return record
        
    async def get_asset_record(self,asset_project_id:int , asset_name:str):
        asset_project_id = ensure_valid_project_id(asset_project_id)
        async with self.db_client() as session:
            async with session.begin():
                get_asset_record_query=select(Asset).where(
                    Asset.asset_project_id==asset_project_id,
                    Asset.asset_name==asset_name
                )
                result=await session.execute(get_asset_record_query)
                record=result.scalar_one_or_none()
        return record
