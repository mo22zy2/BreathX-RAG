from .BaseDataModel import BaseDataModel, ensure_valid_project_id
from .db_schemas import Project
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker


class ProjectModel(BaseDataModel):
    def __init__(self, db_client: sessionmaker):
        super().__init__(db_client=db_client)


    async def create_project(self, project: Project):
        async with self.db_client() as session:
            async with session.begin():
                session.add(project)
            await session.refresh(project)
            
        return project

    async def get_project_or_create_one(self, project_id: int):
        # Guard first: ids below 1 previously created junk rows (-1, 0, …)
        # and 500s. Auto-create on valid ids is preserved (see tech debt).
        project_id = ensure_valid_project_id(project_id)
        async with self.db_client() as session:
            async with session.begin(): 
                query = select(Project).where(Project.project_id==project_id)
                result = await session.execute(query)
                project = result.scalar_one_or_none()
                if project is None:
                    project_record=Project(
                        project_id=project_id
                    )
                    project = await self.create_project(project=project_record)
                    return project
                else:
                    return project

    async def get_all_projects(self, page: int = 1, page_size: int = 10):
        async with self.db_client() as session:
            async with session.begin():
                total_documents=await session.execute(
                    select(
                        func.count(Project.project_id)
                    )
                )
                
                total_documents=total_documents.scalar_one()
                
                total_pages=total_documents // page_size
                
                if total_documents % page_size >0:
                    total_pages+=1
                    
                query = select(Project).offset((page-1)*page_size).limit(page_size)
                result = await session.execute(query)
                projects = result.scalars().all()
                
                return projects ,total_pages