from fastapi import FastAPI

from app.api.routes import states, architecture
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    version=settings.api_version,
    root_prefix=settings.api_prefix
)

app.include_router(states.router, prefix='/states', tags=['States'])
app.include_router(architecture.router, prefix='/architecture', tags=['Architecture'])