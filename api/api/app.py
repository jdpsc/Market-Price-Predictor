from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.views import api_router
from api.config import get_settings


def create_app() -> FastAPI:

    version = get_settings().VERSION

    app = FastAPI(
        title=f"{get_settings().PROJECT_NAME}",
        docs_url=f"/api/{version}/docs",
        openapi_url=f"/api/{version}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=f"/api/{version}")

    return app
