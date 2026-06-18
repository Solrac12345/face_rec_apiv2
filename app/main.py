# EN: FastAPI app with lifespan context manager & security headers
# FR-CA: Application FastAPI avec contexte lifespan et en-têtes de sécurité

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from app.config import Settings, get_settings

# EN: Import all exception handlers
# FR-CA: Importer tous les gestionnaires d'exceptions
from app.exceptions import (
    AppException,
    app_exception_handler,
    validation_exception_handler,
    http_exception_handler
)

# EN: Module-level logger (standard Python pattern)
# FR-CA: Logger au niveau du module (pattern Python standard)
logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """EN: Modern startup/shutdown lifecycle (replaces deprecated @app.on_event)
    FR-CA: Cycle de vie startup/shutdown moderne (remplace @app.on_event déprécié)"""
    logger.info("Face Recognition API starting up...")
    yield
    logger.info("Face Recognition API shutting down...")


def create_app() -> FastAPI:
    """EN: Application factory pattern for testability and config injection
    FR-CA: Patron de fabrique d'application pour la testabilité et l'injection de config"""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None
    )

    # 🔒 Register structured error handlers (order matters: specific → general)
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, http_exception_handler)  # Catch-all fallback

    # EN: CORS & Security middleware
    # FR-CA: Middleware CORS et sécurité
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],  # Tighten in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)

    # EN: Register routes
    # FR-CA: Enregistrement des routes
    from app.routes import face_routes
    app.include_router(face_routes.router, prefix="/face", tags=["face"])

    # EN: Health check endpoint
    # FR-CA: Endpoint de vérification de santé
    @app.get("/health")
    async def health_check(settings: Settings = Depends(get_settings)):
        return {"status": "ok", "version": settings.app_version, "debug": settings.debug}

    return app


# EN: Create app instance for uvicorn
# FR-CA: Création de l'instance d'application pour uvicorn
app = create_app()