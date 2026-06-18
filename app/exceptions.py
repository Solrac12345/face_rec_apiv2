# EN: Global exception handlers with standardized JSON error responses
# FR-CA: Gestionnaires globaux d'exceptions avec réponses d'erreur JSON standardisées

import logging
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """EN: Standardized error payload for all API failures
    FR-CA: Charge utile d'erreur standardisée pour toutes les défaillances API"""
    status: str = "error"
    code: str
    message: str
    details: dict | list | None = None


class AppException(Exception):
    """EN: Custom application exception for business logic errors
    FR-CA: Exception applicative personnalisée pour les erreurs métier"""
    def __init__(self, status_code: int, code: str, message: str, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(code=exc.code, message=exc.message, details=exc.details).model_dump()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.debug(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            code="VALIDATION_ERROR",
            message="Invalid request payload",
            details=exc.errors()
        ).model_dump()
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """EN: Catch-all for FastAPI HTTPException (e.g., 401, 403, 404)
    FR-CA: Interception globale pour HTTPException FastAPI"""
    status_code = getattr(exc, "status_code", 500)
    detail = getattr(exc, "detail", "Internal server error")
    
    if status_code >= 500:
        logger.error(f"Unhandled {status_code} on {request.url}: {detail}")
        
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            code=f"HTTP_{status_code}",
            message=str(detail),
            details=None
        ).model_dump()
    )