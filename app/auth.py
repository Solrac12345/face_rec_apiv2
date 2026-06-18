# EN: Unified authentication dependency (API Key + JWT + Dev Fallback)
# FR-CA: Dépendance d'authentification unifiée (Clé API + JWT + Mode Dev)

import hmac
import logging

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# EN: Security schemes (auto_error=False prevents 403 before our logic runs)
# FR-CA: Schémas de sécurité (auto_error=False évite 403 avant notre logique)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_auth(
    api_key: str | None = Security(api_key_header),
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    EN: Validate API Key or JWT. Returns auth context dict.
    FR-CA: Valider clé API ou JWT. Retourne un dictionnaire de contexte d'authentification.
    """
    # 1️⃣ API Key validation (constant-time comparison prevents timing attacks)
    if api_key and settings.api_key:
        if not hmac.compare_digest(api_key.encode(), settings.api_key.encode()):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")
        return {"auth_type": "api_key", "subject": "api_client"}

    # 2️⃣ JWT Bearer validation
    if credentials and settings.jwt_secret_key:
        try:
            payload = jwt.decode(
                credentials.credentials,
                settings.jwt_secret_key,
                algorithms=[settings.jwt_algorithm],
                options={"verify_exp": True},
            )
            return {"auth_type": "jwt", "subject": payload.get("sub", "unknown")}
        except InvalidTokenError as e:
            logger.warning(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired token"
            ) from e

    # 3️⃣ Dev mode: allow anonymous if neither secret is configured
    if not settings.api_key and not settings.jwt_secret_key:
        logger.debug("Authentication bypassed (dev mode: no secrets configured)")
        return {"auth_type": "none", "subject": "anonymous"}

    # 4️⃣ Fallback: auth required but no valid credentials provided
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
