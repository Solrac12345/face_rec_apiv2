# EN: Application configuration using pydantic-settings with env var support
# FR-CA: Configuration de l'application utilisant pydantic-settings avec support des variables d'environnement

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # EN: Application metadata
    # FR-CA: Métadonnées de l'application
    app_name: str = "Face Recognition API v2"
    app_version: str = "0.1.0"
    debug: bool = False

    # EN: Server configuration
    # FR-CA: Configuration du serveur
    host: str = "0.0.0.0"
    port: int = 8000

    # EN: Face recognition parameters
    # FR-CA: Paramètres de reconnaissance faciale
    face_detection_threshold: float = 0.35  # Cosine distance threshold
    embedding_model: str = "ArcFace"
    min_face_size: int = 90  # Minimum face bounding box size in pixels

    # EN: File paths (configurable via env vars)
    # FR-CA: Chemins de fichiers (configurables via variables d'environnement)
    known_faces_dir: Path = Path("known_faces")
    haarcascade_path: Path = Path("data/haarcascade_frontalface_default.xml")

    # EN: Security & API settings (to be used in future auth step)
    # FR-CA: Paramètres de sécurité et API (à utiliser dans l'étape d'authentification future)
    api_key: str | None = None  # Optional simple API key for now
    jwt_secret_key: str | None = None
    jwt_algorithm: str = "HS256"
    token_expire_minutes: int = 30

    # EN: Performance & limits
    # FR-CA: Performance et limites
    max_upload_size_mb: int = 10
    allowed_image_extensions: list[str] = [".jpg", ".jpeg", ".png", ".webp"]

    # EN: Pydantic-settings config: load from .env file and environment
    # FR-CA: Configuration pydantic-settings: chargement depuis fichier .env et environnement
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """
    EN: Cached settings instance for performance (loaded once per worker)
    FR-CA: Instance de configuration mise en cache pour la performance (chargée une fois par worker)
    """
    return Settings()
