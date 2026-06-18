# EN: Pytest fixtures for async FastAPI testing
# FR-CA: Fixtures pytest pour tests async FastAPI

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.config import Settings, get_settings


@pytest.fixture(scope="function")
def override_settings():
    """EN: Override settings for test environment
    FR-CA: Surcharger les paramètres pour l'environnement de test"""
    original = get_settings.cache_info()
    get_settings.cache_clear()
    
    # EN: Create test-specific settings
    # FR-CA: Créer des paramètres spécifiques aux tests
    class TestSettings(Settings):
        debug: bool = True
        known_faces_dir: str = "tests/test_known_faces"
        haarcascade_path: str = "data/haarcascade_frontalface_default.xml"
        api_key: str | None = None  # Disable auth for tests
        jwt_secret_key: str | None = None
    
    # EN: Mock the dependency
    # FR-CA: Simuler la dépendance
    def _get_test_settings():
        return TestSettings()
    
    yield _get_test_settings
    
    # EN: Restore original cache
    # FR-CA: Restaurer le cache original
    get_settings.cache_clear()


@pytest.fixture(scope="function")
def app(override_settings):
    """EN: Create test app instance with overridden settings
    FR-CA: Créer une instance d'application de test avec paramètres surchargés"""
    from app.config import get_settings
    # EN: Inject test settings
    # FR-CA: Injecter les paramètres de test
    import app.config
    app.config.get_settings = override_settings
    return create_app()


@pytest.fixture(scope="function")
async def client(app):
    """EN: Async HTTP client for testing endpoints
    FR-CA: Client HTTP async pour tester les endpoints"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac