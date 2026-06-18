# EN: Async integration tests for face API routes
# FR-CA: Tests d'intégration async pour les routes API faciales

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
from io import BytesIO


@pytest.mark.asyncio
async def test_health_check(client):
    """EN: Verify health endpoint returns expected structure
    FR-CA: Vérifier que l'endpoint health retourne la structure attendue"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert isinstance(data["debug"], bool)


@pytest.mark.asyncio
async def test_detect_invalid_content_type(client):
    """EN: Reject non-image file uploads with 400
    FR-CA: Rejeter les téléversements non-image avec 400"""
    response = await client.post(
        "/face/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    
    # EN: Check structured error format (code OR message field)
    # FR-CA: Vérifier le format d'erreur structuré (champ code OU message)
    error_msg = data.get("message") or data.get("detail", "")
    assert "Unsupported image format" in str(error_msg) or "HTTP_400" in data.get("code", "")

@pytest.mark.asyncio
async def test_recognize_mock_success(client):
    """EN: Test recognition flow with mocked ML layer
    FR-CA: Tester le flux de reconnaissance avec couche ML simulée"""
    # EN: Minimal valid PNG header (1x1 pixel)
    # FR-CA: En-tête PNG valide minimal (pixel 1x1)
    mock_png = BytesIO(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    mock_png.name = "test.png"
    
    # EN: Mock the service layer to avoid actual ML inference
    # FR-CA: Simuler la couche service pour éviter l'inférence ML réelle
    with patch("app.routes.face_routes._validate_upload") as mock_val, \
         patch("app.services.face_service.FaceService.recognize_faces_async") as mock_rec:
        
        mock_val.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_rec.return_value = {
            "recognized": [{
                "label": "test_person",
                "confidence": 98.5,
                "box": {"x": 10, "y": 10, "width": 50, "height": 50}
            }],
            "unknown_faces": 0
        }
        
        response = await client.post(
            "/face/recognize",
            files={"file": ("test.png", mock_png, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["recognized"]) == 1
        assert data["recognized"][0]["label"] == "test_person"
        assert data["unknown_faces"] == 0


@pytest.mark.asyncio
async def test_auth_bypass_dev_mode(client):
    """EN: Verify auth is bypassed when no secrets configured (dev mode)
    FR-CA: Vérifier que l'auth est contournée quand aucun secret n'est configuré (mode dev)"""
    # EN: Mock the upload validation to avoid actual image parsing
    # FR-CA: Simuler la validation de téléversement pour éviter le parsing d'image réel
    with patch("app.routes.face_routes._validate_upload") as mock_val, \
         patch("app.services.face_service.FaceService.detect_faces_async") as mock_detect:
        
        mock_val.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_detect.return_value = [(10, 10, 50, 50)]  # Mock one face box
        
        response = await client.post(
            "/face/detect",
            files={"file": ("test.jpg", b"mocked", "image/jpeg")}
        )
        
        # EN: Should NOT be 401 in dev mode (auth bypassed)
        # FR-CA: Ne devrait PAS être 401 en mode dev (auth contournée)
        assert response.status_code != 401
        # EN: Should be 200 (success) or 400 (validation) but not auth error
        # FR-CA: Devrait être 200 (succès) ou 400 (validation) mais pas erreur d'auth
        assert response.status_code in [200, 400]