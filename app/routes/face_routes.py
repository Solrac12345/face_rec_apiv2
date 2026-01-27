# API routes for face detection and recognition
# Routes API pour la détection et la reconnaissance faciale

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.services.face_service import FaceService
from app.models.dto import RecognitionResponse, FaceBox

router = APIRouter(prefix="/face", tags=["face"])

# Create service instance once (efficient)
# Créer une instance du service une seule fois (efficace)
face_service = FaceService()


@router.post("/detect", response_model=list[FaceBox])
async def detect_faces(file: UploadFile = File(...)):
    # Detect faces in uploaded image
    # Détecter les visages dans l'image envoyée
    content = await file.read()
    try:
        result = face_service.detect_faces(content)
        return result.boxes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_faces(file: UploadFile = File(...)):
    # Recognize multiple faces in uploaded image
    # Reconnaître plusieurs visages dans l'image envoyée
    content = await file.read()
    try:
        result = face_service.recognize_faces(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))