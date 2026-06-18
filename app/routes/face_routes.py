# EN: API routes for face detection & recognition with auth guard
# FR-CA: Routes API pour détection/reconnaissance faciale avec garde d'authentification

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

from app.config import Settings, get_settings
from app.services.face_service import FaceService
from app.models.dto import RecognitionResponse, FaceBox, RecognizedFace
from app.auth import verify_auth

router = APIRouter()


def get_face_service(settings: Settings = Depends(get_settings)) -> FaceService:
    return FaceService(settings)


def _validate_upload(file: UploadFile, max_size_mb: int) -> np.ndarray:
    allowed = [".jpg", ".jpeg", ".png", ".webp"]
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail="Unsupported image format")
    
    contents = file.file.read()
    if len(contents) > max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds size limit")
    
    image = Image.open(BytesIO(contents)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@router.post("/detect", response_model=list[FaceBox])
async def detect_faces(
    file: UploadFile = File(...),
    service: FaceService = Depends(get_face_service),
    settings: Settings = Depends(get_settings),
    auth: dict = Depends(verify_auth)
):
    image = _validate_upload(file, settings.max_upload_size_mb)
    boxes = await service.detect_faces_async(image)
    return [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in boxes]


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_faces(
    file: UploadFile = File(...),
    service: FaceService = Depends(get_face_service),
    settings: Settings = Depends(get_settings),
    auth: dict = Depends(verify_auth)
):
    image = _validate_upload(file, settings.max_upload_size_mb)
    result = await service.recognize_faces_async(image)
    
    recognized_faces = [
        RecognizedFace(label=item["label"], confidence=item["confidence"], box=FaceBox(**item["box"]))
        for item in result["recognized"]
    ]
    
    return RecognitionResponse(recognized=recognized_faces, unknown_faces=result["unknown_faces"])