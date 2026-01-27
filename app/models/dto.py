# Data Transfer Objects (DTOs)
# Objets de transfert de données (DTOs)

from pydantic import BaseModel


class FaceBox(BaseModel):
    # Bounding box of a detected face
    # Boîte englobante d'un visage détecté
    x: int
    y: int
    width: int
    height: int


class RecognizedFace(BaseModel):
    # A recognized face with label + confidence + bounding box
    # Un visage reconnu avec étiquette + confiance + boîte englobante
    label: str
    confidence: float
    box: FaceBox


class RecognitionResponse(BaseModel):
    # Response containing all recognized faces + unknown count
    # Réponse contenant tous les visages reconnus + nombre d'inconnus
    recognized: list[RecognizedFace]
    unknown_faces: int