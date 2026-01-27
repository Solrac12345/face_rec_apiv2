# FaceService: multi-face detection + recognition
# FaceService : détection et reconnaissance multi-visages

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from deepface import DeepFace

from app.models.dto import FaceBox, RecognizedFace, RecognitionResponse


@dataclass
class FaceDetectionResult:
    # Simple container for detected face boxes
    # Conteneur simple pour les boîtes de visages détectés
    boxes: List[FaceBox]


class FaceService:
    def __init__(
        self,
        known_faces_dir: Path = Path("known_faces"),
        similarity_threshold: float = 0.35,
    ) -> None:

        # Directory containing known face images
        # Dossier contenant les images des visages connus
        self._known_faces_dir = known_faces_dir

        # Cosine distance threshold for recognition
        # Seuil de distance cosinus pour la reconnaissance
        self._similarity_threshold = similarity_threshold

        # ArcFace model (512‑dim embeddings)
        # Modèle ArcFace (vecteurs de 512 dimensions)
        self._model_name = "ArcFace"

        # OpenCV detector backend (stable on Windows)
        # Backend de détection OpenCV (stable sous Windows)
        self._detector_backend = "opencv"

        # List of (label, embedding)
        # Liste de (étiquette, vecteur d'embedding)
        self._embeddings: List[Tuple[str, np.ndarray]] = []

        self._load_known_faces()

    # ---------------------------------------------------------
    # Load known faces and compute embeddings
    # Charger les visages connus et calculer leurs embeddings
    # ---------------------------------------------------------

    def _load_known_faces(self) -> None:
        print(f"[INFO] Loading known faces from: {self._known_faces_dir}")

        if not self._known_faces_dir.exists():
            print("[WARNING] Known faces directory does not exist.")
            return

        image_paths = list(self._known_faces_dir.glob("*.jpg")) + \
                      list(self._known_faces_dir.glob("*.png"))

        print(f"[INFO] Found {len(image_paths)} face images.")

        for image_path in image_paths:
            # Extract label from filename (e.g., carlos_1.jpg → carlos)
            # Extraire l'étiquette du nom de fichier
            label = image_path.stem.split("_")[0]

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"[ERROR] Could not read image: {image_path}")
                continue

            embedding = self._compute_embedding(img)
            if embedding is None:
                print(f"[WARNING] No valid embedding for {image_path}")
                continue

            self._embeddings.append((label, embedding))
            print(f"[OK] Added embedding for '{label}'")

        print(f"[INFO] Total embeddings loaded: {len(self._embeddings)}")

    # ---------------------------------------------------------
    # Decode uploaded image bytes
    # Décoder les octets d'image envoyés
    # ---------------------------------------------------------

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        return image

    # ---------------------------------------------------------
    # Compute embedding using DeepFace (ArcFace)
    # Calculer l'embedding avec DeepFace (ArcFace)
    # ---------------------------------------------------------

    def _compute_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        try:
            reps = DeepFace.represent(
                img_path=image,
                model_name=self._model_name,
                detector_backend=self._detector_backend,
                enforce_detection=False,  # Avoid crashes if no face detected
                # Évite les crashs si aucun visage n'est détecté
            )
        except Exception as e:
            print(f"[ERROR] Failed to compute embedding: {e}")
            return None

        # Your DeepFace version returns a raw list of 512 floats
        # Votre version de DeepFace retourne une liste brute de 512 floats
        if isinstance(reps, list) and len(reps) == 512:
            return np.array(reps, dtype=np.float32)

        print(f"[WARNING] Unexpected embedding format: {type(reps)}")
        return None

    # ---------------------------------------------------------
    # Detect faces using OpenCV Haar cascade
    # Détecter les visages avec le cascade Haar d'OpenCV
    # ---------------------------------------------------------

    def _detect_faces_opencv(self, image: np.ndarray) -> List[FaceBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        detections = face_cascade.detectMultiScale(gray, 1.2, 5)

        boxes: List[FaceBox] = []
        for (x, y, w, h) in detections:
            boxes.append(FaceBox(x=x, y=y, width=w, height=h))

        return boxes

    # ---------------------------------------------------------
    # Compare embedding with known identities (cosine distance)
    # Comparer l'embedding avec les identités connues (distance cosinus)
    # ---------------------------------------------------------

    def _find_best_match(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._embeddings:
            return None, float("inf")

        best_label = None
        best_distance = float("inf")

        for label, stored_embedding in self._embeddings:
            # Cosine distance
            # Distance cosinus
            dot = np.dot(query_embedding, stored_embedding)
            norm_a = np.linalg.norm(query_embedding)
            norm_b = np.linalg.norm(stored_embedding)
            dist = 1 - (dot / (norm_a * norm_b + 1e-10))

            if dist < best_distance:
                best_distance = dist
                best_label = label

        return best_label, best_distance

    # ---------------------------------------------------------
    # Public API: detect faces
    # API publique : détecter les visages
    # ---------------------------------------------------------

    def detect_faces(self, image_bytes: bytes) -> FaceDetectionResult:
        image = self._decode_image(image_bytes)
        boxes = self._detect_faces_opencv(image)
        return FaceDetectionResult(boxes=boxes)

    # ---------------------------------------------------------
    # Public API: recognize multiple faces
    # API publique : reconnaître plusieurs visages
    # ---------------------------------------------------------

    def recognize_faces(self, image_bytes: bytes) -> RecognitionResponse:
        image = self._decode_image(image_bytes)
        boxes = self._detect_faces_opencv(image)

        recognized: List[RecognizedFace] = []
        unknown_count = 0

        if not boxes:
            return RecognitionResponse(recognized=[], unknown_faces=0)

        for box in boxes:
            x, y, w, h = box.x, box.y, box.width, box.height

            # Crop face region
            # Découper la région du visage
            face_roi = image[y:y + h, x:x + w]

            if face_roi.size == 0:
                unknown_count += 1
                continue

            embedding = self._compute_embedding(face_roi)
            if embedding is None:
                unknown_count += 1
                continue

            label, distance = self._find_best_match(embedding)

            if label is not None and distance <= self._similarity_threshold:
                confidence = float(max(0.0, 1.0 - distance)) * 100.0

                recognized.append(
                    RecognizedFace(
                        label=label,
                        confidence=confidence,
                        box=box,
                    )
                )
            else:
                unknown_count += 1

        return RecognitionResponse(
            recognized=recognized,
            unknown_faces=unknown_count,
        )