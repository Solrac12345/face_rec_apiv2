# EN: Async-compatible face recognition service with thread-pooled ML inference
# FR-CA: Service de reconnaissance faciale compatible async avec inférence ML en pool de threads

import asyncio
import logging

import cv2
import numpy as np
from deepface import DeepFace

from app.config import Settings

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.threshold = settings.face_detection_threshold
        self.min_face_size = settings.min_face_size

        cascade_path = settings.haarcascade_path
        if not cascade_path.exists():
            raise FileNotFoundError(f"Haar cascade not found at: {cascade_path}")
        self.detector = cv2.CascadeClassifier(str(cascade_path))

        self.known_faces_dir = settings.known_faces_dir
        self.known_embeddings: dict[str, np.ndarray] = {}
        self._load_known_faces_sync()  # Sync OK: runs during startup, not per-request

    def _load_known_faces_sync(self) -> None:
        """Load embeddings at startup. Kept sync for simplicity during app initialization."""
        # ✅ All method body indented 8 spaces relative to 'def'
        if not self.known_faces_dir.exists():
            logger.warning(f"Creating known_faces_dir: {self.known_faces_dir}")
            self.known_faces_dir.mkdir(parents=True, exist_ok=True)
            return

        for img_path in self.known_faces_dir.glob("*"):
            if img_path.suffix.lower() in self.settings.allowed_image_extensions:
                try:
                    label = img_path.stem
                    result = DeepFace.represent(
                        str(img_path),
                        model_name=self.settings.embedding_model,
                        enforce_detection=False
                    )

                    # EN: Robust embedding extraction for DeepFace API variations
                    # FR-CA: Extraction robuste d'embedding pour les variations d'API DeepFace
                    if not result or len(result) == 0:
                        logger.warning(f"No embedding returned for {img_path}")
                        continue

                    embedding_data = result[0]

                    # Handle dict format: {"embedding": [...], "face_confidence": 0.99}
                    if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                        embedding = embedding_data["embedding"]
                    # Handle direct list/array format
                    elif isinstance(embedding_data, (list, np.ndarray)):
                        embedding = embedding_data
                    # Handle scalar/float (edge case - skip)
                    elif isinstance(embedding_data, (int, float)):
                        logger.warning(f"Unexpected scalar embedding for {img_path}")
                        continue
                    else:
                        logger.warning(f"Unknown embedding format for {img_path}: {type(embedding_data)}")
                        continue

                    # Convert to numpy array and validate dimensionality
                    embedding_array = np.array(embedding, dtype=np.float32)
                    if embedding_array.ndim != 1:
                        logger.warning(f"Unexpected embedding shape for {img_path}: {embedding_array.shape}")
                        continue

                    self.known_embeddings[label] = embedding_array
                    logger.info(f"Loaded known face: {label} ({len(embedding_array)}-dim embedding)")

                except Exception as e:
                    logger.error(f"Failed to load {img_path}: {type(e).__name__}: {e}")

    async def detect_faces_async(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Non-blocking face detection using thread pool."""
        return await asyncio.to_thread(self._detect_faces_sync, image)

    def _detect_faces_sync(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    async def recognize_faces_async(self, image: np.ndarray) -> dict:
        """Non-blocking face recognition with thread-pooled DeepFace calls."""
        boxes = await self.detect_faces_async(image)
        recognized = []
        unknown_count = 0

        for x, y, w, h in boxes:
            face_crop = image[y:y+h, x:x+w]
            try:
                result = await asyncio.to_thread(
                    DeepFace.represent,
                    face_crop,
                    model_name=self.settings.embedding_model,
                    enforce_detection=False
                )
                if not result or len(result) == 0:
                    unknown_count += 1
                    continue

                # EN: Robust query embedding extraction (same logic as _load_known_faces_sync)
                # FR-CA: Extraction robuste d'embedding de requête (même logique que _load_known_faces_sync)
                embedding_data = result[0]
                if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                    query_embedding = np.array(embedding_data["embedding"], dtype=np.float32)
                elif isinstance(embedding_data, (list, np.ndarray)):
                    query_embedding = np.array(embedding_data, dtype=np.float32)
                else:
                    logger.warning(f"Unexpected query embedding format: {type(embedding_data)}")
                    unknown_count += 1
                    continue

                distances = await asyncio.to_thread(
                    self._compute_distances, query_embedding
                )

                if distances:
                    best_label, min_dist = min(distances.items(), key=lambda item: item[1])
                    if min_dist <= self.threshold:
                        confidence = max(0.0, 1.0 - min_dist) * 100
                        recognized.append({
                            "label": best_label,
                            "confidence": round(confidence, 2),
                            "box": {"x": x, "y": y, "width": w, "height": h}
                        })
                    else:
                        unknown_count += 1
                else:
                    unknown_count += 1
            except Exception as e:
                logger.error(f"Recognition failed at ({x},{y}): {e}")
                unknown_count += 1

        return {"recognized": recognized, "unknown_faces": unknown_count}

    def _compute_distances(self, query: np.ndarray) -> dict[str, float]:
        """CPU-bound cosine distance computation."""
        norm_q = np.linalg.norm(query)
        if norm_q == 0:
            return dict.fromkeys(self.known_embeddings, 1.0)
        return {
            lbl: float(1.0 - np.dot(query, emb) / (norm_q * np.linalg.norm(emb)))
            for lbl, emb in self.known_embeddings.items()
        }
