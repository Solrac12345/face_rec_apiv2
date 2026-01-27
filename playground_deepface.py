import cv2
from deepface import DeepFace

img = cv2.imread("test.jpg")
print("Image loaded:", img.shape)

reps = DeepFace.represent(
    img_path=img,
    model_name="ArcFace",
    detector_backend="opencv",
    enforce_detection=False,
)

print("Type:", type(reps))
print("Embedding length:", len(reps))