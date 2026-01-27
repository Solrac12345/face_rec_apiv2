# Auto-enroll mode for face recognition
# Mode d'auto-enregistrement pour la reconnaissance faciale

import cv2
import requests
import time
from pathlib import Path

API_URL = "http://127.0.0.1:8000/face/recognize"
KNOWN_DIR = Path("known_faces")

# Ensure known_faces directory exists
# S'assurer que le dossier known_faces existe
KNOWN_DIR.mkdir(exist_ok=True)

# Load OpenCV Haar cascade for local detection
# Charger le cascade Haar d'OpenCV pour la détection locale
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam / Impossible d'ouvrir la webcam")
    exit()

print("Auto-enroll mode started.")
print("Press Q to quit.")
print("Mode auto-enregistrement démarré.")
print("Appuyez sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame / Échec de capture")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces locally
    # Détecter les visages localement
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # first detected face
        face_roi = frame[y:y+h, x:x+w]

        # Encode cropped face
        _, buffer = cv2.imencode(".jpg", face_roi)
        files = {"file": ("face.jpg", buffer.tobytes(), "image/jpeg")}

        try:
            response = requests.post(API_URL, files=files)
            data = response.json()
        except Exception as e:
            print("API error / Erreur API:", e)
            continue

        recognized = data["recognized"]
        unknown_count = data["unknown_faces"]

        # -----------------------------
        # Case 1: Known person
        # -----------------------------
        if len(recognized) > 0:
            label = recognized[0]["label"]
            conf = recognized[0]["confidence"]

            cv2.putText(
                frame,
                f"{label} ({conf:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # -----------------------------
        # Case 2: Unknown person
        # -----------------------------
        elif unknown_count > 0:
            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

            print("\nUnknown person detected.")
            print("Personne inconnue détectée.")

            # Ask if user wants to enroll
            choice = input("Enroll this person? (y/n) / Enregistrer cette personne (o/n): ").strip().lower()

            if choice in ["y", "o"]:
                name = input("Enter the name / Entrez le nom: ").strip()

                timestamp = int(time.time())
                filename = KNOWN_DIR / f"{name}_{timestamp}.jpg"

                # Save cropped face
                cv2.imwrite(str(filename), face_roi)

                print(f"Saved new face as {filename}")
                print("Nouveau visage enregistré.")

                print("Restart the API to reload embeddings.")
                print("Redémarrez l'API pour recharger les embeddings.")

    # Show webcam feed
    cv2.imshow("Auto-Enroll Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()