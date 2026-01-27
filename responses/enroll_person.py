# Interactive enrollment script (SPACE to capture)
# Script interactif d'enregistrement (ESPACE pour capturer)

import cv2
import requests
import time
from pathlib import Path

API_URL = "http://127.0.0.1:8000/face/recognize"
KNOWN_DIR = Path("known_faces")

# Create directory if missing
# Créer le dossier s'il n'existe pas
KNOWN_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam / Impossible d'ouvrir la webcam")
    exit()

print("Enrollment mode started.")
print("Press SPACE to take a photo, Q to quit.")
print("Mode d'enregistrement démarré.")
print("Appuyez sur ESPACE pour prendre une photo, Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame / Échec de capture")
        break

    # Display webcam feed
    # Afficher le flux de la webcam
    cv2.imshow("Enrollment Mode", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit
    # Quitter
    if key == ord("q"):
        break

    # SPACE pressed → capture photo
    # ESPACE pressé → capturer la photo
    if key == 32:  # 32 = SPACE
        print("Photo captured / Photo capturée")

        # Encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

        try:
            response = requests.post(API_URL, files=files)
            data = response.json()
        except Exception as e:
            print("API error / Erreur API:", e)
            continue

        recognized = data["recognized"]
        unknown_count = data["unknown_faces"]

        # -----------------------------
        # Case 1: Known person detected
        # -----------------------------
        if len(recognized) > 0:
            person = recognized[0]["label"]
            print(f"This person is in my database: {person}")
            print(f"Cette personne est dans ma base de données : {person}")

        # -----------------------------
        # Case 2: Unknown person
        # -----------------------------
        elif unknown_count > 0:
            print("Unknown person detected.")
            print("Personne inconnue détectée.")

            # Ask user if they want to add
            choice = input("Add this person to database? (y/n) / Ajouter cette personne (o/n): ").strip().lower()

            if choice in ["y", "o"]:
                name = input("Enter the name / Entrez le nom: ").strip()

                # Save face image
                timestamp = int(time.time())
                filename = KNOWN_DIR / f"{name}_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)

                print(f"Saved new face as {filename}")
                print("Nouveau visage enregistré.")

                print("Restart the API to reload embeddings.")
                print("Redémarrez l'API pour recharger les embeddings.")