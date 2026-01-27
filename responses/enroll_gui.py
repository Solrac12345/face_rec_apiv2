# GUI Enrollment System for Face Recognition
# Système d'enregistrement avec interface graphique (GUI)

import cv2
import requests
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

API_URL = "http://127.0.0.1:8000/face/recognize"
KNOWN_DIR = Path("known_faces")
KNOWN_DIR.mkdir(exist_ok=True)

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("Face Enrollment System / Système d'enregistrement")

video_label = tk.Label(root)
video_label.pack()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror("Error", "Cannot open webcam / Impossible d'ouvrir la webcam")
    root.destroy()


# -----------------------------
# Take Photo + Recognition Logic
# -----------------------------
def take_photo():
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image / Échec de capture")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        messagebox.showwarning("No Face", "No face detected / Aucun visage détecté")
        return

    (x, y, w, h) = faces[0]
    face_roi = frame[y:y+h, x:x+w]

    # Encode cropped face
    _, buffer = cv2.imencode(".jpg", face_roi)
    files = {"file": ("face.jpg", buffer.tobytes(), "image/jpeg")}

    try:
        response = requests.post(API_URL, files=files)
        data = response.json()
    except Exception as e:
        messagebox.showerror("API Error", f"API error: {e}")
        return

    recognized = data["recognized"]
    unknown_count = data["unknown_faces"]

    # -----------------------------
    # Case 1: Known person
    # -----------------------------
    if len(recognized) > 0:
        name = recognized[0]["label"]
        messagebox.showinfo(
            "Person Found",
            f"This person is in the database:\n{name}\n\n"
            f"Cette personne est dans la base de données :\n{name}"
        )
        return

    # -----------------------------
    # Case 2: Unknown person
    # -----------------------------
    if unknown_count > 0:
        save = messagebox.askyesno(
            "Unknown Person",
            "Unknown person detected.\n"
            "Do you want to save this person?\n\n"
            "Personne inconnue détectée.\n"
            "Voulez-vous enregistrer cette personne ?"
        )

        if not save:
            messagebox.showinfo("Cancelled", "Operation cancelled / Opération annulée")
            root.destroy()
            return

        # Ask for name
        name = simpledialog.askstring(
            "Enter Name",
            "Enter the person's name:\n\nEntrez le nom de la personne :"
        )

        if not name:
            messagebox.showwarning("No Name", "No name entered / Aucun nom entré")
            return

        # Save face
        timestamp = int(time.time())
        filename = KNOWN_DIR / f"{name}_{timestamp}.jpg"
        cv2.imwrite(str(filename), face_roi)

        messagebox.showinfo(
            "Saved",
            f"Face saved as {filename}\n\n"
            "Visage enregistré."
        )

        messagebox.showinfo(
            "Restart Required",
            "Restart the API to reload embeddings.\n\n"
            "Redémarrez l'API pour recharger les embeddings."
        )


# -----------------------------
# Live Webcam Feed
# -----------------------------
def update_frame():
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, update_frame)


# -----------------------------
# Button
# -----------------------------
btn = tk.Button(root, text="Take Photo / Prendre Photo", command=take_photo, font=("Arial", 14))
btn.pack(pady=10)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()