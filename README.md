# Face Recognition API v2

Multi-face detection and recognition using FastAPI + DeepFace (ArcFace).

## Implements

This project implements a modular face recognition API using:

- FastAPI
- DeepFace (ArcFace model, 512-dimensional embeddings)
- OpenCV (face detection)
- Python 3.10

### Features

- Multi-face detection
- Multi-face recognition
- Embedding database from `known_faces/`
- Webcam-based recognition demo
- Clean architecture (services, routes, DTOs)

### Problem & Solution

**Problem:**
Manual face verification is slow, error-prone, and difficult to scale across multiple people and live camera feeds.

**Solution:**
This API automates face detection and recognition with DeepFace and ArcFace embeddings, allowing fast matching from uploaded images or webcam input through a simple FastAPI interface.

### Run the API

```bash
uvicorn app.main:app --reload
```

### Run the demo scripts from another terminal

```bash
python responses/webcam_client.py
python responses/enroll_gui.py
python responses/enroll_person.py
```

### API endpoints

- `POST /face/detect` — detect faces in an uploaded image
- `POST /face/recognize` — recognize faces in an uploaded image
- `GET /health` — health check