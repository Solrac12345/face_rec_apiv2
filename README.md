# Face Recognition API v2  
Multi-face detection and recognition using FastAPI + DeepFace (ArcFace)

## English

This project implements a clean, modular face recognition API using:

- FastAPI
- DeepFace (ArcFace model, 512‑dim embeddings)
- OpenCV (face detection)
- Python 3.10

### Features
- Multi-face detection
- Multi-face recognition
- Embedding database from `known_faces/`
- Webcam client for real-time recognition
- Clean architecture (services, routes, DTOs)
- Fully documented (EN + FR)

### Run the API

```bash
uvicorn app.main:app --reload

###  Run the files another terminal
```bash
python webcam_client.py
python enroll_gui.py 
python enroll_person.py
