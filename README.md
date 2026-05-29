# Face Recognition API v2  
Multi-face detection and recognition using FastAPI + DeepFace (ArcFace)


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

### Problem & Solution

**Problem:**
Manual face verification is slow, error-prone, and difficult to scale across multiple people and live camera feeds.

**Solution:**
This API automates face detection and recognition using DeepFace with ArcFace embeddings, allowing fast multi-face matching from images or webcam input through a simple FastAPI interface.


### Run the API

```bash
uvicorn app.main:app --reload
```

### Run the files in another terminal

```bash
python webcam_client.py
python enroll_gui.py
python enroll_person.py
```
 ### License

This project is licensed under the MIT License. See the LICENSE file for details.