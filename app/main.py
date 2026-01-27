# FastAPI main entry point
# Point d'entrée principal FastAPI

from fastapi import FastAPI
from app.routes.face_routes import router as face_router

app = FastAPI(title="Face Recognition API v2")

# Register face routes
# Enregistrer les routes de reconnaissance faciale
app.include_router(face_router)


@app.get("/health")
def health():
    # Simple health check
    # Vérification simple de l'état du service
    return {"status": "ok"}