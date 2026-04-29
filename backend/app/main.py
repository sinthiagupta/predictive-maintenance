from fastapi import FastAPI
from app.prediction.prediction_routes import router as prediction_router
from app.auth.auth_routes import router as auth_router
from app.config import APP_NAME, VERSION
from app.database import engine, Base
from app import models
from app.explain.explain_routes import router as analysis_router
from app.chatbot.chatbot_routes import router as chat_router

app = FastAPI(title=APP_NAME, version=VERSION)

Base.metadata.create_all(bind=engine)

app.include_router(auth_router)
app.include_router(prediction_router)
app.include_router(analysis_router)
app.include_router(chat_router)


@app.get("/")
def home():
    return {"message": "Backend is running"}