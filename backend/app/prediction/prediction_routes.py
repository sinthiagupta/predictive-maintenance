from fastapi import APIRouter, UploadFile, File
from .prediction_service import run_prediction
from sqlalchemy.orm import Session
from fastapi import Depends
from app.database import get_db
from app.auth.auth_utils import get_current_user
from app.models import Upload
import os
import pandas as pd

router = APIRouter(prefix="/predict", tags=["Prediction"])
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    result, df, predictions, probabilities = run_prediction(file_location)

    failures_predicted = sum(
        1 for p in predictions if str(p).lower() in ["failure", "f", "1"]
    )

    new_upload = Upload(
        file_name=file.filename,
        file_path=file_location,
        total_records=len(df),
        failures_predicted=failures_predicted,
        average_failure_probability=float(probabilities.mean()),
        user_id=current_user.id
    )

    db.add(new_upload)
    db.commit()
    db.refresh(new_upload)

    return {
    "upload_id": new_upload.id,
    "file_name": new_upload.file_name,
    "summary": {
        "total_records": len(df),
        "failures_predicted": failures_predicted,
        "failure_rate_percent": round((failures_predicted / len(df)) * 100, 2),
        "average_failure_probability": float(probabilities.mean())
    }
}

@router.get("/history")
def get_upload_history(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    uploads = db.query(Upload).filter(
        Upload.user_id == current_user.id
    ).order_by(Upload.id.desc()).all()

    return uploads