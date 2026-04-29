import joblib
from pathlib import Path
import sys

# Import your class
from app.prediction.feature_engineering import FeatureEngineer

# 🔥 CRITICAL FIX
# Register FeatureEngineer under __main__
sys.modules["__main__"].FeatureEngineer = FeatureEngineer

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)

def get_pipeline():
    return pipeline