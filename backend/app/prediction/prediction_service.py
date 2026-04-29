import pandas as pd
from .model_loader import get_pipeline

def run_prediction(file_path):

    df = pd.read_csv(file_path)

    pipeline = get_pipeline()

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    result = {
        "total_records": len(predictions),
        "failures_predicted": int(sum(predictions)),
        "failure_rate_percent": round((sum(predictions) / len(predictions)) * 100, 2),
        "average_failure_probability": float(probabilities.mean())
    }

    return result, df, predictions, probabilities