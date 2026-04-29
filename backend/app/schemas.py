from pydantic import BaseModel

class PredictionResponse(BaseModel):
    total_records: int
    failures_predicted: int
    failure_rate_percent: float
    average_failure_probability: float