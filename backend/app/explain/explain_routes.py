from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Upload
import pandas as pd
from app.reports.llm_client import call_llm
from .shap_engine import generate_shap_report

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.get("/upload/{upload_id}")
def shap_full_upload(upload_id: int, db: Session = Depends(get_db)):

    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        return {"error": "Upload not found"}

    df = pd.read_csv(upload.file_path)

    # ✅ Generate SHAP report ONCE
    shap_data = generate_shap_report(df)

    # =========================
    # Build Clean Intelligence Context
    # =========================

    failure_rate = round(
        (upload.failures_predicted / upload.total_records) * 100, 2
    )

    # Extract ONLY feature names (no scores, no dictionaries)
    global_drivers = [f["feature"] for f in shap_data["global_feature_ranking"][:5]]
    failure_drivers = [f["feature"] for f in shap_data["failure_only_ranking"][:5]]
    escalation_drivers = [f["feature"] for f in shap_data["failure_push_ratio"][:5]]

    risk_seg = shap_data["risk_segmentation"]

    context_summary = f"""
    Fleet Overview:
    - Total Machines Analyzed: {upload.total_records}
    - Predicted Failures: {upload.failures_predicted}
    - Failure Rate: {failure_rate}%

    Risk Segmentation:
    - High Risk Machines: {risk_seg['high_risk_machines']}
    - Medium Risk Machines: {risk_seg['medium_risk_machines']}
    - Low Risk Machines: {risk_seg['low_risk_machines']}

    Primary Operational Stress Drivers:
    {global_drivers}

    Failure Escalation Drivers:
    {failure_drivers}

    Factors Most Frequently Increasing Risk:
    {escalation_drivers}
    """

    # =========================
    # Professional Prompt
    # =========================

    prompt = f"""
You are a senior industrial reliability and mechanical systems consultant.

Generate a comprehensive predictive maintenance intelligence report.

CRITICAL RULES:
- Do NOT mention SHAP.
- Do NOT mention model internals.
- Do NOT use feature names like num__, cat__, rpm, etc.
- Do NOT show raw data, dictionaries, or technical variable names.
- Translate all signals into operational mechanical meaning.
- Explain WHY failures are occurring.
- Explain which machine behaviors are mechanically dangerous.
- Provide engineering-level reasoning.
- Write in structured sections.
- Minimum 1000 words.

Required Structure:

1. Executive Summary
2. Operational Risk Overview
3. Dominant Mechanical Stress Factors
4. Failure Behavior Patterns
5. Systemic Risk Concentration
6. Maintenance Strategy Recommendations
7. Strategic Business Impact

Use this operational intelligence:

{context_summary}

Produce a board-level technical reliability report.
"""

    professional_report = call_llm(prompt)

    return {
        "upload_id": upload_id,
        "report": professional_report
    }