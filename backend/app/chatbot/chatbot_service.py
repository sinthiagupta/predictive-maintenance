from app.reports.llm_client import call_llm
from app.database import SessionLocal
from app.models import Upload
from app.explain.shap_engine import generate_shap_report
import pandas as pd


# =====================================================
# DOMAIN FILTER (Optional but Recommended)
# =====================================================



def is_domain_related(message: str) -> bool:
    message = message.lower()
    keywords = [
        "machine",
        "failure",
        "maintenance",
        "risk",
        "dataset",
        "data",
        "analysis",
        "report",
        "prediction",
        "probability",
        "explain",
        "summary"
    ]
    return any(keyword in message for keyword in keywords)


# =====================================================
# MAIN CHATBOT LOGIC
# =====================================================

def generate_chat_response(user_message: str, dataset_id: int) -> str:

    # 🔒 Hard domain restriction
    if not is_domain_related(user_message):
        return (
            "I'm designed specifically for predictive maintenance and "
            "machine reliability insights. I cannot assist with unrelated topics."
        )
    db = SessionLocal()
    upload = db.query(Upload).filter(
    Upload.id == dataset_id
    ).first()
    if not upload:
        return "No dataset found for this ID."
    print("DEBUG USER ID:", user_id)
    print("DEBUG UPLOAD FOUND:", upload)
    if not upload:
        return "No dataset found for this user."
    df = pd.read_csv(upload.file_path)
    shap_data = generate_shap_report(df)

    summary = shap_data["dataset_summary"]
    risk_seg = shap_data["risk_segmentation"]

    context_summary = f"""
    Dataset Overview:
    - Total Machines: {summary['total_records']}
    - Predicted Failures: {summary['total_failures']}
    - Failure Rate: {round(summary['failure_rate_percent'], 2)}%

    Risk Segmentation:
    - High Risk Machines: {risk_seg['high_risk_machines']}
    - Medium Risk Machines: {risk_seg['medium_risk_machines']}
    - Low Risk Machines: {risk_seg['low_risk_machines']}
    """

    system_prompt = f"""
    You are a senior industrial reliability AI advisor integrated
    inside a predictive maintenance analytics platform.

    Your role:
    - Help engineers understand machine health
    - Explain failure mechanisms
    - Interpret operational risk
    - Suggest maintenance strategies

    STRICT RULES:
    - Stay strictly within predictive maintenance domain.
    - Do NOT answer unrelated questions.
    - Do NOT mention model internals.
    - Do NOT mention SHAP.
    - Translate technical signals into mechanical reasoning.
    - Provide structured, professional responses.
    - Be concise but insightful.
    If the question is outside maintenance domain,
        respond with:
        "I'm designed specifically for predictive maintenance insights and cannot assist with that topic."

    Dataset Context:
    {context_summary}

    User Question:
    {user_message}

    Provide a clear engineering-level response.
    """

    response = call_llm(system_prompt)

    return response