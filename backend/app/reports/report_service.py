import pandas as pd
from app.explain.shap_engine import generate_shap_report
from .llm_client import call_llm


# =====================================================
# CLEAN CONTEXT BUILDER
# =====================================================

def build_structured_context(shap_data: dict) -> str:
    """
    Convert SHAP output into executive-level
    analytical context (no raw dictionaries).
    """

    summary = shap_data["dataset_summary"]
    risk_seg = shap_data["risk_segmentation"]

    # Extract only feature names (not scores)
    global_drivers = [
        f["feature"]
        for f in shap_data["global_feature_ranking"][:5]
    ]

    failure_drivers = [
        f["feature"]
        for f in shap_data["failure_only_ranking"][:5]
    ]

    push_drivers = [
        f["feature"]
        for f in shap_data["failure_push_ratio"][:5]
    ]

    context = f"""
Fleet Overview:
- Total Machines: {summary['total_records']}
- Predicted Failures: {summary['total_failures']}
- Failure Rate: {round(summary['failure_rate_percent'], 2)}%
- Average Failure Probability: {round(summary['average_failure_probability'], 4)}

Risk Segmentation:
- High Risk Machines: {risk_seg['high_risk_machines']}
- Medium Risk Machines: {risk_seg['medium_risk_machines']}
- Low Risk Machines: {risk_seg['low_risk_machines']}

Primary Operational Stress Signals:
{global_drivers}

Failure Escalation Signals:
{failure_drivers}

Signals Frequently Increasing Risk:
{push_drivers}
"""

    return context


# =====================================================
# MAIN PROFESSIONAL REPORT GENERATOR
# =====================================================

def generate_professional_report(df: pd.DataFrame):

    shap_data = generate_shap_report(df)

    context = build_structured_context(shap_data)

    prompt = f"""
You are a senior industrial reliability and mechanical systems strategist.

Generate a board-level predictive maintenance intelligence report.

CRITICAL RULES:
- Do NOT mention SHAP.
- Do NOT mention model internals.
- Do NOT repeat feature variable names like num__, cat__, rpm, etc.
- Translate all technical signals into operational mechanical behavior.
- Explain WHY failures are occurring.
- Explain systemic stress interactions.
- Provide engineering-level reasoning.
- Minimum 1000 words.
- Use structured professional sections.

Required Structure:

1. Executive Summary
2. Operational Risk Landscape
3. Dominant Mechanical Stress Drivers
4. Failure Escalation Patterns
5. Systemic Risk Concentration
6. Maintenance Strategy Roadmap
7. Strategic Business Impact

Operational Intelligence Context:
{context}

Write in formal technical consulting style.
"""

    narrative = call_llm(prompt)

    # 🔥 Return ONLY professional report (clean output)
    return {
        "report": narrative,
        "metrics": shap_data["dataset_summary"],
        "charts": {
            "risk_segmentation": shap_data["risk_segmentation"],
            "global_feature_ranking": shap_data["global_feature_ranking"][:10],
            "failure_only_ranking": shap_data["failure_only_ranking"][:10]
        }
}