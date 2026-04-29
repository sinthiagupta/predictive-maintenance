import shap
import numpy as np
import pandas as pd
import os
from app.prediction.model_loader import get_pipeline

pipeline = get_pipeline()


# =====================================================
# TRANSFORMATION
# =====================================================

def _transform_data(df: pd.DataFrame):
    X_eng = pipeline.named_steps["feature_engineering"].transform(df)
    X_proc = pipeline.named_steps["preprocessing"].transform(X_eng)
    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    return X_proc, feature_names


# =====================================================
# SHAP COMPUTATION (FULL DATASET – NO SAMPLING)
# =====================================================

def compute_shap(df: pd.DataFrame):

    X_proc, feature_names = _transform_data(df)

    model = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_proc)

    # Handle binary classification output shape
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, feature_names, X_proc


# =====================================================
# GLOBAL IMPORTANCE
# =====================================================

def global_importance(shap_values, feature_names):

    mean_abs = np.abs(shap_values).mean(axis=0)

    ranking = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {"feature": f, "importance_score": float(score)}
        for f, score in ranking
    ]


# =====================================================
# FAILURE-ONLY IMPORTANCE
# =====================================================

def failure_only_importance(shap_values, feature_names, predictions):

    failed_indices = np.where(predictions == 1)[0]

    if len(failed_indices) == 0:
        return []

    failed_shap = shap_values[failed_indices]

    mean_abs = np.abs(failed_shap).mean(axis=0)

    ranking = sorted(
        zip(feature_names, mean_abs),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {"feature": f, "failure_importance_score": float(score)}
        for f, score in ranking
    ]


# =====================================================
# FAILURE PUSH RATIO
# =====================================================

def feature_failure_push_ratio(shap_values, feature_names, predictions):

    failed_indices = np.where(predictions == 1)[0]

    results = []

    for i, feature in enumerate(feature_names):

        contributions = shap_values[failed_indices, i]

        total = len(contributions)

        if total == 0:
            ratio = 0
        else:
            ratio = float((contributions > 0).sum() / total)

        results.append({
            "feature": feature,
            "positive_push_ratio_in_failures": ratio
        })

    return results


# =====================================================
# MACHINE RISK INTENSITY
# =====================================================

def machine_risk_intensity(shap_values, index):

    contributions = shap_values[index]
    return float(np.sum(np.abs(contributions)))


# =====================================================
# MACHINE EXPLANATION (TOP CONTRIBUTORS ONLY)
# =====================================================

def machine_explanation(shap_values, feature_names, index):

    contributions = shap_values[index]

    ranking = sorted(
        zip(feature_names, contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return [
        {
            "feature": f,
            "contribution": float(val),
            "direction": "increase_failure" if val > 0 else "reduce_failure"
        }
        for f, val in ranking[:5]
    ]


# =====================================================
# MACHINE DEVIATION ANALYSIS
# =====================================================

def machine_deviation_analysis(df, index, predictions):

    machine_row = df.iloc[index]

    failed_df = df[predictions == 1]
    normal_df = df[predictions == 0]

    analysis = []

    for col in df.select_dtypes(include=["int64", "float64"]).columns:

        machine_value = machine_row[col]
        normal_mean = normal_df[col].mean()
        failed_mean = failed_df[col].mean()

        deviation_from_normal = machine_value - normal_mean

        analysis.append({
            "feature": col,
            "machine_value": float(machine_value),
            "normal_mean": float(normal_mean),
            "failed_mean": float(failed_mean),
            "deviation_from_normal": float(deviation_from_normal)
        })

    return analysis[:10]


# =====================================================
# MAIN REPORT GENERATOR (FULL DATASET)
# =====================================================

def generate_shap_report(df):

    # ❌ REMOVED 500 SAMPLING
    # FULL DATASET WILL BE USED

    shap_values, feature_names, X_proc = compute_shap(df)

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    total_records = len(df)
    total_failures = int(predictions.sum())
    failure_rate = float((total_failures / total_records) * 100)

    global_rank = global_importance(shap_values, feature_names)

    failure_rank = failure_only_importance(
        shap_values,
        feature_names,
        predictions
    )

    push_ratio = feature_failure_push_ratio(
        shap_values,
        feature_names,
        predictions
    )

    high_risk = int((probabilities > 0.8).sum())
    medium_risk = int(((probabilities > 0.5) & (probabilities <= 0.8)).sum())
    low_risk = int((probabilities <= 0.5).sum())

    # Only analyze failed machines to reduce memory pressure
    machine_analysis = []
    failed_indices = np.where(predictions == 1)[0]

    for idx in failed_indices:

        machine_analysis.append({
            "machine_index": int(idx),
            "prediction_probability": float(probabilities[idx]),
            "risk_intensity_score": machine_risk_intensity(shap_values, idx),
            "top_contributors": machine_explanation(
                shap_values,
                feature_names,
                idx
            ),
            "feature_deviation_analysis": machine_deviation_analysis(
                df,
                idx,
                predictions
            )
        })

    return {
        "dataset_summary": {
            "total_records": total_records,
            "total_failures": total_failures,
            "failure_rate_percent": failure_rate,
            "average_failure_probability": float(probabilities.mean())
        },
        "global_feature_ranking": global_rank,
        "failure_only_ranking": failure_rank,
        "failure_push_ratio": push_ratio,
        "risk_segmentation": {
            "high_risk_machines": high_risk,
            "medium_risk_machines": medium_risk,
            "low_risk_machines": low_risk
        },
        "machine_level_analysis": machine_analysis
    }