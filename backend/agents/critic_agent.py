import os
import warnings
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from groq import Groq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=_env_path, override=True)

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_MODEL = "llama-3.1-8b-instant"


def _load_data(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    non_numeric = df.select_dtypes(exclude='number').columns.tolist()
    df = df.drop(columns=[c for c in non_numeric if c != target_col])
    X = df.drop(columns=[target_col]).fillna(df.drop(columns=[target_col]).median(numeric_only=True))
    y = df[target_col]
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _build_model(model_name: str, params: dict):
    if model_name == "Random Forest":
        return RandomForestClassifier(**params, random_state=42)
    elif model_name == "XGBoost":
        return XGBClassifier(**params, random_state=42, eval_metric="logloss", verbosity=0)
    elif model_name == "Logistic Regression":
        return LogisticRegression(**params)
    raise ValueError(f"Unknown model: {model_name}")


def _shap_top_features(model_name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name in ("Random Forest", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)

    # For binary classifiers shap_values may be a list [class0_arr, class1_arr] — use class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(X_test.columns.tolist(), mean_abs), key=lambda x: x[1], reverse=True)
    return [(name, round(float(val), 4)) for name, val in ranked[:5]]


def _detect_warnings(csv_path: str, target_col: str, ml_result: dict, optimized_result: dict) -> list:
    found = []

    df = pd.read_csv(csv_path)

    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.2]
    if not high_missing.empty:
        cols = ", ".join(f"{c} ({v:.0%})" for c, v in high_missing.items())
        found.append(f"High missing data: {cols}")

    if target_col in df.columns:
        counts = df[target_col].value_counts(normalize=True)
        if counts.min() < 0.25:
            found.append(f"Class imbalance: minority class is {counts.min():.0%} of data")

    acc = optimized_result.get("best_accuracy") or ml_result.get("best_accuracy", 0)
    if acc < 0.65:
        found.append(f"Low accuracy ({acc:.2%}) — model may not be reliable for production use")

    improvement = optimized_result.get("improvement", 0)
    if improvement > 0.10:
        found.append(
            f"Large tuning gain (+{improvement:.2%}) — verify optimized params on a held-out validation set"
        )

    if len(df) < 500:
        found.append(f"Small dataset ({len(df)} rows) — results may not generalise well")

    return found


def run_critic_agent(
    csv_path: str,
    target_column: str,
    eda_result: str,
    ml_result: dict,
    optimized_result: dict,
) -> dict:
    model_name = ml_result["best_model_name"]
    best_params = optimized_result.get("best_params", {})
    accuracy = optimized_result.get("best_accuracy") or ml_result["best_accuracy"]

    print(f"  Retraining {model_name} with optimized params: {best_params}")
    X_train, X_test, y_train, y_test = _load_data(csv_path, target_column)

    model = _build_model(model_name, best_params)
    model.fit(X_train, y_train)
    retrained_acc = round(accuracy_score(y_test, model.predict(X_test)), 4)
    print(f"  Retrained accuracy: {retrained_acc:.4f}")

    print("  Computing SHAP feature importances...")
    top_features = _shap_top_features(model_name, model, X_train, X_test)
    features_str = ", ".join(f"{name} (SHAP={val})" for name, val in top_features)
    print(f"  Top features: {features_str}")

    warnings_list = _detect_warnings(csv_path, target_column, ml_result, optimized_result)

    prompt = (
        f"You are an expert ML reviewer. The best model is {model_name} with accuracy {accuracy:.2%}.\n"
        f"Top 5 most important features are: {features_str}.\n"
        "Describe what this means in plain English for a business user.\n"
        "Check for any concerns like overfitting or data quality issues.\n"
        "Give 3 actionable recommendations."
    )

    print("  Sending to Groq for critique...")
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    critique = response.choices[0].message.content

    return {
        "critique": critique,
        "top_features": top_features,
        "warnings": warnings_list,
        "retrained_accuracy": retrained_acc,
    }


if __name__ == "__main__":
    titanic_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'titanic.csv')

    ml_result = {
        "best_model_name": "Logistic Regression",
        "best_accuracy": 0.7318,
        "best_f1": 0.7180,
    }
    optimized_result = {
        "best_params": {"C": 0.255, "max_iter": 216, "solver": "liblinear"},
        "best_accuracy": 0.7430,
        "improvement": 0.0112,
    }

    print("=" * 60)
    print("Critic Agent — Titanic dataset")
    print("=" * 60)

    result = run_critic_agent(
        csv_path=titanic_csv,
        target_column="Survived",
        eda_result="",
        ml_result=ml_result,
        optimized_result=optimized_result,
    )

    print("\n" + "=" * 60)
    print("CRITIQUE")
    print("=" * 60)
    print(result["critique"])

    print("\n" + "=" * 60)
    print("TOP FEATURES")
    print("=" * 60)
    for name, val in result["top_features"]:
        print(f"  {name}: {val}")

    print("\n" + "=" * 60)
    print("WARNINGS")
    print("=" * 60)
    if result["warnings"]:
        for w in result["warnings"]:
            print(f"  ! {w}")
    else:
        print("  None")
