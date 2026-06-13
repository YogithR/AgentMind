import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=_env_path, override=True)


def run_ml_agent(csv_path: str, target_col: str, eda_result: str) -> dict:
    df = pd.read_csv(csv_path)

    # Drop non-numeric columns that can't be used as features
    non_numeric = df.select_dtypes(exclude='number').columns.tolist()
    feature_drop = [c for c in non_numeric if c != target_col]
    if feature_drop:
        print(f"Dropping non-numeric columns: {feature_drop}")
        df = df.drop(columns=feature_drop)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if not already numeric
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    # Fill missing values with column medians
    X = X.fillna(X.median(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    }

    all_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        all_results[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4), "model": model}
        print(f"  {name}: accuracy={acc:.4f}  f1={f1:.4f}")

    best_name = max(all_results, key=lambda n: all_results[n]["accuracy"])
    best = all_results[best_name]

    return {
        "best_model_name": best_name,
        "best_accuracy": best["accuracy"],
        "best_f1": best["f1"],
        "all_results": {n: {"accuracy": v["accuracy"], "f1": v["f1"]} for n, v in all_results.items()},
        "model": best["model"],
    }


if __name__ == "__main__":
    sample_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.csv')
    eda_stub = "EDA complete. Target: churn. Classification problem."

    print(f"Training on: {sample_csv}")
    print(f"Target column: churn\n")

    result = run_ml_agent(sample_csv, "churn", eda_stub)

    print(f"\nBest model : {result['best_model_name']}")
    print(f"Accuracy   : {result['best_accuracy']}")
    print(f"F1 score   : {result['best_f1']}")
    print(f"\nAll results:")
    for name, metrics in result['all_results'].items():
        print(f"  {name}: accuracy={metrics['accuracy']}  f1={metrics['f1']}")
