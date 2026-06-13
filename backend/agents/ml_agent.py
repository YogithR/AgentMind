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


def _preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # Step 1: Drop columns where >50% of values are missing (measured on X_train)
    missing_frac = X_train.isnull().mean()
    high_missing = missing_frac[missing_frac > 0.5].index.tolist()
    if high_missing:
        print(f"  Dropping high-missing cols (>50%): {high_missing}")
    X_train = X_train.drop(columns=high_missing)
    X_test = X_test.drop(columns=[c for c in high_missing if c in X_test.columns])

    # Step 5: Drop ID-like columns (every value unique in X_train)
    id_cols = [c for c in X_train.columns if X_train[c].nunique() == len(X_train)]
    if id_cols:
        print(f"  Dropping ID-like cols: {id_cols}")
    X_train = X_train.drop(columns=id_cols)
    X_test = X_test.drop(columns=[c for c in id_cols if c in X_test.columns])

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    # Step 2: Fill numeric missing with median (fit on X_train only)
    num_medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(num_medians)
    X_test[num_cols] = X_test[num_cols].fillna(num_medians)

    # Step 3: Fill categorical missing with mode (fit on X_train only)
    for col in cat_cols:
        mode_series = X_train[col].mode()
        fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        X_train[col] = X_train[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)

    # Step 4: One-hot encode categorical columns; align test to train columns
    if cat_cols:
        print(f"  One-hot encoding: {cat_cols}")
        X_train = pd.get_dummies(X_train, columns=cat_cols)
        X_test = pd.get_dummies(X_test, columns=cat_cols)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, X_test


def run_ml_agent(csv_path: str, target_col: str, eda_result: str) -> dict:
    df = pd.read_csv(csv_path)

    # Encode target if not already numeric
    y = df[target_col]
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Preprocessing...")
    X_train, X_test = _preprocess(X_train.copy(), X_test.copy())
    print(f"  Final feature count: {X_train.shape[1]}")

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
    titanic_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'titanic.csv')
    eda_stub = "EDA complete. Target: Survived. Classification problem."

    print(f"Training on: {titanic_csv}")
    print(f"Target column: Survived\n")

    result = run_ml_agent(titanic_csv, "Survived", eda_stub)

    print(f"\nBest model : {result['best_model_name']}")
    print(f"Accuracy   : {result['best_accuracy']}")
    print(f"F1 score   : {result['best_f1']}")
    print(f"\nAll results:")
    for name, metrics in result['all_results'].items():
        print(f"  {name}: accuracy={metrics['accuracy']}  f1={metrics['f1']}")
