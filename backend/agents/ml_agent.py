import gc
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psutil
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=_env_path, override=True)

_DATE_KEYWORDS = ('date', 'time', 'year', 'month', 'day', 'timestamp')
_MAX_TRAIN_ROWS = 100_000
_MAX_ONEHOT_COLS = 200


def _preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Step 1: Drop date/time columns (high-cardinality, not useful raw, main source of column explosion)
    date_cols = [c for c in X_train.columns if any(word in c.lower() for word in _DATE_KEYWORDS)]
    if date_cols:
        print(f"  Dropping date/time cols: {date_cols}")
    X_train = X_train.drop(columns=date_cols, errors='ignore')
    X_test = X_test.drop(columns=date_cols, errors='ignore')

    # Drop columns where >50% of values are missing (measured on X_train)
    missing_frac = X_train.isnull().mean()
    high_missing = missing_frac[missing_frac > 0.5].index.tolist()
    if high_missing:
        print(f"  Dropping high-missing cols (>50%): {high_missing}")
    X_train = X_train.drop(columns=high_missing)
    X_test = X_test.drop(columns=[c for c in high_missing if c in X_test.columns])

    # Drop ID-like columns (every value unique in X_train)
    id_cols = [c for c in X_train.columns if X_train[c].nunique() == len(X_train)]
    if id_cols:
        print(f"  Dropping ID-like cols: {id_cols}")
    X_train = X_train.drop(columns=id_cols)
    X_test = X_test.drop(columns=[c for c in id_cols if c in X_test.columns])

    # Step 3: Drop high-cardinality categoricals (>20 unique) to keep one-hot encoding bounded
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    high_card = [c for c in cat_cols if X_train[c].nunique() > 20]
    if high_card:
        print(f"  Dropping high-cardinality categoricals (>20 unique): {high_card}")
    X_train = X_train.drop(columns=high_card)
    X_test = X_test.drop(columns=[c for c in high_card if c in X_test.columns])

    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

    # Fill numeric missing with median (fit on X_train only)
    num_medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(num_medians)
    X_test[num_cols] = X_test[num_cols].fillna(num_medians)

    # Fill categorical missing with mode (fit on X_train only)
    for col in cat_cols:
        mode_series = X_train[col].mode()
        fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        X_train[col] = X_train[col].fillna(fill_val)
        X_test[col] = X_test[col].fillna(fill_val)

    # Step 4: One-hot encode remaining categoricals; align test to train columns
    if cat_cols:
        print(f"  One-hot encoding: {cat_cols}")
        X_train = pd.get_dummies(X_train, columns=cat_cols)
        X_test = pd.get_dummies(X_test, columns=cat_cols)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Step 5: If still more than 200 columns, keep the most predictive 200 (fit on X_train only)
    if X_train.shape[1] > _MAX_ONEHOT_COLS:
        print(f"  Reducing {X_train.shape[1]} columns to {_MAX_ONEHOT_COLS} via SelectKBest...")
        selector = SelectKBest(f_classif, k=_MAX_ONEHOT_COLS)
        selector.fit(X_train, y_train)
        keep_cols = X_train.columns[selector.get_support()]
        X_train = X_train[keep_cols]
        X_test = X_test[keep_cols]

    # Step 6: Reduce memory - downcast float64 -> float32, int64 -> int32
    for frame in (X_train, X_test):
        for col in frame.select_dtypes(include=['float64']).columns:
            frame[col] = frame[col].astype('float32')
        for col in frame.select_dtypes(include=['int64']).columns:
            frame[col] = frame[col].astype('int32')

    gc.collect()
    return X_train, X_test


def _train_single_model(args):
    name, model, X_train, X_test, y_train, y_test = args
    try:
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        duration = round(time.time() - start, 2)
        print(f"  done: {name} -- accuracy={accuracy:.4f}  f1={f1:.4f}  ({duration}s)")
        return {
            "name": name,
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "model": model,
            "duration": duration,
        }
    except Exception as e:
        print(f"  failed: {name} -- {e}")
        return {"name": name, "accuracy": 0, "f1": 0, "model": None, "duration": 0}


def run_ml_agent(csv_path: str, target_col: str, eda_result: str) -> dict:
    df = pd.read_csv(csv_path)

    # Drop rows with missing target -- LabelEncoder would otherwise treat NaN as its own class
    df = df.dropna(subset=[target_col])

    # Encode target if not already numeric (keep as a Series aligned to df's index)
    y = df[target_col]
    if y.dtype == object:
        y = pd.Series(LabelEncoder().fit_transform(y), index=df.index, name=target_col)

    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 7: Stratified downsample of the training set for very large datasets
    if len(X_train) > _MAX_TRAIN_ROWS:
        print(f"  Training set has {len(X_train)} rows -- stratified sampling down to {_MAX_TRAIN_ROWS}")
        sss = StratifiedShuffleSplit(n_splits=1, train_size=_MAX_TRAIN_ROWS, random_state=42)
        idx, _ = next(sss.split(X_train, y_train))
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

    print("Preprocessing...")
    X_train, X_test = _preprocess(X_train.copy(), X_test.copy(), y_train)
    print(f"  Final feature count: {X_train.shape[1]}")

    models = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)),
        ("XGBoost", XGBClassifier(
            n_estimators=100, random_state=42,
            tree_method='hist', n_jobs=-1, verbosity=0)),
        ("LightGBM", LGBMClassifier(
            n_estimators=100, random_state=42,
            n_jobs=-1, verbose=-1)),
        ("CatBoost", CatBoostClassifier(
            iterations=100, random_state=42, verbose=0)),
        ("Logistic Regression", LogisticRegression(
            random_state=42, max_iter=500, n_jobs=-1)),
    ]

    n_cores = multiprocessing.cpu_count()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    print(f"Dataset shape  : {X_train.shape}")
    print(f"CPU cores      : {n_cores}")
    print(f"Memory before  : {mem_before:.0f}MB")
    print(f"Training {len(models)} models in parallel...")

    args_list = [
        (name, model, X_train, X_test, y_train, y_test)
        for name, model in models
    ]

    all_results = []
    with ThreadPoolExecutor(max_workers=min(len(models), n_cores)) as executor:
        futures = [executor.submit(_train_single_model, args) for args in args_list]
        for future in as_completed(futures):
            result = future.result()
            if result["accuracy"] > 0:
                all_results.append(result)

    mem_after = process.memory_info().rss / 1024 / 1024
    print(f"Memory after   : {mem_after:.0f}MB")

    if not all_results:
        raise RuntimeError("All models failed to train")

    best = max(all_results, key=lambda r: r["accuracy"])
    print(f"\nBest model: {best['name']} -- {best['accuracy']:.1%}")

    ranked = sorted(all_results, key=lambda r: r["accuracy"], reverse=True)

    return {
        "best_model_name": best["name"],
        "best_accuracy": best["accuracy"],
        "best_f1": best["f1"],
        "best_model": best["model"],
        "all_results": [
            {"name": r["name"], "accuracy": r["accuracy"], "f1": r["f1"], "duration": r["duration"]}
            for r in ranked
        ],
    }


if __name__ == "__main__":
    rain_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'rain_australia.csv')
    eda_stub = "EDA complete. Target: RainTomorrow. Classification problem."

    print(f"Training on: {rain_csv}")
    print(f"Target column: RainTomorrow\n")

    result = run_ml_agent(rain_csv, "RainTomorrow", eda_stub)

    print(f"\nBest model : {result['best_model_name']}")
    print(f"Accuracy   : {result['best_accuracy']}")
    print(f"F1 score   : {result['best_f1']}")
    print(f"\n{'Model':<22}{'Accuracy':<12}{'F1 Score':<12}{'Time (s)':<10}")
    print("-" * 56)
    for r in result['all_results']:
        marker = " *" if r["name"] == result["best_model_name"] else ""
        print(f"{r['name']:<22}{r['accuracy']:<12}{r['f1']:<12}{r['duration']:<10}{marker}")
