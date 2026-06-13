import os
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=_env_path, override=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load_data(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)
    non_numeric = df.select_dtypes(exclude='number').columns.tolist()
    df = df.drop(columns=[c for c in non_numeric if c != target_col])
    X = df.drop(columns=[target_col]).fillna(df.drop(columns=[target_col]).median(numeric_only=True))
    y = df[target_col]
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _make_objective(model_name: str, X_train, X_test, y_train, y_test):
    def objective(trial: optuna.Trial) -> float:
        if model_name == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=42,
            )
        elif model_name == "XGBoost":
            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            )
        elif model_name == "Logistic Regression":
            model = LogisticRegression(
                C=trial.suggest_float("C", 0.01, 10.0, log=True),
                max_iter=trial.suggest_int("max_iter", 100, 500),
                solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            )
        else:
            raise ValueError(f"No hyperparameter space defined for model: {model_name}")

        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    return objective


def _baseline_accuracy(model_name: str, X_train, X_test, y_train, y_test) -> float:
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=200)
    else:
        return 0.0
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))


def run_optimizer_agent(csv_path: str, target_col: str, best_model_name: str) -> dict:
    X_train, X_test, y_train, y_test = _load_data(csv_path, target_col)

    baseline = _baseline_accuracy(best_model_name, X_train, X_test, y_train, y_test)
    print(f"Baseline {best_model_name} accuracy: {baseline:.4f}")

    study = optuna.create_study(direction="maximize")
    study.optimize(_make_objective(best_model_name, X_train, X_test, y_train, y_test), n_trials=30)

    best_acc = round(study.best_value, 4)
    improvement = round(best_acc - baseline, 4)

    print(f"Best tuned accuracy : {best_acc:.4f}")
    print(f"Improvement         : {improvement:+.4f}")
    print(f"Best params         : {study.best_params}")

    return {
        "best_params": study.best_params,
        "best_accuracy": best_acc,
        "improvement": improvement,
    }


if __name__ == "__main__":
    sample_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.csv')

    for model_name in ["Random Forest", "XGBoost", "Logistic Regression"]:
        print(f"\n{'='*60}")
        print(f"Optimizing: {model_name}")
        print('='*60)
        result = run_optimizer_agent(sample_csv, "churn", model_name)
        if result["best_accuracy"] is not None:
            print(f"\nResult: accuracy={result['best_accuracy']}  improvement={result['improvement']:+.4f}")
            print(f"Params: {result['best_params']}")
