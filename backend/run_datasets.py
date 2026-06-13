import os
import sys

_BACKEND = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BACKEND, "agents"))
sys.path.insert(0, os.path.join(_BACKEND, "graph"))

from pipeline import AgentState, build_pipeline

DATASETS = [
    {
        "name": "Telco Customer Churn",
        "csv":  os.path.join(_BACKEND, "data", "telco_churn.csv"),
        "target": "Churn",
        "goal": "predict which customers will cancel their subscription",
    },
    {
        "name": "Heart Disease",
        "csv":  os.path.join(_BACKEND, "data", "heart.csv"),
        "target": "target",
        "goal": "predict which patients are at risk of heart disease",
    },
]


def run_one(dataset: dict) -> dict:
    print(f"\n{'='*64}")
    print(f"  DATASET: {dataset['name']}")
    print(f"{'='*64}")

    initial_state: AgentState = {
        "csv_path":       dataset["csv"],
        "goal":           dataset["goal"],
        "target_column":  dataset["target"],
        "eda_result":     "",
        "ml_result":      {},
        "optimized_result": {},
        "critic_result":  {},
        "critique":       "",
        "top_features":   [],
        "warnings":       [],
        "experiment_id":  "",
        "status":         "start",
    }

    pipeline = build_pipeline()
    final = pipeline.invoke(initial_state)

    ml  = final["ml_result"]
    opt = final["optimized_result"]
    top = final["top_features"]

    accuracy    = opt.get("best_accuracy") or ml.get("best_accuracy", 0)
    improvement = opt.get("improvement", 0.0)
    top_feature = top[0][0] if top else "N/A"

    return {
        "dataset":     dataset["name"],
        "best_model":  ml.get("best_model_name", "N/A"),
        "accuracy":    accuracy,
        "top_feature": top_feature,
        "improvement": improvement,
    }


if __name__ == "__main__":
    rows = []
    for ds in DATASETS:
        rows.append(run_one(ds))

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  FINAL SUMMARY")
    print(f"{'='*90}")
    header = f"{'Dataset':<28}  {'Best Model':<22}  {'Accuracy':>8}  {'Top Feature':<20}  {'Improvement':>11}"
    print(header)
    print("-" * 90)
    for r in rows:
        print(
            f"{r['dataset']:<28}  "
            f"{r['best_model']:<22}  "
            f"{r['accuracy']*100:>7.2f}%  "
            f"{r['top_feature']:<20}  "
            f"{'+' if r['improvement'] >= 0 else ''}{r['improvement']*100:>9.2f}%"
        )
    print("=" * 90)
