import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=_env_path, override=True)

_api_key = os.getenv("GROQ_API_KEY")
print(f".env path : {_env_path}")
print("API KEY FOUND" if _api_key else "API KEY MISSING")

_client = Groq(api_key=_api_key)
_MODEL = "llama-3.1-8b-instant"


def _build_summary(df: pd.DataFrame) -> str:
    lines = []

    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    lines.append("\nColumn types:")
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col}: {dtype}")

    lines.append("\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            lines.append(f"  {col}: {count} ({count / len(df) * 100:.1f}%)")
    if missing.sum() == 0:
        lines.append("  None")

    lines.append("\nBasic statistics:")
    lines.append(df.describe().to_string())

    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] >= 2:
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        top = upper.stack().sort_values(ascending=False).head(10)
        lines.append("\nTop correlations:")
        for (col_a, col_b), val in top.items():
            lines.append(f"  {col_a} <-> {col_b}: {val:.3f}")

    return "\n".join(lines)


def run_eda_agent(csv_path: str, goal: str, past_experiments: str = "") -> str:
    df = pd.read_csv(csv_path)
    summary = _build_summary(df)

    memory_block = f"\n\n{past_experiments}\n" if past_experiments else ""

    prompt = (
        "You are a data scientist. Here is a dataset summary:\n"
        f"{summary}\n\n"
        f"The user's goal is: {goal}\n"
        f"{memory_block}\n"
        "Describe the data, identify if this is a classification or regression problem, "
        "list the most important columns, and recommend the best ML approach."
    )

    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample.csv')

    if os.path.exists(sample_path):
        print(f"Using existing dataset: {sample_path}")
        csv_path = sample_path
    else:
        print("No sample.csv found — generating fake dataset.")
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "age":             rng.integers(18, 70, n),
            "income":          rng.normal(50000, 15000, n).round(2),
            "debt":            rng.normal(10000, 5000, n).round(2),
            "credit_score":    rng.integers(300, 850, n),
            "missed_payments": rng.integers(0, 10, n),
            "defaulted":       rng.choice([0, 1], n, p=[0.8, 0.2]),
        })
        df.loc[rng.choice(n, 10, replace=False), "income"] = np.nan
        csv_path = sample_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved fake dataset to {csv_path}")

    goal = "Predict whether a customer will churn."
    print(f"\nGoal: {goal}\n")
    print("=" * 60)
    result = run_eda_agent(csv_path, goal)
    print(result)
