import json
import os
import uuid
from datetime import datetime

import chromadb

_MEMORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'memory'))
_COLLECTION = "experiments"

_client = chromadb.PersistentClient(path=_MEMORY_DIR)
_collection = _client.get_or_create_collection(_COLLECTION)


def save_experiment(state: dict) -> str:
    """Persist a completed pipeline run to ChromaDB. Returns the document id."""
    ml = state.get("ml_result", {})
    opt = state.get("optimized_result", {})
    critic = state.get("critic_result", {})

    model_name = ml.get("best_model_name", "unknown")
    accuracy = opt.get("best_accuracy") or ml.get("best_accuracy", 0.0)
    top_features = critic.get("top_features", [])
    dataset_name = os.path.basename(state.get("csv_path", "unknown"))
    goal = state.get("goal", "")
    timestamp = datetime.utcnow().isoformat()

    # The document text is what ChromaDB embeds for similarity search
    document = (
        f"Goal: {goal}\n"
        f"Dataset: {dataset_name}\n"
        f"Best model: {model_name} (accuracy={accuracy:.4f})\n"
        f"Top features: {', '.join(f[0] if isinstance(f, (list, tuple)) else str(f) for f in top_features)}"
    )

    metadata = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "dataset_name": dataset_name,
        "goal": goal,
        "top_features": json.dumps(top_features),
        "timestamp": timestamp,
    }

    doc_id = str(uuid.uuid4())
    _collection.add(documents=[document], metadatas=[metadata], ids=[doc_id])
    print(f"  [Memory] Saved experiment {doc_id[:8]}... ({model_name}, acc={accuracy:.4f})")
    return doc_id


def get_similar_experiments(goal: str, n: int = 3) -> str:
    """
    Query ChromaDB for the n most similar past runs.
    Returns a formatted string ready to inject into a prompt, or '' if none exist.
    """
    count = _collection.count()
    if count == 0:
        return ""

    results = _collection.query(
        query_texts=[goal],
        n_results=min(n, count),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return ""

    lines = ["=== Similar past experiments (from memory) ==="]
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        lines.append(
            f"\n[Experiment {i}] {meta.get('timestamp', '')[:19]}\n{doc}"
        )
    lines.append("=== End of past experiments ===")
    return "\n".join(lines)


def get_recent_experiments(n: int = 5) -> list:
    """Return the n most recent experiments ordered by timestamp descending."""
    count = _collection.count()
    if count == 0:
        return []

    all_docs = _collection.get(include=["documents", "metadatas"])
    items = list(zip(all_docs["ids"], all_docs["documents"], all_docs["metadatas"]))
    items.sort(key=lambda x: x[2].get("timestamp", ""), reverse=True)

    return [
        {
            "experiment_id": item_id,
            "model_name": meta.get("model_name"),
            "accuracy": meta.get("accuracy"),
            "dataset_name": meta.get("dataset_name"),
            "goal": meta.get("goal"),
            "top_features": json.loads(meta.get("top_features", "[]")),
            "timestamp": meta.get("timestamp"),
        }
        for item_id, _, meta in items[:n]
    ]


if __name__ == "__main__":
    print("=" * 60)
    print("Memory Agent — save & retrieve test")
    print("=" * 60)

    fake_state = {
        "csv_path": "backend/data/titanic.csv",
        "goal": "predict which passengers survived the Titanic disaster",
        "target_column": "Survived",
        "ml_result": {
            "best_model_name": "Logistic Regression",
            "best_accuracy": 0.7318,
        },
        "optimized_result": {
            "best_accuracy": 0.7430,
            "best_params": {"C": 0.255, "max_iter": 216, "solver": "liblinear"},
            "improvement": 0.0112,
        },
        "critic_result": {
            "top_features": [
                ["Pclass", 0.4538],
                ["Fare", 0.2583],
                ["Age", 0.2012],
                ["SibSp", 0.1277],
                ["Parch", 0.1213],
            ],
        },
    }

    doc_id = save_experiment(fake_state)
    print(f"\nSaved with id: {doc_id}")

    print("\n--- Retrieving similar experiments ---")
    context = get_similar_experiments("predict passenger survival on a ship", n=3)
    print(context if context else "(no results)")
