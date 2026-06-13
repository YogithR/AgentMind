import os
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, END

# Make sure backend/agents is importable regardless of cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents')))

from eda_agent import run_eda_agent
from ml_agent import run_ml_agent
from optimizer_agent import run_optimizer_agent
from critic_agent import run_critic_agent
from memory_agent import save_experiment, get_similar_experiments


class AgentState(TypedDict):
    csv_path: str
    goal: str
    target_column: str
    eda_result: str
    ml_result: dict
    optimized_result: dict
    critic_result: dict
    critique: str
    top_features: list
    warnings: list
    experiment_id: str
    status: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

def eda_node(state: AgentState) -> dict:
    print("\n[EDA Node] Running EDA agent...")
    past = get_similar_experiments(state["goal"], n=3)
    if past:
        print("[EDA Node] Injecting context from memory...")
    result = run_eda_agent(state["csv_path"], state["goal"], past_experiments=past)
    return {"eda_result": result, "status": "eda_done"}


def ml_node(state: AgentState) -> dict:
    print("\n[ML Node] Running ML training agent...")
    result = run_ml_agent(state["csv_path"], state["target_column"], state["eda_result"])
    return {"ml_result": result, "status": "ml_done"}


def optimizer_node(state: AgentState) -> dict:
    print("\n[Optimizer Node] Running hyperparameter optimizer...")
    best_model = state["ml_result"]["best_model_name"]
    result = run_optimizer_agent(state["csv_path"], state["target_column"], best_model)
    return {"optimized_result": result, "status": "optimizer_done"}


def critic_node(state: AgentState) -> dict:
    print("\n[Critic Node] Running critic agent...")
    result = run_critic_agent(
        csv_path=state["csv_path"],
        target_column=state["target_column"],
        eda_result=state["eda_result"],
        ml_result=state["ml_result"],
        optimized_result=state["optimized_result"],
    )
    return {
        "critic_result": result,
        "critique": result["critique"],
        "top_features": result["top_features"],
        "warnings": result["warnings"],
        "status": "critic_done",
    }


def memory_node(state: AgentState) -> dict:
    print("\n[Memory Node] Saving experiment to ChromaDB...")
    doc_id = save_experiment(state)
    return {"experiment_id": doc_id, "status": "complete"}


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_pipeline(progress_cb=None):
    def _wrap(fn, node_name, pct):
        def wrapped(state):
            if progress_cb:
                progress_cb(node_name, pct)
            return fn(state)
        return wrapped

    graph = StateGraph(AgentState)

    graph.add_node("eda_node",       _wrap(eda_node,       "eda_node",       20))
    graph.add_node("ml_node",        _wrap(ml_node,        "ml_node",        40))
    graph.add_node("optimizer_node", _wrap(optimizer_node, "optimizer_node", 60))
    graph.add_node("critic_node",    _wrap(critic_node,    "critic_node",    80))
    graph.add_node("memory_node",    _wrap(memory_node,    "memory_node",    95))

    graph.set_entry_point("eda_node")
    graph.add_edge("eda_node",       "ml_node")
    graph.add_edge("ml_node",        "optimizer_node")
    graph.add_edge("optimizer_node", "critic_node")
    graph.add_edge("critic_node",    "memory_node")
    graph.add_edge("memory_node",    END)

    return graph.compile()


# ── Test ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_csv = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'titanic.csv')
    )

    initial_state: AgentState = {
        "csv_path": sample_csv,
        "goal": "predict which passengers survived the Titanic disaster",
        "target_column": "Survived",
        "eda_result": "",
        "ml_result": {},
        "optimized_result": {},
        "critic_result": {},
        "critique": "",
        "top_features": [],
        "warnings": [],
        "experiment_id": "",
        "status": "start",
    }

    pipeline = build_pipeline()
    final_state = pipeline.invoke(initial_state)

    # ── Clean summary ──────────────────────────────────────────────
    ml = final_state["ml_result"]
    opt = final_state["optimized_result"]
    top3 = final_state["top_features"][:3]

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)

    print(f"\nBest model  : {ml.get('best_model_name')}")
    print(f"Accuracy    : {opt.get('best_accuracy', ml.get('best_accuracy')):.4f}  "
          f"(+{opt.get('improvement', 0):.4f} from tuning)")

    print("\nTop 3 features:")
    for name, shap_val in top3:
        print(f"  {name:20s}  SHAP={shap_val}")

    if final_state["warnings"]:
        print("\nWarnings:")
        for w in final_state["warnings"]:
            print(f"  ! {w}")

    print(f"\nExperiment saved  : {final_state['experiment_id']}")

    print("\n" + "-" * 60)
    print("GROQ CRITIQUE")
    print("-" * 60)
    print(final_state["critique"])
