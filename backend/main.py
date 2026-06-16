import os
import sys
import uuid
import threading
import traceback
from pathlib import Path

# ── Path setup so agents/ and graph/ are importable ──────────────────────────
_BACKEND = Path(__file__).parent
sys.path.insert(0, str(_BACKEND / "agents"))
sys.path.insert(0, str(_BACKEND / "graph"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import AgentState, build_pipeline
from memory_agent import get_recent_experiments

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="AgentMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_origin_regex=r"https://.*\.(vercel\.app|onrender\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = _BACKEND / "data"
DATA_DIR.mkdir(exist_ok=True)

# job_id -> job dict
jobs: dict = {}


# ── Background runner ─────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, csv_path: str, goal: str, target_column: str) -> None:
    def progress_cb(node_name: str, pct: int) -> None:
        if jobs.get(job_id, {}).get("status") == "running":
            jobs[job_id]["current_agent"] = node_name
            jobs[job_id]["progress"] = pct

    try:
        pipeline = build_pipeline(progress_cb=progress_cb)

        initial_state: AgentState = {
            "csv_path": csv_path,
            "goal": goal,
            "target_column": target_column,
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

        final_state = pipeline.invoke(initial_state)

        ml = final_state["ml_result"]
        opt = final_state["optimized_result"]

        jobs[job_id] = {
            "status": "done",
            "result": {
                "best_model": ml.get("best_model_name"),
                "accuracy": opt.get("best_accuracy") or ml.get("best_accuracy"),
                "improvement": opt.get("improvement", 0.0),
                "top_features": final_state["top_features"],
                "critique": final_state["critique"],
                "warnings": final_state["warnings"],
                "experiment_id": final_state["experiment_id"],
                "all_results": ml.get("all_results", []),
            },
        }

    except Exception:
        jobs[job_id] = {
            "status": "error",
            "message": traceback.format_exc(),
        }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/run")
async def run_pipeline(
    csv_file: UploadFile = File(...),
    goal: str = Form(...),
    target_column: str = Form(...),
):
    job_id = str(uuid.uuid4())

    csv_path = str(DATA_DIR / f"{job_id}.csv")
    with open(csv_path, "wb") as f:
        f.write(await csv_file.read())

    jobs[job_id] = {
        "status": "running",
        "current_agent": "eda_node",
        "progress": 0,
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, csv_path, goal, target_column),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/history")
async def get_history():
    return get_recent_experiments(n=5)


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
