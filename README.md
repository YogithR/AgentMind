# AgentMind

**Autonomous Multi-Agent ML System — upload any dataset, get a trained and explained ML model in minutes**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.124-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-FF6B35?style=for-the-badge&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

---

## Demo

📹 **Demo Video:** [Watch on YouTube](link-here)
🚀 **Live Demo:** Coming soon

---

## What is AgentMind?

AgentMind is a fully autonomous machine learning pipeline powered by five collaborating AI agents — you upload a CSV, name your target column, and the system handles everything from exploratory analysis to model training, hyperparameter tuning, SHAP explainability, and a plain-English business critique, all without writing a single line of code. It is built for data scientists who want to accelerate experimentation, ML engineers evaluating model quality across datasets, and product teams who need fast, interpretable predictions without deep ML expertise. Each pipeline run is saved to a ChromaDB vector store so the system can learn from past experiments and inject relevant context into future analyses.

---

## How It Works

AgentMind chains five specialised agents through a LangGraph state graph. Each agent fires in sequence and passes its output to the next:

| # | Agent | Role |
|---|-------|------|
| 🧠 | **Orchestrator Agent** | Plans the run, injects past experiment context from ChromaDB memory |
| 🔍 | **EDA Agent** | Profiles the dataset and generates a natural-language summary using Groq LLM |
| ⚙️ | **ML Training Agent** | Preprocesses data and trains Random Forest, XGBoost, and Logistic Regression |
| 🔁 | **Optimizer Agent** | Runs 30 Bayesian hyperparameter tuning trials via Optuna on the best model |
| 🧾 | **Critic Agent** | Computes SHAP feature importances and sends a business-quality critique to Groq |

The preprocessing pipeline (applied before any training) automatically drops high-missing columns, fills numeric gaps with medians, fills categorical gaps with mode, one-hot encodes categoricals, and removes ID-like columns — all fitted on the training split to prevent leakage.

---

## Architecture

```mermaid
graph LR
    U(["👤 User"])

    subgraph FE ["⚛️ React Frontend"]
        F1["Upload Page"]
        F2["Progress Page"]
        F3["Results Page"]
    end

    subgraph BE ["🚀 FastAPI Backend"]
        B1["/run · /status"]
    end

    subgraph PL ["🔗 LangGraph Pipeline"]
        P1["🧠 Orchestrator"]
        P2["🔍 EDA Agent"]
        P3["⚙️ ML Agent"]
        P4["🔁 Optimizer"]
        P5["🧾 Critic Agent"]
        P6["💾 Memory Agent"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    G[("☁️ Groq LLM")]
    C[("🗄️ ChromaDB")]
    S["📦 scikit-learn + XGBoost"]

    U -->|"upload CSV + goal"| FE
    FE -->|"POST /run"| BE
    FE -.->|"poll /status"| BE
    BE -->|"invoke"| PL

    P2 -->|"LLM call"| G
    P5 -->|"LLM call"| G
    P3 -->|"train"| S
    P6 -->|"store"| C
    C -.->|"retrieve context"| P1

    classDef fe     fill:#E0F2FE,stroke:#0EA5E9,color:#0369A1
    classDef be     fill:#ECFDF5,stroke:#10B981,color:#065F46
    classDef agent  fill:#1E3A5F,stroke:#3B82F6,color:#FFFFFF
    classDef ext    fill:#FFF7ED,stroke:#F59E0B,color:#92400E
    classDef db     fill:#F5F3FF,stroke:#8B5CF6,color:#5B21B6

    class F1,F2,F3 fe
    class B1 be
    class P1,P2,P3,P4,P5,P6 agent
    class S ext
    class G,C db
```

---

## Results on Real Datasets

Benchmarked on three public datasets using the full 5-agent pipeline:

| Dataset | Rows | Best Model | Accuracy | Top Feature (SHAP) |
|---------|------|-----------|----------|--------------------|
| Titanic survival | 891 | Random Forest | **82.1%** | `Sex_male` |
| Telco Customer Churn | 7,043 | Logistic Regression | **80.8%** | `tenure` |
| Heart Disease (UCI) | 303 | Logistic Regression | **88.5%** | `cp` (chest pain type) |

---

## Tech Stack

| Backend | Frontend |
|---------|----------|
| Python 3.11 | React 18 |
| FastAPI 0.124 | Vite |
| LangGraph 1.2 | Recharts |
| scikit-learn | Axios |
| XGBoost 3.2 | React Dropzone |
| Optuna 4.9 (Bayesian tuning) | |
| SHAP 0.52 | |
| Groq SDK 1.4 (LLM critique) | |
| ChromaDB 1.5 (vector memory) | |

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YogithR/agentmind.git
cd agentmind
```

### 2. Set up the Python environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install fastapi uvicorn langgraph scikit-learn xgboost optuna shap \
            chromadb groq pandas numpy python-dotenv python-multipart
```

### 3. Add your API keys

Create `backend/.env`:

```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### 4. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8080
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173), upload a CSV, set your target column and goal, and hit **Run Pipeline**.

---

## Project Structure

```
agentmind/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py     # coordinates the pipeline run
│   │   ├── eda_agent.py        # dataset profiling + Groq summary
│   │   ├── ml_agent.py         # preprocessing + model training
│   │   ├── optimizer_agent.py  # Optuna Bayesian tuning
│   │   ├── critic_agent.py     # SHAP importances + Groq critique
│   │   └── memory_agent.py     # ChromaDB read/write
│   ├── graph/
│   │   └── pipeline.py         # LangGraph state graph
│   ├── data/                   # uploaded CSVs (auto-created)
│   ├── memory/                 # ChromaDB vector store (auto-created)
│   ├── main.py                 # FastAPI app + /run and /status endpoints
│   └── .env                    # API keys (not committed)
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # page router
│   │   ├── UploadPage.jsx      # CSV upload + goal form
│   │   ├── ProgressPage.jsx    # live 5-step agent progress view
│   │   └── ResultsPage.jsx     # model results + SHAP chart + download
│   ├── index.html
│   └── vite.config.js
├── README.md
└── .gitignore
```

---

## Roadmap

| Version | Theme | Planned Features |
|---------|-------|-----------------|
| **v1.5** | Better models | CatBoost support, automatic feature selection, cross-validation scores, regression task support |
| **v2.0** | Scale & auth | Multi-CSV joins, time-series datasets, API key authentication, rate limiting |
| **v2.5** | Cloud | One-click deploy to Render / Railway, user accounts, persistent experiment history dashboard |
| **v3.0** | Natural language | Chat interface to refine models, NL-driven pipeline config, automated report generation as PDF |

---

## Author

Built by **YogithR**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile-here)

---

*If you find this useful, please consider starring the repository.*
