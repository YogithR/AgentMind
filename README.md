<div align="center">

# 🧠 AgentMind
### Autonomous Multi-Agent Machine Learning System

*Upload any dataset. Describe your goal. 5 AI agents do the rest.*

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-agent--mind--ucf3.vercel.app-blue?style=for-the-badge)](https://agent-mind-ucf3.vercel.app)
[![API Docs](https://img.shields.io/badge/🔗_API_Docs-agentmind--api.onrender.com-green?style=for-the-badge)](https://agentmind-api.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/📁_GitHub-YogithR/AgentMind-black?style=for-the-badge)](https://github.com/YogithR/AgentMind)

</div>

---

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

## Features

- Upload any CSV and describe your goal in plain English — no code required
- 5 collaborating AI agents handle EDA, training, tuning, SHAP explainability, and critique automatically
- 5 ML models trained in parallel: Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression
- Handles large datasets up to 145,000+ rows with automatic memory optimization and chunked processing
- Bayesian hyperparameter tuning with Optuna (30 trials, TPE sampler)
- SHAP-based feature explainability and a plain-English business critique via Groq LLM
- Learns from past experiments via a ChromaDB vector memory store

---

## How It Works

AgentMind chains five specialised agents through a LangGraph state graph. Each agent fires in sequence and passes its output to the next:

| # | Agent | Role |
|---|-------|------|
| 🧠 | **Orchestrator Agent** | Plans the run, injects past experiment context from ChromaDB memory |
| 🔍 | **EDA Agent** | Profiles the dataset and generates a natural-language summary using Groq LLM |
| ⚙️ | **ML Training Agent** | Preprocesses data and trains 5 models in parallel — Random Forest, XGBoost, LightGBM, CatBoost, and Logistic Regression |
| 🔁 | **Optimizer Agent** | Runs 30 Bayesian hyperparameter tuning trials via Optuna on the best model |
| 🧾 | **Critic Agent** | Computes SHAP feature importances and sends a business-quality critique to Groq |

The preprocessing pipeline (applied before any training) automatically drops high-missing columns, fills numeric gaps with medians, fills categorical gaps with mode, one-hot encodes categoricals, and removes ID-like columns — all fitted on the training split to prevent leakage.

---

## How Each Agent Works

### 🧠 Orchestrator Agent

The Orchestrator is the master controller of the entire pipeline. When a user uploads a dataset and describes their goal, the Orchestrator reads both and creates an execution plan. It uses Groq LLM with chain-of-thought reasoning to decide which agent runs next based on what has already been completed. It maintains a shared AgentState object that carries all data, results, and context across every agent in the pipeline — so each agent knows exactly what the previous one found.

**What it does technically:**
- Initializes the AgentState with `csv_path`, `goal`, and `target_column`
- Manages the LangGraph StateGraph execution order
- Passes updated state between agents after each node completes
- Handles errors and decides whether to retry or skip a step

---

### 🔍 EDA Agent (Exploratory Data Analysis)

The EDA Agent is the first agent to touch the data. It reads the uploaded CSV using pandas and generates a full statistical profile: number of rows and columns, data types, missing value counts, descriptive statistics, and feature correlations. It then sends this entire profile to the Groq LLM with a structured prompt asking it to identify the problem type (classification or regression), flag data quality issues, and recommend the most suitable ML approach.

Before analyzing, the EDA Agent queries ChromaDB to retrieve any past experiments run on similar datasets. These past results are injected into the Groq prompt as context — so the agent genuinely learns from history and gives smarter recommendations over time.

**What it does technically:**
- Loads CSV with pandas, runs `df.describe()`, `df.dtypes`, `df.isnull()`
- Calculates top feature correlations with the target column
- Queries ChromaDB vector memory for similar past experiments
- Sends full summary + memory context to Groq LLM
- Returns structured EDA result string into AgentState

---

### ⚙️ ML Training Agent

The ML Training Agent reads the EDA result and prepares the data for model training. It first runs an automatic preprocessing pipeline: dropping columns with more than 50% missing values, filling remaining missing numerics with the median, filling missing categoricals with the mode, and one-hot encoding all categorical columns consistently across train and test sets.

It then trains five models — Random Forest, XGBoost, LightGBM, CatBoost, and Logistic Regression — using an 80/20 train-test split. Parallel model training using Python ThreadPoolExecutor across all available CPU cores — all 5 models train simultaneously. Each model is evaluated on accuracy and F1-score. The best performing model is selected and its name, score, and trained object are passed to the next agent.

**What it does technically:**
- Automatic preprocessing: median/mode imputation, one-hot encoding
- Trains `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`, `LogisticRegression`
- Trains all 5 models concurrently with `ThreadPoolExecutor`, one thread per CPU core
- Evaluates all five with `accuracy_score` and `f1_score`
- Selects best model and passes it forward in AgentState
- Returns `all_results` list so the UI can show all five scores

---

### 🔁 Optimizer Agent (Reinforcement-Style Tuning)

The Optimizer Agent takes the best model from the ML Training Agent and improves it using Bayesian hyperparameter optimization powered by Optuna. This process is modeled as a reinforcement learning problem: each trial is an action, the hyperparameter configuration is the policy, and the improvement in validation accuracy is the reward signal. The agent runs 30 trials, and Optuna's Tree-structured Parzen Estimator (TPE) learns which configurations are likely to score higher, focusing its search on the most promising regions.

Each model type has its own tunable parameter space:
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`
- **XGBoost:** `n_estimators`, `max_depth`, `learning_rate`
- **Logistic Regression:** `C`, `max_iter`, `solver`

After 30 trials the agent returns the best parameters and the accuracy improvement achieved over the baseline.

**What it does technically:**
- Uses `optuna.create_study(direction="maximize")`
- Objective function returns validation accuracy as reward
- TPE sampler focuses search on high-reward parameter regions
- Runs 30 trials, returns `best_params` and `best_accuracy`
- Calculates and stores improvement delta in AgentState

---

### 🧾 Critic Agent

The Critic Agent performs two jobs: technical evaluation and business translation. On the technical side, it retrains the best model with optimized parameters and runs SHAP (SHapley Additive Explanations) to calculate how much each feature contributed to the model's predictions. SHAP values are model-agnostic and mathematically grounded — they show exactly which features push predictions higher or lower and by how much.

On the business side, it sends the model name, accuracy, top SHAP features, and any data quality warnings to Groq LLM with a prompt asking it to explain everything in plain English for a non-technical business user, identify any concerns like overfitting or bias, and give 3 actionable recommendations.

**What it does technically:**
- Retrains best model with optimized hyperparameters from Optimizer
- Runs `shap.TreeExplainer` on the trained model
- Calculates mean absolute SHAP values per feature
- Ranks top 5 features by importance
- Detects warnings: high missing data, class imbalance, low variance
- Sends full context to Groq for business critique generation

---

### 💾 Memory Agent

The Memory Agent is what makes AgentMind get smarter over time. After every successful pipeline run, it serializes the full experiment — model name, accuracy, top features, goal text, dataset name, and timestamp — and saves it to ChromaDB as a vector document. The goal text is embedded as a vector so future runs can search for semantically similar past experiments.

At the start of every new run, the EDA Agent queries this memory store and retrieves the 2–3 most similar past experiments. These are injected into the EDA prompt as context, so the agent avoids strategies that previously failed and builds on ones that worked. This is the long-term memory layer that separates AgentMind from every existing AutoML tool.

**What it does technically:**
- Uses `chromadb.PersistentClient` for memory that survives restarts
- Embeds experiment metadata as vector documents
- `save_experiment()` serializes full AgentState result to ChromaDB
- `get_similar_experiments()` does semantic search by goal similarity
- Returns `experiment_id` (UUID) that is stored in final AgentState

---

## How AgentMind is Different from Existing ML Tools

| Feature | Google AutoML | DataRobot | H2O.ai | AgentMind |
|---|---|---|---|---|
| Natural language goal input | ❌ | ❌ | ❌ | ✅ Plain English |
| LLM-based reasoning | ❌ | ❌ | ❌ | ✅ Groq LLM agents |
| Explains decisions in plain English | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited | ✅ Groq critique |
| Learns from past experiments | ❌ | ❌ | ❌ | ✅ ChromaDB memory |
| SHAP feature explainability | ⚠️ Basic | ✅ Yes | ✅ Yes | ✅ Yes |
| RL-style self-improvement loop | ❌ | ❌ | ⚠️ Partial | ✅ Optuna TPE |
| Open source and free | ❌ | ❌ | ⚠️ Partial | ✅ Fully open |
| Approximate monthly cost | $300–$500 | $2,500+ | $200+ | $0 |
| Multi-agent architecture | ❌ | ❌ | ❌ | ✅ 5-agent system |
| Memory injection into prompts | ❌ | ❌ | ❌ | ✅ RAG-powered |

### The core difference

Every existing AutoML tool executes a fixed, predefined pipeline. They are fast but they do not think. AgentMind is built differently — each agent reasons about what it finds, adapts its approach based on the data, critiques its own output, and remembers what worked before. This is the difference between automation and intelligence.

Traditional AutoML tools ask: *"what is the best model for this data?"*
AgentMind asks: *"what is the best way to understand this problem, train a model for it, improve it, explain it to a human, and remember this for next time?"*

That reasoning layer — powered by Groq LLM, LangGraph orchestration, and ChromaDB memory — is what no existing commercial tool has.

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

Benchmarked on four public datasets using the full 5-agent pipeline:

| Dataset | Rows | Best Model | Accuracy | Top Feature |
|---|---|---|---|---|
| Titanic Survival | 891 | Random Forest | 76.0% | Pclass |
| Heart Disease | 303 | Logistic Regression | 88.5% | cp |
| Telco Customer Churn | 7,043 | Logistic Regression | 80.8% | tenure |
| Rain in Australia | 145,460 | XGBoost | 85.3% | Humidity3pm |

## Real World Performance

AgentMind was tested on 4 real-world datasets ranging from 303 to 145,460 rows across healthcare, business, and environmental domains. The system automatically selected the best algorithm for each dataset without any manual configuration — demonstrating that the multi-agent architecture generalizes across industries and dataset sizes. On the Rain in Australia dataset (145,460 rows), AgentMind achieved 85.3% accuracy using XGBoost and correctly identified Humidity3pm as the strongest predictor of next-day rainfall — a result consistent with real meteorological research.

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
| LightGBM | |
| CatBoost | |
| psutil (memory monitoring) | |

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
