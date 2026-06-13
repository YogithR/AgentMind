import { useEffect, useState } from "react";
import axios from "axios";

const API = "http://localhost:8080";

const STEPS = [
  { key: "eda_node",       label: "EDA Analysis",     icon: "🔍", pct: 20 },
  { key: "ml_node",        label: "ML Training",       icon: "🤖", pct: 40 },
  { key: "optimizer_node", label: "Hyperparameter Tuning", icon: "⚙️", pct: 60 },
  { key: "critic_node",    label: "Critic Review",     icon: "🧠", pct: 80 },
  { key: "memory_node",    label: "Memory Storage",    icon: "💾", pct: 95 },
];

const styles = {
  page: {
    minHeight: "100vh",
    background: "#F5F7FA",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "Arial, sans-serif",
  },
  card: {
    background: "#fff",
    borderRadius: 12,
    boxShadow: "0 4px 24px rgba(0,0,0,0.10)",
    padding: "48px 40px",
    width: 520,
    maxWidth: "95vw",
  },
  heading: { color: "#1F4E79", fontSize: 24, fontWeight: 700, marginBottom: 6 },
  sub: { color: "#6B7280", fontSize: 14, marginBottom: 36 },
  step: (state) => ({
    display: "flex",
    alignItems: "center",
    gap: 14,
    padding: "14px 18px",
    borderRadius: 10,
    marginBottom: 12,
    background: state === "done" ? "#F0FDF4" : state === "running" ? "#EFF6FF" : "#F8FAFC",
    border: `1.5px solid ${state === "done" ? "#86EFAC" : state === "running" ? "#93C5FD" : "#E2E8F0"}`,
    transition: "all 0.3s",
  }),
  icon: { fontSize: 22, width: 28, textAlign: "center" },
  stepLabel: (state) => ({
    flex: 1,
    fontSize: 15,
    fontWeight: state === "running" ? 700 : 500,
    color: state === "done" ? "#15803D" : state === "running" ? "#1D4ED8" : "#9CA3AF",
  }),
  badge: (state) => ({
    fontSize: 12,
    fontWeight: 600,
    padding: "3px 10px",
    borderRadius: 20,
    background: state === "done" ? "#DCFCE7" : state === "running" ? "#DBEAFE" : "#F1F5F9",
    color: state === "done" ? "#15803D" : state === "running" ? "#1D4ED8" : "#94A3B8",
  }),
  progressBar: {
    height: 6,
    background: "#E2E8F0",
    borderRadius: 99,
    margin: "28px 0 8px",
    overflow: "hidden",
  },
  progressFill: (pct) => ({
    height: "100%",
    width: `${pct}%`,
    background: "linear-gradient(90deg, #1F4E79, #3B82F6)",
    borderRadius: 99,
    transition: "width 0.5s ease",
  }),
  pctLabel: { color: "#6B7280", fontSize: 13, textAlign: "right", marginBottom: 4 },
  error: {
    color: "#DC2626",
    background: "#FEF2F2",
    borderRadius: 8,
    padding: "12px 16px",
    fontSize: 13,
    marginTop: 20,
  },
  spinner: {
    display: "inline-block",
    width: 14,
    height: 14,
    border: "2px solid #93C5FD",
    borderTop: "2px solid #1D4ED8",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
    marginLeft: 8,
    verticalAlign: "middle",
  },
};

function getStepState(step, currentAgent, progress, overallStatus) {
  if (overallStatus === "done") return "done";
  const stepIdx = STEPS.findIndex((s) => s.key === step.key);
  const currentIdx = STEPS.findIndex((s) => s.key === currentAgent);
  if (stepIdx < currentIdx) return "done";
  if (step.key === currentAgent) return "running";
  return "pending";
}

export default function ProgressPage({ jobId, onDone }) {
  const [job, setJob] = useState({ status: "running", current_agent: "eda_node", progress: 0 });
  const [error, setError] = useState("");

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const { data } = await axios.get(`${API}/status/${jobId}`);
        setJob(data);
        if (data.status === "done") {
          clearInterval(interval);
          setTimeout(() => onDone(data.result), 600);
        }
        if (data.status === "error") {
          clearInterval(interval);
          setError(data.message || "An error occurred.");
        }
      } catch {
        setError("Lost connection to server.");
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [jobId, onDone]);

  const progress = job.progress || 0;

  return (
    <div style={styles.page}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      <div style={styles.card}>
        <div style={styles.heading}>Running Pipeline</div>
        <div style={styles.sub}>Job ID: {jobId}</div>

        {STEPS.map((step) => {
          const state = getStepState(step, job.current_agent, progress, job.status);
          return (
            <div key={step.key} style={styles.step(state)}>
              <div style={styles.icon}>{step.icon}</div>
              <div style={styles.stepLabel(state)}>{step.label}</div>
              <div style={styles.badge(state)}>
                {state === "done" && "Done"}
                {state === "running" && (<>Running<span style={styles.spinner} /></>)}
                {state === "pending" && "Waiting"}
              </div>
            </div>
          );
        })}

        <div style={styles.pctLabel}>{progress}%</div>
        <div style={styles.progressBar}>
          <div style={styles.progressFill(progress)} />
        </div>

        {error && <div style={styles.error}><strong>Error:</strong> {error}</div>}
      </div>
    </div>
  );
}
