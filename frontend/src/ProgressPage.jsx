import { useEffect, useRef, useState } from "react";
import axios from "axios";

const API = "http://localhost:8080";

const STEPS = [
  { key: "eda_node",       num: 1, label: "EDA Agent",     subtitle: "Analyzing your dataset" },
  { key: "ml_node",        num: 2, label: "ML Training",   subtitle: "Training 3 machine learning models" },
  { key: "optimizer_node", num: 3, label: "Optimizer",     subtitle: "Running 30 Bayesian tuning trials" },
  { key: "critic_node",    num: 4, label: "Critic Agent",  subtitle: "Evaluating model quality with SHAP" },
  { key: "memory_node",    num: 5, label: "Memory Agent",  subtitle: "Saving experiment to memory" },
];

function getStepState(stepKey, currentAgent, overallStatus) {
  if (overallStatus === "done") return "completed";
  const stepIdx = STEPS.findIndex((s) => s.key === stepKey);
  const currentIdx = STEPS.findIndex((s) => s.key === currentAgent);
  if (currentIdx < 0) return "waiting";
  if (stepIdx < currentIdx) return "completed";
  if (stepIdx === currentIdx) return "running";
  return "waiting";
}

function StepCircle({ state, num }) {
  if (state === "waiting") {
    return (
      <div style={{
        width: 44, height: 44, borderRadius: "50%",
        background: "#E9EEF4",
        display: "flex", alignItems: "center", justifyContent: "center",
        color: "#A0AABB", fontWeight: 700, fontSize: 16, flexShrink: 0,
      }}>
        {num}
      </div>
    );
  }
  if (state === "running") {
    return (
      <div style={{
        width: 44, height: 44, borderRadius: "50%",
        border: "3.5px solid #BFDBFE",
        borderTop: "3.5px solid #2563EB",
        animation: "spin 0.85s linear infinite, glow 1.6s ease-in-out infinite",
        boxSizing: "border-box",
        flexShrink: 0,
      }} />
    );
  }
  // completed
  return (
    <div style={{
      width: 44, height: 44, borderRadius: "50%",
      background: "linear-gradient(135deg, #22C55E, #16A34A)",
      display: "flex", alignItems: "center", justifyContent: "center",
      color: "#fff", fontSize: 20, fontWeight: 800, flexShrink: 0,
      boxShadow: "0 2px 8px rgba(34,197,94,0.35)",
    }}>
      ✓
    </div>
  );
}

function StepRow({ step, state, elapsed, duration }) {
  const isRunning = state === "running";
  const isDone = state === "completed";

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 16,
      padding: "16px 20px",
      borderRadius: 14,
      marginBottom: 12,
      background: isDone ? "#F0FDF4" : isRunning ? "#EFF6FF" : "#FAFBFC",
      border: `2px solid ${isDone ? "#86EFAC" : isRunning ? "#3B82F6" : "#E8ECF2"}`,
      boxShadow: isRunning ? "0 0 0 4px rgba(59,130,246,0.08)" : "none",
      transition: "background 0.35s, border-color 0.35s, box-shadow 0.35s",
    }}>
      <StepCircle state={state} num={step.num} />

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: 15,
          fontWeight: isRunning ? 800 : isDone ? 700 : 500,
          color: isDone ? "#15803D" : isRunning ? "#1D4ED8" : "#A0AABB",
          marginBottom: 3,
          letterSpacing: isRunning ? "0.01em" : 0,
        }}>
          {step.label}
        </div>
        <div style={{
          fontSize: 13,
          color: isDone ? "#4ADE80" : isRunning ? "#60A5FA" : "#C8D0DC",
        }}>
          {step.subtitle}
        </div>
      </div>

      <div style={{ textAlign: "right", flexShrink: 0, minWidth: 110 }}>
        {isRunning && (
          <div style={{
            fontSize: 22,
            fontWeight: 800,
            color: "#2563EB",
            fontVariantNumeric: "tabular-nums",
            letterSpacing: "-0.5px",
          }}>
            {elapsed.toFixed(1)}s
          </div>
        )}
        {isDone && duration != null && (
          <div style={{
            fontSize: 12,
            color: "#16A34A",
            fontWeight: 600,
            background: "#DCFCE7",
            padding: "4px 10px",
            borderRadius: 20,
            display: "inline-block",
          }}>
            completed in {duration.toFixed(1)}s
          </div>
        )}
      </div>
    </div>
  );
}

export default function ProgressPage({ jobId, onDone }) {
  const [job, setJob] = useState({ status: "running", current_agent: "eda_node", progress: 0 });
  const [error, setError] = useState("");
  // Increments every second to force elapsed-timer re-renders
  const [, forceRender] = useState(0);

  const timingRef = useRef({ starts: {}, durations: {} });
  const prevAgentRef = useRef(null);

  useEffect(() => {
    const ticker = setInterval(() => forceRender((n) => n + 1), 1000);
    return () => clearInterval(ticker);
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const { data } = await axios.get(`${API}/status/${jobId}`);
        const now = Date.now();
        const currentAgent = data.current_agent;
        const prevAgent = prevAgentRef.current;

        if (currentAgent && currentAgent !== prevAgent) {
          if (!timingRef.current.starts[currentAgent]) {
            timingRef.current.starts[currentAgent] = now;
          }
          if (prevAgent && timingRef.current.starts[prevAgent] && !timingRef.current.durations[prevAgent]) {
            timingRef.current.durations[prevAgent] =
              (now - timingRef.current.starts[prevAgent]) / 1000;
          }
          prevAgentRef.current = currentAgent;
        }

        setJob(data);

        if (data.status === "done") {
          clearInterval(interval);
          const last = data.current_agent || prevAgentRef.current;
          if (last && timingRef.current.starts[last] && !timingRef.current.durations[last]) {
            timingRef.current.durations[last] = (now - timingRef.current.starts[last]) / 1000;
          }
          setTimeout(() => onDone(data.result), 1000);
        }
        if (data.status === "error") {
          clearInterval(interval);
          setError(data.message || "An error occurred.");
        }
      } catch {
        setError("Lost connection to server.");
        clearInterval(interval);
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId, onDone]);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #F0F4FA 0%, #E8EEF7 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'Inter', 'Segoe UI', Arial, sans-serif",
    }}>
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 0 0 rgba(37,99,235,0.25); }
          50%       { box-shadow: 0 0 0 8px rgba(37,99,235,0); }
        }
      `}</style>

      <div style={{
        background: "#fff",
        borderRadius: 20,
        boxShadow: "0 8px 40px rgba(0,0,0,0.10)",
        padding: "52px 44px",
        width: 580,
        maxWidth: "95vw",
      }}>
        <div style={{ color: "#1F4E79", fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
          Running Pipeline
        </div>
        <div style={{ color: "#A0AABB", fontSize: 13, marginBottom: 40 }}>
          Job ID: <span style={{ fontFamily: "monospace", color: "#64748B" }}>{jobId}</span>
        </div>

        {STEPS.map((step) => {
          const state = getStepState(step.key, job.current_agent, job.status);
          const startTime = timingRef.current.starts[step.key];
          const elapsed = state === "running" && startTime
            ? (Date.now() - startTime) / 1000
            : 0;
          const duration = timingRef.current.durations[step.key] ?? null;

          return (
            <StepRow
              key={step.key}
              step={step}
              state={state}
              elapsed={elapsed}
              duration={duration}
            />
          );
        })}

        {error && (
          <div style={{
            color: "#DC2626",
            background: "#FEF2F2",
            border: "1.5px solid #FCA5A5",
            borderRadius: 10,
            padding: "14px 18px",
            fontSize: 13,
            marginTop: 20,
          }}>
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>
    </div>
  );
}
