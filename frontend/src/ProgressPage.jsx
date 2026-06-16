import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { COLORS, cardStyle } from "./theme";

const API = "http://localhost:8080";

const AGENTS = [
  { key: "eda_node", icon: "🔍", label: "EDA Agent", description: "Analyzing your dataset" },
  { key: "ml_node", icon: "🧠", label: "ML Training", description: "Training 3 machine learning models" },
  { key: "optimizer_node", icon: "⚙️", label: "Optimizer", description: "Running 30 Bayesian tuning trials" },
  { key: "critic_node", icon: "🔁", label: "Critic Agent", description: "Evaluating model quality with SHAP" },
  { key: "memory_node", icon: "🧾", label: "Memory Agent", description: "Saving experiment to memory" },
];

function getAgentState(agentKey, currentAgent, overallStatus) {
  if (overallStatus === "done") return "completed";
  const idx = AGENTS.findIndex((a) => a.key === agentKey);
  const currentIdx = AGENTS.findIndex((a) => a.key === currentAgent);
  if (currentIdx < 0) return "queued";
  if (idx < currentIdx) return "completed";
  if (idx === currentIdx) return "running";
  return "queued";
}

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function StatusBadge({ state }) {
  const config = {
    completed: { label: "Completed", color: COLORS.green, bg: "rgba(34,197,94,0.12)" },
    running: { label: "In Progress", color: COLORS.accent, bg: "var(--accent-tint)" },
    queued: { label: "Queued", color: COLORS.textSecondary, bg: "rgba(136,136,160,0.1)" },
  }[state];

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        color: config.color,
        background: config.bg,
        borderRadius: 20,
        padding: "4px 12px",
        fontSize: 12,
        fontWeight: 600,
        animation: state === "running" ? "badge-pulse 1.6s ease-in-out infinite" : "none",
      }}
    >
      {config.label}
    </span>
  );
}

function OrbitalAnimation() {
  return (
    <div style={{ position: "relative", width: 200, height: 200, margin: "0 auto" }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: "50%",
          border: `1.5px solid ${COLORS.border}`,
        }}
      />
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          fontSize: 48,
          animation: "brain-glow 2.2s ease-in-out infinite",
        }}
      >
        🧠
      </div>
      {[0, 1, 2, 3, 4].map((i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            inset: 0,
            animation: "orbit-spin 8s linear infinite",
            animationDelay: `${-(i * 1.6)}s`,
          }}
        >
          <div
            style={{
              position: "absolute",
              top: -5,
              left: "50%",
              width: 11,
              height: 11,
              borderRadius: "50%",
              background: COLORS.accentLight,
              transform: "translateX(-50%)",
              boxShadow: `0 0 10px ${COLORS.accentLight}`,
            }}
          />
        </div>
      ))}
    </div>
  );
}

export default function ProgressPage({ jobId, onDone }) {
  const [job, setJob] = useState({ status: "running", current_agent: "eda_node", progress: 0 });
  const [error, setError] = useState("");
  const [showDetails, setShowDetails] = useState(true);
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
            timingRef.current.durations[prevAgent] = (now - timingRef.current.starts[prevAgent]) / 1000;
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

  const currentIdx = AGENTS.findIndex((a) => a.key === job.current_agent);
  const activeCount = job.status === "done" ? 5 : Math.max(1, currentIdx + 1);
  const stepsLeft = job.status === "done" ? 0 : AGENTS.length - activeCount;
  const estMinutes = stepsLeft === 0 ? 0 : Math.max(1, Math.round(stepsLeft * 0.6));

  const stats = [
    { label: "Agents Active", value: `${activeCount} of 5` },
    { label: "Rows Processed", value: "—" },
    { label: "Est. Time Left", value: job.status === "done" ? "Done" : `~${estMinutes} min` },
  ];

  return (
    <div style={{ maxWidth: 820, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <h1 style={{ color: COLORS.textPrimary, fontSize: 30, fontWeight: 800, marginBottom: 6 }}>
            Analysis in Progress
          </h1>
          <p style={{ color: COLORS.textSecondary, fontSize: 14.5 }}>
            Our AI agents are working to deliver the best insights
          </p>
        </div>
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            color: COLORS.green,
            background: "rgba(34,197,94,0.12)",
            borderRadius: 20,
            padding: "5px 14px",
            fontSize: 12.5,
            fontWeight: 600,
            flexShrink: 0,
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: COLORS.green,
              animation: "pulse-dot 1.8s ease-in-out infinite",
            }}
          />
          Live
        </span>
      </div>

      <div style={{ ...cardStyle, padding: "40px 24px", marginTop: 28, textAlign: "center" }}>
        <OrbitalAnimation />
        <p style={{ color: COLORS.textSecondary, fontSize: 13.5, marginTop: 20 }}>
          This may take a few minutes depending on dataset size
        </p>
      </div>

      <div style={{ ...cardStyle, padding: "24px 28px", marginTop: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
          <span style={{ color: COLORS.textPrimary, fontSize: 16, fontWeight: 700 }}>Agent Workflow</span>
          <span
            onClick={() => setShowDetails((v) => !v)}
            style={{ color: COLORS.accentLight, fontSize: 12.5, cursor: "pointer", fontWeight: 600 }}
          >
            {showDetails ? "Hide Details" : "Show Details"}
          </span>
        </div>

        {AGENTS.map((agent, idx) => {
          const state = getAgentState(agent.key, job.current_agent, job.status);
          const startTime = timingRef.current.starts[agent.key];
          const elapsed = state === "running" && startTime ? (Date.now() - startTime) / 1000 : 0;
          const duration = timingRef.current.durations[agent.key] ?? null;

          return (
            <div
              key={agent.key}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 14,
                padding: "14px 4px",
                borderBottom: idx < AGENTS.length - 1 ? `1px solid ${COLORS.border}` : "none",
              }}
            >
              <span style={{ fontSize: 22, flexShrink: 0 }}>{agent.icon}</span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ color: COLORS.textPrimary, fontSize: 14.5, fontWeight: 600 }}>{agent.label}</div>
                {showDetails && (
                  <div style={{ color: COLORS.textSecondary, fontSize: 12.5, marginTop: 2 }}>
                    {agent.description}
                  </div>
                )}
                {showDetails && state === "running" && (
                  <div className="am-progress-track" style={{ marginTop: 8, maxWidth: 220 }}>
                    <div className="am-progress-fill" />
                  </div>
                )}
              </div>
              <div style={{ textAlign: "right", flexShrink: 0, minWidth: 90 }}>
                <StatusBadge state={state} />
                {state === "running" && (
                  <div
                    style={{
                      color: COLORS.accent,
                      fontSize: 12,
                      marginTop: 4,
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {formatElapsed(elapsed)}
                  </div>
                )}
                {state === "completed" && duration != null && (
                  <div style={{ color: COLORS.textSecondary, fontSize: 11.5, marginTop: 4 }}>
                    {formatElapsed(duration)}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: "flex", gap: 16, marginTop: 20 }}>
        {stats.map((stat) => (
          <div key={stat.label} style={{ ...cardStyle, flex: 1, padding: "16px 20px" }}>
            <div style={{ color: COLORS.textSecondary, fontSize: 12 }}>{stat.label}</div>
            <div style={{ color: COLORS.textPrimary, fontSize: 18, fontWeight: 700, marginTop: 4 }}>
              {stat.value}
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div
          style={{
            color: "var(--danger-text)",
            background: "rgba(220,38,38,0.1)",
            border: "1px solid rgba(220,38,38,0.3)",
            borderRadius: 10,
            padding: "12px 16px",
            fontSize: 13,
            marginTop: 20,
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
}
