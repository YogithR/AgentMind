import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const styles = {
  page: {
    minHeight: "100vh",
    background: "#F5F7FA",
    fontFamily: "Arial, sans-serif",
    padding: "40px 16px",
    boxSizing: "border-box",
  },
  container: { maxWidth: 700, margin: "0 auto" },
  heading: { color: "#1F4E79", fontSize: 28, fontWeight: 700, marginBottom: 4 },
  sub: { color: "#6B7280", fontSize: 14, marginBottom: 32 },
  card: {
    background: "#fff",
    borderRadius: 12,
    boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
    padding: "28px 32px",
    marginBottom: 20,
  },
  sectionTitle: {
    color: "#1F4E79",
    fontSize: 13,
    fontWeight: 700,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 16,
  },
  modelRow: { display: "flex", alignItems: "baseline", gap: 16, flexWrap: "wrap" },
  modelName: { color: "#1F4E79", fontSize: 32, fontWeight: 700 },
  accuracy: { color: "#0EA5E9", fontSize: 40, fontWeight: 800 },
  improvement: {
    background: "#DCFCE7",
    color: "#15803D",
    borderRadius: 20,
    padding: "4px 14px",
    fontSize: 15,
    fontWeight: 700,
  },
  warningBox: {
    background: "#FFFBEB",
    border: "1.5px solid #FCD34D",
    borderRadius: 10,
    padding: "14px 18px",
    marginBottom: 8,
  },
  warningText: { color: "#92400E", fontSize: 14 },
  warningIcon: { marginRight: 8 },
  critiqueBox: {
    background: "#F8FAFC",
    border: "1.5px solid #E2E8F0",
    borderRadius: 10,
    padding: "16px 18px",
    maxHeight: 280,
    overflowY: "auto",
    fontSize: 14,
    color: "#374151",
    lineHeight: 1.7,
    whiteSpace: "pre-wrap",
  },
  button: {
    display: "block",
    width: "100%",
    padding: "13px 0",
    background: "#1F4E79",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    fontSize: 16,
    fontWeight: 700,
    cursor: "pointer",
    marginTop: 8,
  },
  expId: { color: "#94A3B8", fontSize: 11, marginTop: 12, textAlign: "center" },
};

const BAR_COLORS = ["#1F4E79", "#3B82F6", "#93C5FD"];

export default function ResultsPage({ result, onReset }) {
  const {
    best_model,
    accuracy,
    improvement,
    top_features = [],
    warnings = [],
    critique = "",
    experiment_id = "",
  } = result;

  const chartData = top_features.slice(0, 5).map(([name, val]) => ({
    name,
    shap: val,
  }));

  const improvementPct = improvement != null ? (improvement * 100).toFixed(2) : null;
  const accuracyPct = accuracy != null ? (accuracy * 100).toFixed(1) : "—";

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.heading}>Results</div>
        <div style={styles.sub}>Pipeline complete — here's what the agents found</div>

        {/* Model + Accuracy */}
        <div style={styles.card}>
          <div style={styles.sectionTitle}>Best Model</div>
          <div style={styles.modelRow}>
            <div style={styles.modelName}>{best_model}</div>
            <div style={styles.accuracy}>{accuracyPct}%</div>
            {improvementPct && (
              <div style={styles.improvement}>+{improvementPct}% from tuning</div>
            )}
          </div>
        </div>

        {/* Top Features Bar Chart */}
        {chartData.length > 0 && (
          <div style={styles.card}>
            <div style={styles.sectionTitle}>Top Features (SHAP importance)</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 12 }} tickFormatter={(v) => v.toFixed(2)} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 13, fill: "#374151" }} width={80} />
                <Tooltip formatter={(v) => v.toFixed(4)} />
                <Bar dataKey="shap" radius={[0, 6, 6, 0]}>
                  {chartData.map((_, i) => (
                    <Cell key={i} fill={BAR_COLORS[i] || "#93C5FD"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Warnings */}
        {warnings.length > 0 && (
          <div style={styles.card}>
            <div style={styles.sectionTitle}>Warnings</div>
            {warnings.map((w, i) => (
              <div key={i} style={styles.warningBox}>
                <span style={styles.warningIcon}>⚠️</span>
                <span style={styles.warningText}>{w}</span>
              </div>
            ))}
          </div>
        )}

        {/* Groq Critique */}
        <div style={styles.card}>
          <div style={styles.sectionTitle}>AI Critique (Groq)</div>
          <div style={styles.critiqueBox}>{critique}</div>
        </div>

        {/* Run Another */}
        <div style={styles.card}>
          <button style={styles.button} onClick={onReset}>
            Run Another Analysis
          </button>
          {experiment_id && (
            <div style={styles.expId}>Experiment ID: {experiment_id}</div>
          )}
        </div>
      </div>
    </div>
  );
}
