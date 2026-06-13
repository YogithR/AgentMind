import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";

const API = "http://localhost:8080";

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
    width: 500,
    maxWidth: "95vw",
  },
  heading: {
    color: "#1F4E79",
    fontSize: 28,
    fontWeight: 700,
    marginBottom: 6,
  },
  sub: {
    color: "#6B7280",
    fontSize: 14,
    marginBottom: 32,
  },
  dropzone: (active, hasFile) => ({
    border: `2px dashed ${active ? "#1F4E79" : hasFile ? "#22C55E" : "#CBD5E1"}`,
    borderRadius: 10,
    padding: "32px 20px",
    textAlign: "center",
    cursor: "pointer",
    background: active ? "#EFF6FF" : hasFile ? "#F0FDF4" : "#F8FAFC",
    transition: "all 0.2s",
    marginBottom: 24,
  }),
  dropIcon: { fontSize: 36, marginBottom: 8 },
  dropText: { color: "#374151", fontSize: 14 },
  fileName: { color: "#1F4E79", fontWeight: 600, fontSize: 13, marginTop: 6 },
  label: {
    display: "block",
    color: "#374151",
    fontSize: 13,
    fontWeight: 600,
    marginBottom: 6,
  },
  input: {
    width: "100%",
    padding: "10px 14px",
    borderRadius: 8,
    border: "1.5px solid #CBD5E1",
    fontSize: 14,
    marginBottom: 18,
    boxSizing: "border-box",
    outline: "none",
    fontFamily: "Arial, sans-serif",
  },
  button: (disabled) => ({
    width: "100%",
    padding: "13px 0",
    background: disabled ? "#93C5FD" : "#1F4E79",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    fontSize: 16,
    fontWeight: 700,
    cursor: disabled ? "not-allowed" : "pointer",
    transition: "background 0.2s",
    marginTop: 4,
  }),
  error: {
    color: "#DC2626",
    fontSize: 13,
    marginTop: 12,
    background: "#FEF2F2",
    borderRadius: 6,
    padding: "8px 12px",
  },
};

export default function UploadPage({ onJobStarted }) {
  const [file, setFile] = useState(null);
  const [goal, setGoal] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onDrop = useCallback((accepted) => {
    if (accepted.length > 0) setFile(accepted[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    multiple: false,
  });

  const handleSubmit = async () => {
    if (!file || !goal.trim() || !targetColumn.trim()) {
      setError("Please fill in all fields and upload a CSV file.");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const form = new FormData();
      form.append("csv_file", file);
      form.append("goal", goal.trim());
      form.append("target_column", targetColumn.trim());
      const { data } = await axios.post(`${API}/run`, form);
      onJobStarted(data.job_id);
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to start job. Is the server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.heading}>AgentMind</div>
        <div style={styles.sub}>Upload a dataset and let 5 AI agents analyze it for you</div>

        <div {...getRootProps()} style={styles.dropzone(isDragActive, !!file)}>
          <input {...getInputProps()} />
          <div style={styles.dropIcon}>{file ? "✅" : "📂"}</div>
          {file ? (
            <>
              <div style={styles.dropText}>File selected</div>
              <div style={styles.fileName}>{file.name}</div>
            </>
          ) : (
            <div style={styles.dropText}>
              {isDragActive ? "Drop your CSV here..." : "Drag & drop a CSV file, or click to browse"}
            </div>
          )}
        </div>

        <label style={styles.label}>What do you want to predict?</label>
        <input
          style={styles.input}
          placeholder='e.g. "predict which passengers survived"'
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
        />

        <label style={styles.label}>Target column name</label>
        <input
          style={styles.input}
          placeholder='e.g. "Survived" or "churn"'
          value={targetColumn}
          onChange={(e) => setTargetColumn(e.target.value)}
        />

        <button
          style={styles.button(loading || !file || !goal || !targetColumn)}
          onClick={handleSubmit}
          disabled={loading || !file || !goal || !targetColumn}
        >
          {loading ? "Starting..." : "Run Pipeline"}
        </button>

        {error && <div style={styles.error}>{error}</div>}
      </div>
    </div>
  );
}
