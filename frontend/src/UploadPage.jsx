import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { CloudArrowUpIcon, ShieldCheckIcon, CheckCircleIcon } from "@heroicons/react/24/outline";
import { COLORS } from "./theme";

const API = "http://localhost:8080";

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

  const canSubmit = file && goal.trim() && targetColumn.trim() && !loading;

  return (
    <div style={{ maxWidth: 760, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16 }}>
        <div>
          <h1 style={{ color: COLORS.textPrimary, fontSize: 30, fontWeight: 800, marginBottom: 6 }}>
            Upload Your Dataset
          </h1>
          <p style={{ color: COLORS.textSecondary, fontSize: 14.5 }}>
            Start by uploading your data and telling us what to predict
          </p>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 20,
            padding: "6px 14px",
            flexShrink: 0,
            whiteSpace: "nowrap",
          }}
        >
          <ShieldCheckIcon style={{ width: 15, height: 15, color: COLORS.accentLight }} />
          <span style={{ color: COLORS.textSecondary, fontSize: 12 }}>Enterprise Grade Security</span>
        </div>
      </div>

      <div
        {...getRootProps()}
        style={{
          marginTop: 32,
          border: `2px dashed ${isDragActive ? COLORS.accent : file ? COLORS.green : COLORS.border}`,
          background: "var(--input-bg)",
          borderRadius: 14,
          padding: "48px 24px",
          textAlign: "center",
          cursor: "pointer",
          transition: "border-color 0.2s",
        }}
      >
        <input {...getInputProps()} />
        {file ? (
          <>
            <CheckCircleIcon style={{ width: 40, height: 40, color: COLORS.green, margin: "0 auto 12px" }} />
            <div style={{ color: COLORS.textPrimary, fontWeight: 600, fontSize: 15 }}>{file.name}</div>
            <div style={{ color: COLORS.textSecondary, fontSize: 12.5, marginTop: 4 }}>
              Ready to upload — drop a new file to replace it
            </div>
          </>
        ) : (
          <>
            <CloudArrowUpIcon style={{ width: 40, height: 40, color: COLORS.accent, margin: "0 auto 12px" }} />
            <div style={{ color: COLORS.textPrimary, fontSize: 15, marginBottom: 6 }}>
              {isDragActive ? "Drop your file here..." : "Drag & drop your file here"}
            </div>
            <div style={{ color: COLORS.accentLight, fontSize: 13.5, textDecoration: "underline", marginBottom: 14 }}>
              or browse to upload
            </div>
            <div style={{ color: COLORS.textSecondary, fontSize: 12, marginBottom: 14 }}>
              Supports: CSV files | Max file size: 500MB
            </div>
            <span
              style={{
                display: "inline-block",
                border: `1px solid ${COLORS.accent}`,
                color: COLORS.accent,
                borderRadius: 20,
                padding: "3px 14px",
                fontSize: 11.5,
                fontWeight: 600,
              }}
            >
              CSV
            </span>
          </>
        )}
      </div>

      <div style={{ display: "flex", gap: 16, marginTop: 24 }}>
        <div style={{ flex: 1 }}>
          <label
            style={{
              display: "block",
              color: COLORS.textSecondary,
              fontSize: 12.5,
              fontWeight: 600,
              marginBottom: 6,
            }}
          >
            What do you want to predict?
          </label>
          <input
            className="am-input"
            placeholder='e.g. "predict which passengers survived"'
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label
            style={{
              display: "block",
              color: COLORS.textSecondary,
              fontSize: 12.5,
              fontWeight: 600,
              marginBottom: 6,
            }}
          >
            Target column name
          </label>
          <input
            className="am-input"
            placeholder='e.g. "Survived" or "churn"'
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
          />
        </div>
      </div>

      <button
        className="am-btn-primary"
        style={{ width: "100%", padding: "15px 0", marginTop: 28 }}
        onClick={handleSubmit}
        disabled={!canSubmit}
      >
        {loading ? "Starting..." : "Upload & Start Analysis →"}
      </button>

      {error && (
        <div
          style={{
            color: "var(--danger-text)",
            background: "rgba(220,38,38,0.1)",
            border: "1px solid rgba(220,38,38,0.3)",
            borderRadius: 10,
            padding: "10px 14px",
            fontSize: 13,
            marginTop: 16,
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
}
