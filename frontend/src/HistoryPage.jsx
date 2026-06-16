import { useEffect, useState } from "react";
import axios from "axios";
import { ClockIcon } from "@heroicons/react/24/outline";
import { COLORS, cardStyle } from "./theme";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8080";

export default function HistoryPage() {
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    axios
      .get(`${API_BASE_URL}/history`)
      .then(({ data }) => setItems(data))
      .catch(() => setError("Could not load experiment history."));
  }, []);

  return (
    <div style={{ maxWidth: 820, margin: "0 auto" }}>
      <h1 style={{ color: COLORS.textPrimary, fontSize: 30, fontWeight: 800, marginBottom: 6 }}>
        Experiment History
      </h1>
      <p style={{ color: COLORS.textSecondary, fontSize: 14.5, marginBottom: 28 }}>
        Your 5 most recent analyses
      </p>

      {error && <div style={{ color: "var(--danger-text)", fontSize: 13.5 }}>{error}</div>}

      {items?.length === 0 && (
        <div style={{ color: COLORS.textSecondary, fontSize: 14 }}>
          No experiments yet — run an analysis to see it here.
        </div>
      )}

      {items?.map((item) => (
        <div key={item.experiment_id} style={{ ...cardStyle, padding: "18px 22px", marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
            <div style={{ color: COLORS.textPrimary, fontSize: 16, fontWeight: 700 }}>
              {item.model_name || "Unknown model"}
            </div>
            <div style={{ color: COLORS.accent, fontSize: 18, fontWeight: 800, flexShrink: 0 }}>
              {item.accuracy != null ? `${(item.accuracy * 100).toFixed(1)}%` : "—"}
            </div>
          </div>
          <div style={{ color: COLORS.textSecondary, fontSize: 13, marginTop: 4 }}>
            {item.goal || item.dataset_name || "—"}
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              color: COLORS.textSecondary,
              fontSize: 11.5,
              marginTop: 10,
            }}
          >
            <ClockIcon style={{ width: 13, height: 13 }} />
            {item.timestamp || "—"}
          </div>
        </div>
      ))}
    </div>
  );
}
