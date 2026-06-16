import {
  ArrowUpTrayIcon,
  ChartBarIcon,
  ChartPieIcon,
  ClockIcon,
} from "@heroicons/react/24/outline";
import { COLORS } from "../theme";
import { useTheme } from "../ThemeContext";

const NAV_ITEMS = [
  { key: "upload", label: "Upload", Icon: ArrowUpTrayIcon },
  { key: "progress", label: "Progress", Icon: ChartBarIcon },
  { key: "results", label: "Results", Icon: ChartPieIcon },
  { key: "history", label: "History", Icon: ClockIcon },
];

export default function Sidebar({ activePage, enabled = {}, onNavigate }) {
  const { theme, toggleTheme } = useTheme();

  return (
    <div
      style={{
        width: 220,
        flexShrink: 0,
        background: COLORS.sidebarBg,
        borderRight: `1px solid ${COLORS.border}`,
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
      }}
    >
      <div style={{ padding: "28px 24px", display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 22 }}>🧠</span>
        <span style={{ color: COLORS.textPrimary, fontWeight: 800, fontSize: 17, letterSpacing: "-0.01em" }}>
          AgentMind
        </span>
      </div>

      <nav style={{ flex: 1, padding: "8px 0" }}>
        {NAV_ITEMS.map(({ key, label, Icon }) => {
          const isActive = activePage === key;
          const isEnabled = enabled[key] !== false;
          return (
            <div
              key={key}
              onClick={() => isEnabled && onNavigate?.(key)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "12px 24px",
                borderLeft: `3px solid ${isActive ? COLORS.accent : "transparent"}`,
                background: isActive ? "var(--accent-tint-soft)" : "transparent",
                color: isActive ? COLORS.accent : isEnabled ? COLORS.textSecondary : "var(--nav-disabled)",
                fontWeight: isActive ? 700 : 500,
                fontSize: 14,
                cursor: isEnabled ? "pointer" : "default",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              <Icon style={{ width: 18, height: 18, flexShrink: 0 }} />
              {label}
            </div>
          );
        })}
      </nav>

      <div style={{ padding: "16px 24px", borderTop: `1px solid ${COLORS.border}` }}>
        <div style={{ color: COLORS.textSecondary, fontSize: 11, fontWeight: 600, marginBottom: 8 }}>
          Theme
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 14 }}>☀️</span>
          <button
            type="button"
            className="theme-toggle-track"
            data-theme={theme}
            onClick={toggleTheme}
            aria-label="Toggle light/dark theme"
          >
            <span className="theme-toggle-thumb" />
          </button>
          <span style={{ fontSize: 14 }}>🌙</span>
        </div>
      </div>

      <div style={{ padding: "20px 24px", borderTop: `1px solid ${COLORS.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: COLORS.green,
              animation: "pulse-dot 1.8s ease-in-out infinite",
              flexShrink: 0,
            }}
          />
          <span style={{ color: COLORS.textSecondary, fontSize: 12.5 }}>AI Agents — 5/5 active</span>
        </div>
      </div>
    </div>
  );
}
