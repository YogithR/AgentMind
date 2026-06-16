import { CheckIcon } from "@heroicons/react/24/solid";
import { COLORS } from "../theme";

const STEPS = [
  { num: 1, label: "UPLOAD" },
  { num: 2, label: "PROGRESS" },
  { num: 3, label: "RESULTS" },
];

export default function StepIndicator({ currentStep }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "22px 0",
        borderBottom: `1px solid ${COLORS.border}`,
      }}
    >
      {STEPS.map((step, i) => {
        const state = step.num < currentStep ? "completed" : step.num === currentStep ? "active" : "future";
        return (
          <div key={step.num} style={{ display: "flex", alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  width: 26,
                  height: 26,
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 12,
                  fontWeight: 700,
                  flexShrink: 0,
                  background: state === "future" ? "transparent" : COLORS.accent,
                  border: `1.5px solid ${state === "future" ? COLORS.border : COLORS.accent}`,
                  color: state === "future" ? COLORS.textSecondary : "var(--btn-text)",
                }}
              >
                {state === "completed" ? <CheckIcon style={{ width: 14, height: 14 }} /> : step.num}
              </div>
              <span
                style={{
                  fontSize: 12.5,
                  fontWeight: 700,
                  letterSpacing: "0.06em",
                  color: state === "future" ? COLORS.textSecondary : COLORS.accent,
                }}
              >
                {step.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div
                style={{
                  width: 64,
                  height: 1.5,
                  margin: "0 16px",
                  background: step.num < currentStep ? COLORS.accent : COLORS.border,
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
