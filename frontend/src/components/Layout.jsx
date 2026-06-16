import Sidebar from "./Sidebar";
import StepIndicator from "./StepIndicator";
import { useTheme } from "../ThemeContext";

export default function Layout({ activePage, flowStep, navEnabled, onNavigate, children }) {
  const { theme } = useTheme();

  return (
    <div className={`theme-${theme}`} style={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar activePage={activePage} enabled={navEnabled} onNavigate={onNavigate} />
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <StepIndicator currentStep={flowStep} />
        <div style={{ flex: 1, padding: "40px 48px" }}>{children}</div>
      </div>
    </div>
  );
}
