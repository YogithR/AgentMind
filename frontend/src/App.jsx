import { useState } from "react";
import Layout from "./components/Layout";
import UploadPage from "./UploadPage";
import ProgressPage from "./ProgressPage";
import ResultsPage from "./ResultsPage";
import HistoryPage from "./HistoryPage";

function App() {
  const [page, setPage] = useState("upload");
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);

  const handleJobStarted = (id) => {
    setJobId(id);
    setPage("progress");
  };

  const handleDone = (resultData) => {
    setResult(resultData);
    setPage("results");
  };

  const handleReset = () => {
    setJobId(null);
    setResult(null);
    setPage("upload");
  };

  const handleNavigate = (target) => {
    if (target === "progress" && !jobId) return;
    if (target === "results" && !result) return;
    setPage(target);
  };

  const furthestStep = result ? 3 : jobId ? 2 : 1;
  const flowStep = page === "upload" ? 1 : page === "progress" ? 2 : page === "results" ? 3 : furthestStep;

  return (
    <Layout
      activePage={page}
      flowStep={flowStep}
      navEnabled={{ upload: true, progress: !!jobId, results: !!result, history: true }}
      onNavigate={handleNavigate}
    >
      {page === "upload" && <UploadPage onJobStarted={handleJobStarted} />}
      {page === "progress" && <ProgressPage jobId={jobId} onDone={handleDone} />}
      {page === "results" && result && <ResultsPage result={result} onReset={handleReset} />}
      {page === "history" && <HistoryPage />}
    </Layout>
  );
}

export default App;
