import { useState } from "react";
import UploadPage from "./UploadPage";
import ProgressPage from "./ProgressPage";
import ResultsPage from "./ResultsPage";

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

  if (page === "upload") return <UploadPage onJobStarted={handleJobStarted} />;
  if (page === "progress") return <ProgressPage jobId={jobId} onDone={handleDone} />;
  if (page === "results") return <ResultsPage result={result} onReset={handleReset} />;
}

export default App;
