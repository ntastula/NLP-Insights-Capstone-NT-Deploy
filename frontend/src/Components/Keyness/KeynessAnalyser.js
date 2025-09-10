import React, { useState } from "react";
import ResultsSummary from "./ResultsSummary";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import CreativeKeynessResults from "./CreativeKeynessResults";
import "./ProgressBar.css";

const ProgressBar = ({ loading }) => {
  const [progress, setProgress] = useState(0);

  React.useEffect(() => {
    let timer;
    if (loading) {
      setProgress(0);
      timer = setInterval(() => {
        setProgress((old) => (old < 90 ? old + Math.random() * 3 : old));
      }, 200);
    } else {
      setProgress(100);
      const reset = setTimeout(() => setProgress(0), 500);
      return () => clearTimeout(reset);
    }
    return () => clearInterval(timer);
  }, [loading]);

  return (
    <div className="progress-container">
      <div className="progress-fill" style={{ width: `${progress}%` }}></div>
      <div className="progress-text">{Math.floor(progress)}%</div>
    </div>
  );
};

const KeynessAnalyser = ({ uploadedText, uploadedPreview, corpusPreview, method, onBack }) => {
  const [comparisonResults, setComparisonResults] = useState([]);
  const [stats, setStats] = useState({ uploaded_total: 0, sample_total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisDone, setAnalysisDone] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState("nltk");
  const [filterMode, setFilterMode] = useState("content");

  const performAnalysis = async (method) => {
  if (!uploadedText) return;
  setLoading(true);
  setError("");
  setAnalysisDone(false);
  setSelectedMethod(method);

  try {
    const response = await fetch("http://localhost:8000/api/analyse-keyness/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        uploaded_text: uploadedText,
        method: method.toLowerCase(),
        filter_mode: filterMode,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      // data.results is always a flat array
      const resultsArray = Array.isArray(data.results) ? data.results : [];

      setComparisonResults(resultsArray);

      setStats({
        uploadedTotal: data.uploaded_total ?? uploadedText.split(/\s+/).length,
        corpusTotal: data.corpus_total ?? 0,
      });

      setAnalysisDone(true);
    } else {
      setError(data.error || "Analysis failed");
    }
  } catch (err) {
    console.error("Analysis error:", err);
    setError("Analysis failed: " + err.message);
  } finally {
    setLoading(false);
  }
};




  return (
    
    <div className="mb-6">
      <button
  onClick={onBack}
  className="mb-6 bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded shadow"
>
  ← Back
</button>

{/* Word Filtering Options */}
    <div className="mb-6 text-center">
      <p className="mb-2 font-medium">Select an option for what words in your text you would like analysed:</p>
      <div className="flex justify-center gap-6">
        <label className="flex items-center gap-2">
          <input
            type="radio"
            name="filterMode"
            value="content"
            checked={filterMode === "content"}
            onChange={(e) => setFilterMode(e.target.value)}
            className="mr-1"
          />
          <span>Only content words (nouns, verbs, adjectives, adverbs)</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="radio"
            name="filterMode"
            value="all"
            checked={filterMode === "all"}
            onChange={(e) => setFilterMode(e.target.value)}
            className="mr-1"
          />
          <span>All words</span>
        </label>
      </div>
    </div>

      {/* Analyse Button */}
      <div className="text-center mb-6 flex justify-center gap-4">
  <button
    onClick={() => performAnalysis("NLTK")}
    disabled={loading || !uploadedText}
    className="btn"
  >
    Analyse with NLTK
  </button>

  <button
    onClick={() => performAnalysis("sklearn")}
    disabled={loading || !uploadedText}
    className="btn"
  >
    Analyse with Scikit-Learn
  </button>

  <button
    onClick={() => performAnalysis("gensim")}
    disabled={loading || !uploadedText}
    className="btn"
  >
    Analyse with Gensim
  </button>

  <button
  onClick={() => performAnalysis("spaCy")}
  disabled={loading || !uploadedText}
  className="btn"
>
  Analyse with spaCy
</button>

</div>

     {loading && (
  <div className="w-full max-w-xl mx-auto mt-4">
  <ProgressBar loading={loading} />
</div>
)}

      {error && <p className="text-red-500">{error}</p>}

      {analysisDone && (
  <CreativeKeynessResults
    results={comparisonResults}
    uploadedText={uploadedText}
    method={selectedMethod}
    stats={stats}
  />
)}

    </div>
  );
};

export default KeynessAnalyser;
