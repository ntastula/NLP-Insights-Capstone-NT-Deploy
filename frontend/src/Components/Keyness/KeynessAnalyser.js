// src/Components/KeynessAnalyser.js
import React, { useState } from "react";
import ResultsTable from "./ResultsTable";
import KeynessResultsGrid from "./KeynessResultsGrid";
import Charts from "./Charts";
import ResultsSummary from "./ResultsSummary";

const KeynessAnalyser = ({ uploadedText, uploadedPreview, corpusPreview, method, onBack }) => {
  const [comparisonResults, setComparisonResults] = useState([]);
  const [stats, setStats] = useState({ uploaded_total: 0, sample_total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisDone, setAnalysisDone] = useState(false);
  const [selectedMethod, setSelectedMethod] = useState("nltk"); 

  const performAnalysis = async (method) => {
  if (!uploadedText) return;
  setLoading(true);
  setError("");
  setAnalysisDone(false);
  setSelectedMethod(method);

  try {
    console.log("Perform analysis clicked. Method:", method);

    const response = await fetch("http://localhost:8000/api/analyse-keyness/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uploaded_text: uploadedText, method: method.toLowerCase() }),
    });

    const data = await response.json();

    if (data.error) throw new Error(data.error);

    console.log("Received data:", data);

    // Update state with results
    setComparisonResults(data.results.results || data.results); // support both methods
    setStats({
  uploadedTotal: data.uploaded_total || uploadedText.split(/\s+/).length,
  corpusTotal: data.corpus_total || 0
});


    setSelectedMethod(method); // store which method was used
    setAnalysisDone(true);

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


      {loading && <p className="text-gray-500 italic">Analyzing text...</p>}
      {error && <p className="text-red-500">{error}</p>}

      {analysisDone && (
  <>
    {/* Results Summary */}
    <ResultsSummary
  stats={stats}
  selectedMethod={selectedMethod}
  comparisonResults={comparisonResults}
/>



    {/* Significant Keywords Grid */}
    <KeynessResultsGrid results={comparisonResults.slice(0, 20)} method={selectedMethod} />

    {/* Charts */}
    <Charts results={comparisonResults.results ?? comparisonResults} method={selectedMethod} />


    {/* Full Results Table */}
    <ResultsTable results={comparisonResults} method={selectedMethod} />
  </>
)}


    </div>
  );
};

export default KeynessAnalyser;
