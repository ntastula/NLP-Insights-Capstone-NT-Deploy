// src/Components/KeynessAnalyser.js
import React, { useState, useEffect } from "react";
import Charts from "./Charts";
import { BarChart3, Loader2 } from "lucide-react";
import ResultsSummary from "./ResultsSummary";
import KeynessResultsGrid from "./KeynessResultsGrid";
import ResultsTable from "./ResultsTable";

const KeynessAnalyser = ({ uploadedText, uploadedPreview, corpusPreview, method = "NLTK" }) => {
  const [comparisonResults, setComparisonResults] = useState([]);
  const [stats, setStats] = useState({ uploadedTotal: 0, corpusTotal: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisDone, setAnalysisDone] = useState(false);

  // Perform keyness analysis
  const performAnalysis = async () => {
    console.log("Perform analysis clicked. Uploaded text:", uploadedText);
if (!uploadedText) return;

    if (!uploadedText) return;
    setLoading(true);
    setError("");
    setAnalysisDone(false);
    

    try {
      console.log("Button clicked!");
      const response = await fetch("http://localhost:8000/api/analyse-keyness/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uploaded_text: uploadedText }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      console.log("Received data:", data);
      // if (data.error) throw new Error(data.error);

      // const results = data.nltk || [];
      // setComparisonResults(results);

      // // Calculate stats
      // setStats({
      //   uploadedTotal: uploadedText.split(/\s+/).length,
      //   corpusTotal: data.corpus_total || 0,
      // });
      // Set state using the correct data shape
    setComparisonResults(data.results || []); 
    setStats({
  uploadedTotal: uploadedText.split(/\s+/).length,
  corpusTotal: data.corpus_total || 0
});

    setAnalysisDone(true);
    console.log("stats:", stats);
console.log("comparisonResults:", comparisonResults);

    

      // setAnalysisDone(true);
    } catch (err) {
      console.error("Analysis error:", err);
      setError("Analysis failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Analyse Button */}
      <div className="text-center mb-6">
        <button
          onClick={performAnalysis}
          disabled={loading || !uploadedText}
          className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-8 py-4 rounded-lg font-bold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:from-green-600 hover:to-emerald-700 transform hover:-translate-y-1 transition-all shadow-lg flex items-center gap-3 mx-auto"
        >
          {loading ? (
            <>
              <Loader2 className="w-6 h-6 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <BarChart3 className="w-6 h-6" />
              Analyse Keyness
            </>
          )}
        </button>
      </div>

      

{/* Analysis Summary */}
{analysisDone && (
  <>
{/* Summary */}
          <ResultsSummary
    uploadedTotal={stats.uploadedTotal}
    corpusTotal={stats.corpusTotal}
    sigKeywords={comparisonResults.length}
    method={method}
  />

          {/* Significant keywords grid */}
          <KeynessResultsGrid results={comparisonResults} />

          {/* Charts */}
          <Charts results={comparisonResults} />

          {/* Results Table */}
          {analysisDone && comparisonResults.length > 0 && (
          <ResultsTable results={comparisonResults} />
)}


          </>
)}

      {/* Loading / Error messages */}
      {loading && <p className="text-center text-gray-600">Loading analysis...</p>}
      {error && <p className="text-center text-red-500">{error}</p>}
    </div>
    
  );
};

export default KeynessAnalyser;
