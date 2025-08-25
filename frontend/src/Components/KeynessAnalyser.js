// src/Components/KeynessAnalyser.js
import React, { useState } from "react";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import ResultsSummary from "./ResultsSummary";
import { Loader2, BarChart3 } from "lucide-react";

const KeynessAnalyser = ({ uploadedText }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const performAnalysis = async () => {
    if (!uploadedText) return;
    setLoading(true);
    setError("");
    setResults(null);

    try {
      const response = await fetch("http://localhost:8000/api/analyse-keyness/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uploaded_text: uploadedText }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const analysisResults = await response.json();
      if (analysisResults.error) throw new Error(analysisResults.error);

      setResults(analysisResults);
    } catch (err) {
      console.error(err);
      setError("Analysis failed: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-6">
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
              Analyse Keyness Statistics
            </>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-300 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <>
          <ResultsSummary results={results} />
          <Charts results={results} />
          <ResultsTable results={results} />
        </>
      )}
    </div>
  );
};

export default KeynessAnalyser;