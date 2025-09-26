import React, { useState, useMemo } from "react";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import ResultsSummary from "./ResultsSummary";
import KeynessResultsGrid from "./KeynessResultsGrid";
import "./CreativeKeynessResults.css";
import { exportKeynessToXlsx } from "./ExportXlsx";

const posColors = {
  NOUN: "noun",
  VERB: "verb",
  ADJ: "adj",
  ADV: "adv",
  OTHER: "other",
};

const CreativeKeynessResults = ({ results, stats, method, uploadedText, genre, onWordDetail }) => {
  const [activeView, setActiveView] = useState("keywords");
  const [summary, setSummary] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);

  const safeResults = Array.isArray(results) ? results : [];

  // Group by POS and filter to only words from uploaded text
  const uploadedWordsSet = useMemo(() => {
    if (!uploadedText) return new Set();
    return new Set(
      uploadedText
        .toLowerCase()
        .match(/\b\w+\b/g)
    );
  }, [uploadedText]);

  const posGroups = useMemo(() => {
    const groups = {};
    safeResults.forEach((r) => {
      if (!uploadedWordsSet.has(r.word.toLowerCase())) return;
      if (r.pos === "PROPN") return; // Skip proper nouns

      const pos = (r.pos || r.pos_tag || "OTHER").toUpperCase();
      if (!groups[pos]) groups[pos] = [];
      groups[pos].push(r);
    });
    return groups;
  }, [safeResults, uploadedWordsSet]);

  // Fetch summary for summary tab
  const fetchSummary = async () => {
    setSummaryLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/get-keyness-summary/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keyness_results: results }),
      });
      const data = await response.json();
      setSummary(data.summary || "No summary available");
    } catch (err) {
      setSummary("Error retrieving summary");
    } finally {
      setSummaryLoading(false);
    }
  };

  React.useEffect(() => {
    if (activeView === "summary" && !summary && !summaryLoading) {
      fetchSummary();
    }
  }, [activeView]);

  // Handle clicking on a keyword
  const handleKeywordClick = (w) => {
  if (!w) return;
  const wordData = safeResults.find(item => item.word?.toLowerCase() === w.word.toLowerCase());
  if (wordData && onWordDetail) {
    onWordDetail({
      word: w.word,
      wordData,
      uploadedText,
      method,
      results: safeResults
    });
  }
};

  const closeModal = () => {
    // No longer needed since we're navigating to a new page
    // Keep for backward compatibility if needed elsewhere
  };

  if (!results || Object.keys(results).length === 0) {
    return (
      <div className="results-container">
        <div className="no-results">
          No significant keywords found.
        </div>
      </div>
    );
  }

  const chartData = results?.slice(0, 20).map((r) => ({
    label: r.word,
    value: r.keyness ?? r.log_likelihood ?? r.chi2 ?? r.tfidf_score ?? 0,
  }));

  const viewLabels = {
    keywords: "Keywords",
    charts: "Charts",
    table: "Table",
    summary: "Summary"
  };

  return (
    <div className="results-container">
      <ResultsSummary 
        stats={stats} 
        selectedMethod={method} 
        comparisonResults={safeResults} 
        genre={genre} 
      />

      {/* Main View Toggle Buttons */}
      <div className="view-controls">
        {Object.keys(viewLabels).map((view) => (
          <button
            key={view}
            className={`btn ${activeView === view ? "bg-blue-500 text-white" : ""}`}
            onClick={() => setActiveView(view)}
          >
            {viewLabels[view]}
          </button>
        ))}

        {/* Download button */}
        <button
          className="btn bg-green-500 text-white"
          onClick={() =>
            exportKeynessToXlsx(
              safeResults,
              method,
              stats,
              posGroups,
              [],
              chartData
            )
          }
        >
          Download XLSX
        </button>
      </div>

      {/* Summary View */}
      {activeView === "summary" && (
        <div className="keyness-summary">
          {summaryLoading
            ? "Loading summary..."
            : summary.split(/\n{2,}|(?<=\.)\s+/).map((p, i) => (
                <p key={i}>{p.trim()}</p>
              ))}
        </div>
      )}

      {/* Keywords View */}
      {activeView === "keywords" && (
        <div className="creative-results">
          <div className="keywords-header">
            <h2>Top 50 Most Significant Keywords from Your Text</h2>
            <p>Click on a word to see detailed analysis including sentences, synonyms, and more.</p>
          </div>

          {Object.keys(posGroups).length === 0 ? (
            <div className="no-keywords">No significant keywords found</div>
          ) : (
            <>
              {Object.entries(posGroups).map(([pos, words]) => {
                const posFullNames = {
                  ADV: "Adverbs",
                  NOUN: "Nouns",
                  VERB: "Verbs",
                  ADJ: "Adjectives",
                  OTHER: "Other Words",
                };
                const posLabel = posFullNames[pos] || pos;
                const posKey = pos.toLowerCase();

                return (
                  <div key={pos} className="pos-section">
                    <h3 data-pos={posKey}>{posLabel}</h3>
                    <div className="word-list">
                      {words.map((w, idx) => (
                        <span
                          key={`${w.word}-${idx}`}
                          className={`keyword keyword-pill ${posColors[w.pos] || posColors.OTHER}`}
                          onClick={() => handleKeywordClick(w)}
                          tabIndex={0}
                          role="button"
                          onKeyPress={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              handleKeywordClick(w);
                            }
                          }}
                          title={`Click for detailed analysis of "${w.word}"`}
                        >
                          {w.word}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}

      {/* Charts View */}
      {activeView === "charts" && <Charts results={safeResults} method={method} />}

      {/* Table View */}
      {activeView === "table" && <ResultsTable results={safeResults} method={method} />}
    </div>
  );
};

export default CreativeKeynessResults;