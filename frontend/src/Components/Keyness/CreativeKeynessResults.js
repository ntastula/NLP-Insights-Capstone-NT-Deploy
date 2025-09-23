import React, { useState, useMemo } from "react";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import ResultsSummary from "./ResultsSummary";
import SentenceModal from "./SentenceModal";
import KeynessResultsGrid from "./KeynessResultsGrid"
import "./CreativeKeynessResults.css";
import { exportAnalysisToCSV } from "./ExportCsv";
import { exportKeynessToXlsx } from "./ExportXlsx";
import { generateChartData } from "./GenerateChartData";

const posColors = {
  NOUN: "noun",
  VERB: "verb",
  ADJ: "adj",
  ADV: "adv",
  OTHER: "other",
};

const CreativeKeynessResults = ({ results, stats, method, uploadedText }) => {
  const [activeView, setActiveView] = useState("keywords");
  const [selectedWord, setSelectedWord] = useState(null);
  const [sentences, setSentences] = useState([]);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);

  // Ensure results is always an array
  const safeResults = Array.isArray(results) ? results : [];
  console.log("API results:", results);
  console.log("Safe results:", safeResults);

  // Group by POS
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
    safeResults.forEach((item) => {
      if (!uploadedWordsSet.has(item.word.toLowerCase())) return; 
      if (item.pos === "PROPN") return; 

      const pos = (item.pos || item.pos_tag || "OTHER").toUpperCase();
      if (!groups[pos]) groups[pos] = [];
      groups[pos].push(item);
    });
    return groups;
  }, [safeResults, uploadedWordsSet]);

  // Fetch sentences from backend
  const getSentencesContaining = async (word) => {
    if (!uploadedText) return [];
    try {
      const response = await fetch("http://localhost:8000/api/get-sentences/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          uploaded_text: uploadedText,  
          word: word                     
        })
      });
      const data = await response.json();
      return data.sentences || [];
    } catch (err) {
      console.error("Error fetching sentences:", err);
      return [];
    }
  };

  const fetchSummary = async () => {
    setSummaryLoading(true);
    try {
      const response = await fetch("http://localhost:8000/api/get-keyness-summary/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keyness_results: results })
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

  const handleKeywordClick = async (word) => {
    setSelectedWord(word);
    setLoading(true);
    try {
      const fetchedSentences = await getSentencesContaining(word);
      setSentences(fetchedSentences);
    } catch (err) {
      console.error("Error fetching sentences:", err);
      setSentences([]);
    } finally {
      setLoading(false);
    }
  };

  const closeModal = () => {
    setSelectedWord(null);
    setSentences([]);
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

  return (
    <div className="results-container">
      <ResultsSummary stats={stats} selectedMethod={method} comparisonResults={safeResults} />

      {/* View Toggle Buttons */}
      <div className="view-controls">
        {["keywords", "charts", "table", "wordData", "summary"].map((view) => (
          <button
            key={view}
            className={`btn ${activeView === view ? "bg-blue-500 text-white" : ""}`}
            onClick={() => setActiveView(view)}
          >
            {view === "keywords"
              ? "Top Keywords"
              : view === "charts"
              ? "Charts"
              : view === "table"
              ? "Table"
              : view === "wordData"
              ? "Word Data"
              : "Summary"}
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
          {summaryLoading ? (
            "Loading summary..."
          ) : (
            summary
              .split(/\n{2,}|(?<=\.)\s+/) // Split on double newlines or after periods
              .map((p, i) => (
                <p key={i}>{p.trim()}</p>
              ))
          )}
        </div>
      )}

      {/* Keywords View */}
      {activeView === "keywords" && (
        <div className="creative-results">
          {Object.entries(posGroups).map(([pos, words]) => {
            // Map POS codes to full names
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
                      className={`keyword keyword-pill ${posColors[pos] || posColors.OTHER}`}
                      onClick={() => handleKeywordClick(w.word)}
                      tabIndex={0}
                      role="button"
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          handleKeywordClick(w.word);
                        }
                      }}
                      title={`Click to see sentences containing "${w.word}"`}
                    >
                      {w.word}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Charts View */}
      {activeView === "charts" && <Charts results={safeResults} method={method} />}

      {/* Table View */}
      {activeView === "table" && <ResultsTable results={safeResults} method={method} />}

      {/* Word Data View */}
      {activeView === "wordData" && <KeynessResultsGrid results={safeResults} method={method} />}

      {/* Modal for Sentences */}
      {selectedWord && (
        <SentenceModal 
          word={selectedWord} 
          sentences={sentences} 
          onClose={closeModal}
          loading={loading}
        />
      )}
    </div>
  );
};

export default CreativeKeynessResults;