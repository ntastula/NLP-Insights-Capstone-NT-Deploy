import React, { useState, useMemo } from "react";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import ResultsSummary from "./ResultsSummary";
import SentenceModal from "./SentenceModal";
import KeynessResultsGrid from "./KeynessResultsGrid";
import "./CreativeKeynessResults.css";
import { exportAnalysisToCSV } from "./ExportCsv";
import { exportKeynessToXlsx } from "./ExportXlsx";
import { generateChartData } from "./GenerateChartData";
import axios from "axios";
import Modal from "../Modal";

const posColors = {
  NOUN: "noun",
  VERB: "verb",
  ADJ: "adj",
  ADV: "adv",
  OTHER: "other",
};

const CreativeKeynessResults = ({ results, stats, method, uploadedText, genre }) => {
  const [activeView, setActiveView] = useState("keywords");
  const [selectedWord, setSelectedWord] = useState(null);
  const [sentences, setSentences] = useState([]);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [synonyms, setSynonyms] = useState([]);
  const [loadingSynonyms, setLoadingSynonyms] = useState(false);
  const [analysis, setAnalysis] = useState(null);

  const safeResults = Array.isArray(results) ? results : [];

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
          word: word,
        }),
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

  // Sentences tab handler
  const handleKeywordClickSentences = async (word) => {
    setSelectedWord(word);
    setLoading(true);
    setSentences([]);
    try {
      const fetchedSentences = await getSentencesContaining(word);
      console.log("Fetched sentences for", word, fetchedSentences);
      setSentences(fetchedSentences);
    } catch (err) {
      console.error("Error fetching sentences:", err);
      setSentences([]);
    } finally {
      setLoading(false);
    }
  };

  // Synonyms tab handler
 const handleKeywordClickSynonyms = async (word) => {
  setSelectedWord(word);
  setLoadingSynonyms(true);
  setSynonyms([]); // Clear previous synonyms
  setAnalysis(null); // You'll need to add this state variable

  try {
    const response = await fetch("http://localhost:8000/api/get-synonyms/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ word }),
    });

    const data = await response.json();
    console.log("Response data:", data);
    
    // The new API returns 'analysis' instead of 'synonyms'
    if (data.analysis) {
      setAnalysis(data.analysis);
    } else if (data.synonyms) {
      // Fallback for the old format
      setSynonyms(data.synonyms);
    } else {
      setSynonyms([]);
      setAnalysis(null);
    }
  } catch (err) {
    console.error("Error fetching synonyms:", err);
    setSynonyms([]);
    setAnalysis("Could not fetch synonyms.");
  } finally {
    setLoadingSynonyms(false);
  }
};

const highlightWord = (sentence, word) => {
  if (!word) return sentence;

  const regex = new RegExp(`\\b(${word.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')})\\b`, "gi");

  const parts = [];
  let lastIndex = 0;

  sentence.replace(regex, (match, p1, offset) => {
    if (offset > lastIndex) {
      parts.push(sentence.slice(lastIndex, offset));
    }
    parts.push(<mark key={offset}>{match}</mark>);
    lastIndex = offset + match.length;
  });

  if (lastIndex < sentence.length) {
    parts.push(sentence.slice(lastIndex));
  }

  return parts;
};

const SentenceModal = ({ word, sentences, onClose, loading }) => {
  return (
    <Modal title={`Sentences containing "${word}"`} onClose={onClose}>
      {loading ? (
        <p>Loading sentences...</p>
      ) : sentences.length > 0 ? (
        <ul>
          {sentences.map((s, idx) => (
            <li key={idx}>{highlightWord(s, word)}</li>
          ))}
        </ul>
      ) : (
        <p>No sentences found.</p>
      )}
    </Modal>
  );
};

  const closeModal = () => {
    setSelectedWord(null);
    setSentences([]);
    setSynonyms([]);
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

  const SynonymModal = ({ word, synonyms, analysis, onClose, loading }) => {
  return (
    <Modal title={`Synonyms for "${word}"`} onClose={onClose}>
      {loading ? (
        <div style={{ textAlign: 'center', padding: '40px 20px' }}>
          <div style={{ marginBottom: '20px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              border: '4px solid #f3f3f3',
              borderTop: '4px solid #3498db',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto'
            }}></div>
          </div>
          <p style={{ margin: '10px 0', fontSize: '16px' }}>
            Analyzing synonyms for "{word}"...
          </p>
          <div style={{
            width: '100%',
            height: '6px',
            backgroundColor: '#f0f0f0',
            borderRadius: '3px',
            overflow: 'hidden',
            margin: '20px 0'
          }}>
            <div style={{
              width: '100%',
              height: '100%',
              background: 'linear-gradient(90deg, #3498db, #2ecc71, #3498db)',
              backgroundSize: '200% 100%',
              animation: 'progressMove 2s ease-in-out infinite'
            }}></div>
          </div>
          <p style={{ 
            fontSize: '14px', 
            color: '#666',
            margin: '10px 0'
          }}>
            Your synonyms are being generated. Please don't go anywhere...
          </p>
        </div>
      ) : analysis ? (
        <div 
          style={{ 
            whiteSpace: 'pre-wrap',
            lineHeight: '1.6',
            fontSize: '14px',
            maxHeight: '70vh',
            overflowY: 'auto',
            padding: '15px',
            fontFamily: 'inherit'
          }}
        >
          {analysis}
        </div>
      ) : synonyms && synonyms.length > 0 ? (
        <ul>
          {synonyms.map((syn, idx) => (
            <li key={idx}>
              {typeof syn === 'string' ? syn : syn.synonym}
            </li>
          ))}
        </ul>
      ) : (
        <p>No synonyms found.</p>
      )}
      
      {/* Add CSS for animations */}
      <style jsx>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @keyframes progressMove {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </Modal>
  );
};

  const chartData = results?.slice(0, 20).map((r) => ({
    label: r.word,
    value: r.keyness ?? r.log_likelihood ?? r.chi2 ?? r.tfidf_score ?? 0,
  }));

  const viewLabels = {
    keywords: "Top Keywords",
    charts: "Charts",
    table: "Table",
    wordData: "Word Data",
    summary: "Summary",
    alternateWords: "Alternate Words",
    overusedWords: "Overused Words",
    concepts: "Concepts",
  };

  return (
    <div className="results-container">
      <ResultsSummary stats={stats} selectedMethod={method} comparisonResults={safeResults} genre={genre} />

      {/* View Toggle Buttons */}
      <div className="view-controls">
        {Object.keys(viewLabels).map((view) => (
          <button
            key={view}
            className={`btn ${activeView === view ? "bg-blue-500 text-white" : ""}`}
            onClick={() => {
              setActiveView(view);
              setSelectedWord(null); // reset selection on tab switch
              setSentences([]);
              setSynonyms([]);
            }}
          >
            {viewLabels[view]}
          </button>
        ))}

        {/* Download button inline */}
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
              .split(/\n{2,}|(?<=\.)\s+/)
              .map((p, i) => (
                <p key={i}>{p.trim()}</p>
              ))
          )}
        </div>
      )}

      {/* Keywords View (Sentences) */}
      {activeView === "keywords" && (
        <>
          <div className="creative-results">
            {/* Section Heading */}
            <div className="keywords-header">
              <h2>Top 50 Most Significant Keywords from Your Text</h2>
              <p>Click on a word to display the sentences from your text that contain that keyword.</p>
            </div>
            {/* POS Sections */}
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
                        className={`keyword keyword-pill ${posColors[pos] || posColors.OTHER}`}
                        onClick={() => handleKeywordClickSentences(w.word)}
                        tabIndex={0}
                        role="button"
                        onKeyPress={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            handleKeywordClickSentences(w.word);
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
          {/* Only show SentenceModal in keywords tab */}
          {selectedWord && sentences.length > 0 && (
  <SentenceModal
    word={selectedWord}
    sentences={sentences}
    loading={loading}
    onClose={closeModal}
  />
)}
        </>
      )}

      {/* Alternate Words View (Synonyms) */}
      {activeView === "alternateWords" && (
        <div className="creative-results">
          <div className="keywords-header">
            <h2>Top 50 Most Significant Keywords from Your Text</h2>
            <p>Click on a word to display synonyms for that keyword.</p>
          </div>
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
                      className={`keyword keyword-pill ${posColors[pos] || posColors.OTHER}`}
                      onClick={() => handleKeywordClickSynonyms(w.word)}
                      tabIndex={0}
                      role="button"
                      onKeyPress={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          handleKeywordClickSynonyms(w.word);
                        }
                      }}
                      title={`Click to see synonyms for "${w.word}"`}
                    >
                      {w.word}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
          {/* Only show synonym modal in alternateWords tab */}
{selectedWord && (
  <SynonymModal
    word={selectedWord}
    synonyms={synonyms}
    analysis={analysis}
    loading={loadingSynonyms}
    onClose={closeModal}
  />
)}
        </div>
      )}

      {/* Charts View */}
      {activeView === "charts" && <Charts results={safeResults} method={method} />}

      {/* Table View */}
      {activeView === "table" && <ResultsTable results={safeResults} method={method} />}

      {/* Word Data View */}
      {activeView === "wordData" && <KeynessResultsGrid results={safeResults} method={method} />}

      {/* Overused Words and Concepts Views can be implemented similarly */}
    </div>
  );
};

export default CreativeKeynessResults;