import React, { useState, useMemo } from "react";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import ResultsSummary from "./ResultsSummary";
import SentenceModal from "./SentenceModal";
import KeynessResultsGrid from "./KeynessResultsGrid"
import "./CreativeKeynessResults.css";

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

  // Ensure results is always an array
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
    return <p>No significant keywords found.</p>;
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6">
      <ResultsSummary stats={stats} selectedMethod={method} comparisonResults={safeResults} />

      {/* View Toggle Buttons */}
      <div className="flex gap-4 mb-6 justify-center">
        {["keywords", "charts", "table", "wordData"].map((view) => (
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
              : "Word Data"}
          </button>
        ))}
      </div>

      {/* Keywords View */}
{activeView === "keywords" && (
  <div className="creative-results">
    {Object.entries(posGroups).map(([pos, words]) => {
      // Map POS codes to full names
      const posFullNames = {
        ADV: "Adverb",
        NOUN: "Noun",
        VERB: "Verb",
        ADJ: "Adjective",
        OTHER: "Other",
      };
      const posLabel = posFullNames[pos] || pos;

      return (
        <div key={pos} className="pos-section">
          <h3>{posLabel}</h3>
          <div
            className="word-list"
            style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}
          >
            {words.map((w, idx) => (
              <span
                key={idx}
                className={`keyword keyword-pill ${posColors[pos] || posColors.OTHER} cursor-pointer`}
                onClick={() => handleKeywordClick(w.word)}
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
        <SentenceModal word={selectedWord} sentences={sentences} onClose={closeModal} />
      )}
    </div>
  );
};

export default CreativeKeynessResults;
