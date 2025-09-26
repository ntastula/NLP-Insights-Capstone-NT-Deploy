import React, { useState, useEffect } from "react";
import "./KeynessWordDetail.css";
import "../ProgressBar.css";

const ProgressBar = ({ loading }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let interval;
    if (loading) {
      setProgress(0);
      interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 95) return prev; 
          return prev + Math.random() * 5;
        });
      }, 150);
    } else {
      setProgress(100); 
      const timeout = setTimeout(() => setProgress(0), 500); 
      return () => clearTimeout(timeout);
    }

    return () => clearInterval(interval);
  }, [loading]);

  return (
    <div className="progress-container" style={{ width: "100%", maxWidth: "100%" }}>
      <div className="progress-fill" style={{ width: `${progress}%` }}></div>
      <div className="progress-text">{Math.round(progress)}%</div>
    </div>
  );
};

const KeynessWordDetail = ({
    word,
    wordData,
    uploadedText,
    method,
    onBack
}) => {
    const [activeTab, setActiveTab] = useState("wordData");
    const [sentences, setSentences] = useState([]);
    const [loadingSentences, setLoadingSentences] = useState(false);
    const [synonymsAnalysis, setSynonymsAnalysis] = useState("");
    const [conceptsAnalysis, setConceptsAnalysis] = useState("");
    const [loadingSynonyms, setLoadingSynonyms] = useState(false);
    const [loadingConcepts, setLoadingConcepts] = useState(false);

    const methodUpper = method?.toUpperCase() || "";
    const isSklearn = methodUpper === "SKLEARN";
    const isGensim = methodUpper === "GENSIM";
    const isSpacy = methodUpper === "SPACY";
    const isNltk = methodUpper === "NLTK";

    if (!wordData) {
        return (
            <div className="keyness-word-detail-container">
                <div className="tab-content-container">
                    <div className="tab-content">
                        <h3>Loading word details...</h3>
                        <ProgressBar loading={true} />
                    </div>
                </div>
            </div>
        );
    }

    // Fetch sentences
    const fetchSentences = async () => {
        if (sentences.length > 0) return;
        setLoadingSentences(true);
        try {
            const response = await fetch("http://localhost:8000/api/get-sentences/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ uploaded_text: uploadedText, word }),
            });
            const data = await response.json();
            setSentences(data.sentences || []);
        } catch (err) {
            console.error(err);
            setSentences([]);
        } finally {
            setLoadingSentences(false);
        }
    };

    // Fetch synonym analysis
    const fetchSynonyms = async () => {
        if (synonymsAnalysis) return;
        setLoadingSynonyms(true);
        try {
            const response = await fetch("http://localhost:8000/api/get-synonyms/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ word }),
            });
            const data = await response.json();
            setSynonymsAnalysis(data.analysis || "No analysis available");
        } catch (err) {
            console.error(err);
            setSynonymsAnalysis("Error fetching synonyms.");
        } finally {
            setLoadingSynonyms(false);
        }
    };

    // Fetch concepts analysis
    const fetchConcepts = async () => {
        if (conceptsAnalysis) return;
        setLoadingConcepts(true);
        try {
            const response = await fetch("http://localhost:8000/api/get-concepts/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ word, uploaded_text: uploadedText }), 
            });
            const data = await response.json();
            setConceptsAnalysis(data.analysis || "No analysis available");
        } catch (err) {
            console.error(err);
            setConceptsAnalysis("Error fetching concepts.");
        } finally {
            setLoadingConcepts(false);
        }
    };

    const handleTabChange = (tab) => {
        setActiveTab(tab);
        if (tab === "sentences") fetchSentences();
        if (tab === "alternateWords") fetchSynonyms();
        if (tab === "concepts") fetchConcepts();
    };

    const highlightWord = (sentence, targetWord) => {
        if (!targetWord) return sentence;
        const regex = new RegExp(`\\b(${targetWord.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')})\\b`, "gi");
        const parts = [];
        let lastIndex = 0;
        sentence.replace(regex, (match, _, offset) => {
            if (offset > lastIndex) parts.push(sentence.slice(lastIndex, offset));
            parts.push(<mark key={offset}>{match}</mark>);
            lastIndex = offset + match.length;
        });
        if (lastIndex < sentence.length) parts.push(sentence.slice(lastIndex));
        return parts;
    };

    // Determine method explanation
    const getMethodExplanation = () => {
        if (methodUpper === "SKLEARN") return {
            title: "Chi-Square Analysis Results",
            description: "These keywords show the strongest statistical association with your text using chi-square testing.",
            focus: "Look for words with high Chi² values and low p-values (< 0.05) for the most significant keywords.",
            icon: "🔬"
        };
        if (methodUpper === "GENSIM") return {
            title: "TF-IDF Analysis Results",
            description: "These keywords are identified as distinctive using Term Frequency-Inverse Document Frequency scoring.",
            focus: "Higher TF-IDF scores indicate words frequent in your text but rare in the comparison sample.",
            icon: "📈"
        };
        if (methodUpper === "SPACY") return {
            title: "Multi-Method Analysis Results",
            description: "These keywords combine multiple statistical measures to identify the most distinctive words in your text.",
            focus: "Consider words with high keyness scores and significant statistical values across multiple measures.",
            icon: "🧠"
        };
        return {
            title: "Log-Likelihood Analysis Results",
            description: "These keywords show the strongest statistical distinctiveness using log-likelihood testing.",
            focus: "Higher keyness and log-likelihood values indicate more distinctive words in your text.",
            icon: "📊"
        };
    };
    
    const methodInfo = getMethodExplanation();

    const viewLabels = {
        wordData: "📊 Word Data",
        sentences: "📝 Sentences",
        alternateWords: "🔄 Alternate Words",
        overusedWords: "⚠️ Overused Words",
        concepts: "💡 Concepts"
    };

    // Enhanced placeholder tab content
    const renderPlaceholder = (tabName) => (
        <div className="tab-content">
            <h3>{tabName}</h3>
            <div style={{
                padding: "3rem",
                textAlign: "center",
                background: "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)",
                borderRadius: "16px",
                border: "2px dashed #cbd5e1",
                margin: "2rem 0"
            }}>
                <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>🚧</div>
                <h4 style={{ color: "#64748b", marginBottom: "0.5rem" }}>Coming Soon</h4>
                <p style={{ color: "#94a3b8" }}>This functionality is currently under development and will be available in a future update.</p>
            </div>
        </div>
    );

    // Enhanced empty state for sentences
    const renderEmptyState = (type, icon = "🔍") => (
        <div style={{
            padding: "3rem",
            textAlign: "center",
            background: "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)",
            borderRadius: "16px",
            border: "2px solid #e2e8f0",
            margin: "2rem 0"
        }}>
            <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>{icon}</div>
            <h4 style={{ color: "#64748b", marginBottom: "0.5rem" }}>No {type} Found</h4>
            <p style={{ color: "#94a3b8" }}>We couldn't find any {type.toLowerCase()} for this word in the current text.</p>
        </div>
    );

    return (
        <div className="keyness-word-detail-container">
            <h1 className="page-heading">
                Keyword Analysis: "{word}"
            </h1>

            {/* Main View Toggle Buttons */}
            <div className="view-controls">
                {Object.keys(viewLabels).map((view) => (
                    <button
                        key={view}
                        className={`btn ${activeTab === view ? "bg-blue-500 text-white" : ""}`}
                        onClick={() => handleTabChange(view)}
                    >
                        {viewLabels[view]}
                    </button>
                ))}
            </div>

            <div className="tab-content-container">
                {/* Word Data */}
                {activeTab === "wordData" && (
                    <div className="results-section">
                        <h2 className="results-title">
                            {methodInfo.icon} Word Detail: {word}
                        </h2>

                        {/* Method Explanation */}
                        <div className="method-explanation">
                            <h3 className="method-title">{methodInfo.title}</h3>
                            <p className="method-description">{methodInfo.description}</p>
                            <div className="method-focus">
                                <strong>💡 What to look for:</strong> {methodInfo.focus}
                            </div>
                        </div>

                        <div className="keyword-card">
                            <div className="keyword-header">
                                <h4 className="keyword-word">{wordData.word}</h4>
                                <div className="keyword-pos">
                                    {wordData.pos || wordData.pos_tag || "Unknown POS"}
                                </div>
                            </div>

                            <div className="keyword-stats">
                                {/* --- sklearn --- */}
                                {isSklearn && (
                                    <>
                                        <div className="stat-item">
                                            <span className="stat-label">📄 Your Text:</span>
                                            <span className="stat-value">
                                                {wordData.uploaded_count ?? wordData.count_a}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">📚 Corpus:</span>
                                            <span className="stat-value">
                                                {wordData.sample_count ?? wordData.count_b}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">🔬 Chi²:</span>
                                            <span className="stat-value">
                                                {wordData.chi2?.toFixed(3)}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">📈 P-Value:</span>
                                            <span className="stat-value">
                                                {wordData.p_value?.toExponential(2)}
                                            </span>
                                        </div>
                                    </>
                                )}

                                {/* --- gensim --- */}
                                {isGensim && (
                                    <>
                                        <div className="stat-item">
                                            <span className="stat-label">📄 Your Text:</span>
                                            <span className="stat-value">
                                                {wordData.uploaded_count}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">📚 Corpus:</span>
                                            <span className="stat-value">{wordData.sample_count}</span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">📊 TF-IDF:</span>
                                            <span className="stat-value">
                                                {wordData.tfidf_score?.toFixed(3)}
                                            </span>
                                        </div>
                                    </>
                                )}

                                {/* --- spacy --- */}
                                {isSpacy && (
                                    <>
                                        <div className="stat-item">
                                            <span className="stat-label">📄 Your Text:</span>
                                            <span className="stat-value">
                                                {wordData.uploaded_count ?? wordData.count_a}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">📚 Corpus:</span>
                                            <span className="stat-value">
                                                {wordData.sample_count ?? wordData.count_b}
                                            </span>
                                        </div>
                                        {wordData.chi2 !== undefined && (
                                            <div className="stat-item">
                                                <span className="stat-label">🔬 Chi²:</span>
                                                <span className="stat-value">
                                                    {wordData.chi2?.toFixed(3)}
                                                </span>
                                            </div>
                                        )}
                                        {wordData.p_value !== undefined && (
                                            <div className="stat-item">
                                                <span className="stat-label">📈 P-Value:</span>
                                                <span className="stat-value">
                                                    {wordData.p_value?.toExponential(2)}
                                                </span>
                                            </div>
                                        )}
                                        {wordData.tfidf_score !== undefined && (
                                            <div className="stat-item">
                                                <span className="stat-label">📊 TF-IDF:</span>
                                                <span className="stat-value">
                                                    {wordData.tfidf_score?.toFixed(3)}
                                                </span>
                                            </div>
                                        )}
                                        <div className="stat-item">
                                            <span className="stat-label">📉 Log-Likelihood:</span>
                                            <span className="stat-value">
                                                {wordData.log_likelihood}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">⚡ Effect Size:</span>
                                            <span className="stat-value">{wordData.effect_size}</span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">🎯 Keyness:</span>
                                            <span className="stat-value">{wordData.keyness_score}</span>
                                        </div>
                                    </>
                                )}

                                {/* --- nltk --- */}
                                {isNltk && (
                                    <>
                                        <div className="stat-item">
                                            <span className="stat-label">📉 Log-Likelihood:</span>
                                            <span className="stat-value">
                                                {wordData.log_likelihood}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">📄 Your Text:</span>
                                            <span className="stat-value">
                                                {wordData.uploaded_count ?? wordData.count_a}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">📚 Corpus:</span>
                                            <span className="stat-value">
                                                {wordData.sample_count ?? wordData.count_b}
                                            </span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">⚡ Effect Size:</span>
                                            <span className="stat-value">{wordData.effect_size}</span>
                                        </div>
                                        <div className="stat-item">
                                            <span className="stat-label">🎯 Keyness:</span>
                                            <span className="stat-value">{wordData.keyness_score}</span>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Sentences */}
                {activeTab === "sentences" && (
                    <div className="tab-content">
                        <h3>📝 Sentences containing "{word}"</h3>
                        {loadingSentences ? (
                            <div>
                                <p style={{ textAlign: "center", color: "#64748b", marginBottom: "1rem" }}>
                                    Finding sentences containing "{word}"...
                                </p>
                                <ProgressBar loading={true} />
                            </div>
                        ) : sentences.length > 0 ? (
                            <div>
                                <p style={{ 
                                    color: "#64748b", 
                                    marginBottom: "1.5rem",
                                    padding: "1rem",
                                    background: "#f8fafc",
                                    borderRadius: "8px",
                                    border: "1px solid #e2e8f0"
                                }}>
                                    Found {sentences.length} sentence{sentences.length !== 1 ? 's' : ''} containing "<strong>{word}</strong>":
                                </p>
                                <ul className="sentences-list">
                                    {sentences.map((s, idx) => (
                                        <li key={idx}>{highlightWord(s, word)}</li>
                                    ))}
                                </ul>
                            </div>
                        ) : (
                            renderEmptyState("sentences", "📝")
                        )}
                    </div>
                )}

                {/* Alternate Words */}
                {activeTab === "alternateWords" && (
                    <div className="tab-content">
                        <h3>🔄 Alternate Words for "{word}"</h3>
                        {loadingSynonyms ? (
                            <div>
                                <p style={{ textAlign: "center", color: "#64748b", marginBottom: "1rem" }}>
                                    Analyzing alternate words and synonyms for "{word}"...
                                </p>
                                <ProgressBar loading={true} />
                            </div>
                        ) : synonymsAnalysis ? (
                            <div
                                style={{
                                    whiteSpace: "pre-wrap",
                                    lineHeight: 1.7,
                                    fontSize: "15px",
                                    maxHeight: "65vh",
                                    overflowY: "auto",
                                    padding: "2rem",
                                    fontFamily: "inherit",
                                    color: "#374151"
                                }}
                            >
                                {synonymsAnalysis}
                            </div>
                        ) : (
                            renderEmptyState("alternate words", "🔄")
                        )}
                    </div>
                )}

                {/* Overused Words */}
                {activeTab === "overusedWords" && (
                    <div className="tab-content">
                        {renderPlaceholder("⚠️ Overused Words")}
                    </div>
                )}

                {/* Concepts */}
                {activeTab === "concepts" && (
                    <div className="tab-content">
                        <h3>💡 Concepts related to "{word}"</h3>
                        {loadingConcepts ? (
                            <div>
                                <p style={{ textAlign: "center", color: "#64748b", marginBottom: "1rem" }}>
                                    Analyzing concepts and themes related to "{word}"...
                                </p>
                                <ProgressBar loading={true} />
                            </div>
                        ) : conceptsAnalysis ? (
                            <div
                                style={{
                                    whiteSpace: "pre-wrap",
                                    lineHeight: 1.7,
                                    fontSize: "15px",
                                    maxHeight: "65vh",
                                    overflowY: "auto",
                                    padding: "2rem",
                                    fontFamily: "inherit",
                                    color: "#374151"
                                }}
                            >
                                {conceptsAnalysis}
                            </div>
                        ) : (
                            renderEmptyState("concepts", "💡")
                        )}
                    </div>
                )}

                {/* Back Button */}
                <button
                    className="back-button"
                    onClick={() => {
                        console.log("Back clicked, current wordData:", wordData);
                        onBack();
                    }}
                >
                    ← Back to all keywords
                </button>
            </div>
        </div>
    );
};

export default KeynessWordDetail;