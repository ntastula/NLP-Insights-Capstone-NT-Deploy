import React, { useState, useEffect } from "react";
import ResultsSummary from "./ResultsSummary";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import CreativeKeynessResults from "./CreativeKeynessResults";
import '../ProgressBar.css';
import './KeynessAnalyser.css';

console.log("KeynessAnalyser file loaded");

/**
 * Progress Bar
 */
const ProgressBar = ({ loading }) => {
    const [progress, setProgress] = useState(0);

    useEffect(() => {
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

const KeynessAnalyser = ({
    uploadedText,
    uploadedPreview,
    corpusPreview,
    method,
    comparisonMode = "corpus",
    referenceText,
    onBack,
    genre,
    onWordDetail,
    onResults
}) => {
    const [comparisonResults, setComparisonResults] = useState([]);
    const [stats, setStats] = useState({ uploadedTotal: 0, corpusTotal: 0, totalSignificant: 0 });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [analysisDone, setAnalysisDone] = useState(false);
    const [selectedMethod, setSelectedMethod] = useState(method || "nltk");
    const [filterMode, setFilterMode] = useState("content");
    const [showLibraryOptions, setShowLibraryOptions] = useState(true);
    const [showResults, setShowResults] = useState(false);

    const handleChangeMethod = () => {
        console.log("handleChangeMethod called!");
        setAnalysisDone(false);
        setSelectedMethod("");
        setShowLibraryOptions(true);
        setComparisonResults([]);
        setShowResults(false);
    };

    console.log("KeynessAnalyser component rendered");
    console.log("handleChangeMethod exists:", typeof handleChangeMethod);
    console.log("handleChangeMethod defined:", handleChangeMethod);

    useEffect(() => {
        window.scrollTo(0, 0);
    }, []);

    // Library descriptions and configurations
    const libraries = [
        {
            id: "nltk",
            name: "NLTK",
            title: "Get the complete statistical picture of your word choices",
            description: "Choose NLTK when you want a comprehensive analysis that shows not just which words make your writing distinctive, but how strongly they stand out and how significant that difference really is. This gives you the most thorough understanding of your unique voice by combining multiple statistical measures to paint a complete picture of your writing style."
        },
        {
            id: "sklearn",
            name: "Scikit-Learn",
            title: "Find your most statistically significant word patterns",
            description: "Choose Scikit-learn when you want to discover which words in your writing are genuinely meaningful patterns versus just random occurrences. This analysis focuses on statistical confidence, helping you identify the word choices that truly define your writing style rather than words that might just appear unusual by chance."
        },
        {
            id: "gensim",
            name: "Gensim",
            title: "Discover your most important and distinctive terms",
            description: "Choose Gensim when you want to find the words that carry the most weight and meaning in your writing. This analysis identifies terms that are both frequent in your work AND rare elsewhere, revealing the vocabulary that makes your writing uniquely valuable and memorable to readers."
        },
        {
            id: "spacy",
            name: "spaCy",
            title: "Get the most complete analysis with positive and negative patterns",
            description: "Choose spaCy when you want the fullest possible analysis that not only shows your distinctive words but also reveals what you avoid using compared to other writers. This comprehensive approach combines statistical significance with effect sizes and can highlight both your signature word choices and notable gaps in your vocabulary."
        }
    ];

    const performAnalysis = async (methodName) => {
        if (!uploadedText) {
            setError("No text to analyse");
            return;
        }
        if (comparisonMode === "user_text") {
            if (!referenceText) {
                setError("No reference text available for comparison");
                console.error("Missing reference text in user_text mode");
                return;
            }
        } else {
            if (!genre) {
                setError("No genre/corpus selected");
                console.error("Missing genre in corpus mode");
                return;
            }
        }
        setLoading(true);
        setError("");
        setAnalysisDone(false);
        setSelectedMethod(methodName);
        setShowLibraryOptions(false);

        try {
            const payload = {
                comparison_mode: comparisonMode,
                uploaded_text: uploadedText,   
                reference_text: comparisonMode === "user_text" ? referenceText : undefined,
                method: methodName.toLowerCase(),
                filter_mode: filterMode,
                corpus_name: comparisonMode === "corpus" ? genre : undefined
            };

            if (comparisonMode === "user_text") {
                payload.comparison_mode = "user_text";
                payload.reference_text = referenceText;
                console.log("Target text type:", typeof uploadedText);
                console.log("Reference text type:", typeof referenceText);
                console.log("User text comparison:", {
                    method: methodName,
                    targetLength: uploadedText.length,
                    referenceLength: referenceText.length
                });
            } else {
                payload.comparison_mode = "corpus";
                payload.corpus_name = genre;
                console.log("Corpus comparison:", {
                    method: methodName,
                    genre: genre
                });
            }

            const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/analyse-keyness/`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const json = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(json?.error || `HTTP ${res.status}`);
            }

            const resultsArray = Array.isArray(json.results) ? json.results : [];
            setComparisonResults(resultsArray);

            setStats({
                uploadedTotal: json.uploaded_total ?? uploadedText.split(/\s+/).filter(Boolean).length,
                corpusTotal: json.corpus_total ?? 0,
                totalSignificant: json.total_significant ?? (resultsArray ? resultsArray.length : 0)
            });

            setAnalysisDone(true);
            if (onResults) {
                onResults({
                    results: resultsArray,
                    method: methodName,
                    comparisonMode: comparisonMode,
                    uploadedText: uploadedText,
                    referenceText: comparisonMode === "user_text" ? referenceText : undefined,
                    stats: {
                        uploadedTotal: json.uploaded_total ?? uploadedText.split(/\s+/).filter(Boolean).length,
                        corpusTotal: json.corpus_total ?? 0,
                        totalSignificant: json.total_significant ?? (resultsArray ? resultsArray.length : 0)
                    }

                });
            }
            console.log("Analysis completed successfully:", {
                mode: comparisonMode,
                resultsCount: resultsArray.length
            });
        } catch (e) {
            setError(e.message || "Analysis failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="mb-6">
            <button
                onClick={onBack}
                className="keyness-back-button"
            >
                ← Back
            </button>

            {/* Word Filtering Options */}
            <div className="filter-section">
                <p className="filter-title">
                    What words in your text would you like analysed:
                </p>
                <div className="filter-options">
                    <label className="filter-option">
                        <input
                            type="radio"
                            name="filterMode"
                            value="content"
                            checked={filterMode === "content"}
                            onChange={(e) => setFilterMode(e.target.value)}
                        />
                        <span>Only content words (nouns, verbs, adjectives, adverbs)</span>
                    </label>
                    <label className="filter-option">
                        <input
                            type="radio"
                            name="filterMode"
                            value="all"
                            checked={filterMode === "all"}
                            onChange={(e) => setFilterMode(e.target.value)}
                        />
                        <span>All words</span>
                    </label>
                </div>
            </div>

            {/* Library Selection Section */}
            {showLibraryOptions ? (
                <div className="library-selection">
                    <h2 className="library-selection-title">Choose Your Analysis Method</h2>
                    <div className="library-container">
                        {libraries.map((library) => (
                            <div key={library.id} className="library-card">
                                <div className="library-card-content">
                                    {/* Left side - Description */}
                                    <div className="library-description">
                                        <h3 className="library-title">
                                            {library.name}: {library.title}
                                        </h3>
                                        <p className="library-text">
                                            {library.description}
                                        </p>
                                    </div>

                                    {/* Right side - Button */}
                                    <div className="library-button-container">
                                        <button
                                            onClick={() => performAnalysis(library.id)}
                                            disabled={loading || !uploadedText}
                                            className="analysis-button"
                                        >
                                            Analyse with {library.name}
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <div className="collapsed-library-selection">
                    <div className="current-analysis-info">
                        <span className="current-analysis-text">
                            Analysing with <strong>{libraries.find(lib => lib.id === selectedMethod)?.name || selectedMethod.toUpperCase()}</strong>
                        </span>
                        <button
                            onClick={handleChangeMethod}
                            className="change-method-button"
                            disabled={loading}
                        >
                            Change Method
                        </button>
                    </div>
                </div>
            )}

            {loading && (
                <div className="progress-container-wrapper">
                    <ProgressBar loading={loading} />
                </div>
            )}

            {error && (
                <div className="error-message">
                    Error: {error}
                </div>
            )}

            {analysisDone && (
                <CreativeKeynessResults
                    results={comparisonResults}
                    uploadedText={uploadedText}
                    method={selectedMethod}
                    stats={stats}
                    genre={genre}
                    onWordDetail={onWordDetail}
                    onChangeMethod={handleChangeMethod}
                    loading={loading}
                />
            )}
        </div>
    );
};

export default KeynessAnalyser;