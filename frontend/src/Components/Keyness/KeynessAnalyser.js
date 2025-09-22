// frontend/src/Components/Keyness/KeynessAnalyser.js
import React, { useState, useEffect } from "react";
import ResultsSummary from "./ResultsSummary";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import CreativeKeynessResults from "./CreativeKeynessResults";
import '../ProgressBar.css';

/**
 * Lightweight progress bar (unchanged visual).
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

/**
 * KeynessAnalyser:
 * - PRESERVES your structure and UI.
 * - ONLY adds passing `genre` as `corpus_name` in the POST body.
 */
const KeynessAnalyser = ({
    uploadedText,
    uploadedPreview,
    corpusPreview,
    method,      // optional external method default
    onBack,
    genre        // <-- NEW: forwarded to backend as corpus_name
}) => {
    const [comparisonResults, setComparisonResults] = useState([]);
    const [stats, setStats] = useState({ uploadedTotal: 0, corpusTotal: 0, totalSignificant: 0 });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [analysisDone, setAnalysisDone] = useState(false);
    const [selectedMethod, setSelectedMethod] = useState(method || "nltk");
    const [filterMode, setFilterMode] = useState("content"); // "content" | "all"

    const performAnalysis = async (methodName) => {
        if (!uploadedText) return;
        setLoading(true);
        setError("");
        setAnalysisDone(false);
        setSelectedMethod(methodName);

        try {
            const res = await fetch("http://localhost:8000/api/analyse-keyness/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    uploaded_text: uploadedText,
                    method: methodName.toLowerCase(), // "nltk" | "sklearn" | "gensim" | "spacy"
                    filter_mode: filterMode,          // "content" | "all"
                    corpus_name: genre || ""          // <-- genre wiring (ONLY addition)
                })
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
        } catch (e) {
            console.error("Keyness analysis failed:", e);
            setError(e.message || "Analysis failed");
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

            {/* Word Filtering Options (PRESERVED) */}
            <div className="mb-6 text-center">
                <p className="mb-2 font-medium">
                    Select an option for what words in your text you would like analysed:
                </p>
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

            {/* Analyse Buttons (PRESERVED) */}
            <div className="text-center mb-6 flex justify-center gap-4">
                <button onClick={() => performAnalysis("nltk")} disabled={loading || !uploadedText} className="btn">
                    Analyse with NLTK
                </button>
                <button onClick={() => performAnalysis("sklearn")} disabled={loading || !uploadedText} className="btn">
                    Analyse with Scikit-Learn
                </button>
                <button onClick={() => performAnalysis("gensim")} disabled={loading || !uploadedText} className="btn">
                    Analyse with Gensim
                </button>
                <button onClick={() => performAnalysis("spacy")} disabled={loading || !uploadedText} className="btn">
                    Analyse with spaCy
                </button>
            </div>

            {loading && (
                <div className="w-full max-w-xl mx-auto mt-4">
                    <ProgressBar loading={loading} />
                </div>
            )}

            {error && (
                <div className="bg-red-50 border border-red-200 rounded-2xl p-6 text-red-700 mb-6">
                    Error: {error}
                </div>
            )}

            {analysisDone && (
                <CreativeKeynessResults
                    results={comparisonResults}
                    uploadedText={uploadedText}
                    method={selectedMethod}
                    stats={stats}
                />
            )}

            {/* Keep these if your downstream UI uses them */}
            {/* <ResultsSummary data={comparisonResults} /> */}
            {/* <Charts data={comparisonResults} /> */}
            {/* <ResultsTable data={comparisonResults} /> */}
        </div>
    );
};

export default KeynessAnalyser;
