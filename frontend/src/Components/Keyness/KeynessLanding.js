import React, { useState, useEffect } from "react";
import TextInputSection from "../TextInputSection";
import KeynessAnalyser from "./KeynessAnalyser";
import "./KeynessLanding.css";

const KeynessLanding = ({ onBack, genre, onWordDetail, onResults }) => {
    const [pastedText, setPastedText] = useState("");
    const [uploadedText, setUploadedText] = useState("");
    const [uploadedPreview, setUploadedPreview] = useState("");
    const [activeInput, setActiveInput] = useState(""); // "text" | "file" | ""
    const [error, setError] = useState("");
    const [analysisStarted, setAnalysisStarted] = useState(false);
    const [corpusPreview, setCorpusPreview] = useState("");
    const [pastedWordCount, setPastedWordCount] = useState(0);

    useEffect(() => {
        let cancelled = false;

        async function fetchCorpusPreview() {
            try {
                if (!genre) return;
                
                const url = genre
                    ? `http://localhost:8000/api/corpus-preview-keyness/?name=${encodeURIComponent(genre)}`
                    : "http://localhost:8000/api/corpus-preview-keyness/";

                const response = await fetch(url, { credentials: "include" });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                if (!cancelled) {
                    setCorpusPreview(data.preview || "");
                }
            } catch (err) {
                if (!cancelled) setCorpusPreview("");
            }
        }

        fetchCorpusPreview();
        return () => { cancelled = true; };
    }, [genre]);

    const handleTextPaste = (e) => {
        const text = e.target.value || "";
        setPastedText(text);
        setUploadedText(text);
        setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
        setActiveInput("text");

        const words = text.trim().split(/\s+/).filter(Boolean);
        setPastedWordCount(words.length);
    };

    const handleFilesUploaded = (combinedText /*, files */) => {
        const text = combinedText || "";
        setUploadedText(text);
        setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
        setActiveInput("file");
        setError("");
    };

    const handleContinue = () => {
        if (!uploadedText.trim()) {
            setError("Please enter or upload some text before continuing.");
            return;
        }
        setAnalysisStarted(true);
    };

    if (analysisStarted) {
        return (
            <KeynessAnalyser
                uploadedText={uploadedText}
                uploadedPreview={uploadedPreview}
                corpusPreview={corpusPreview}
                onBack={() => setAnalysisStarted(false)}
                genre={genre} 
                onWordDetail={onWordDetail}
                onResults={onResults}
            />
        );
    }

    return (
        <div className="keyness-landing-wrapper">
            <button
                onClick={onBack}
                className="keyness-back-button"
            >
                ← Back
            </button>

            <div className="keyness-header">
                <h1 className="keyness-title">Keyness Analysis</h1>
                <p className="keyness-subtitle">
                    Find the words that stand out most in your writing, showing what makes your voice and style different from other texts.
                </p>
            </div>

            <div className="keyness-container">
                <div className="keyness-content-card">
                    <TextInputSection
                        pastedText={pastedText}
                        handleTextPaste={handleTextPaste}
                        pastedWordCount={pastedWordCount}
                        uploadedPreview={uploadedPreview}
                        corpusPreview={corpusPreview}
                        error={error}
                        onFilesUploaded={handleFilesUploaded}
                    />

                    {error && (
                        <div className="keyness-error-message">
                            {error}
                        </div>
                    )}

                    <div className="keyness-continue-section">
                        <button
                            onClick={handleContinue}
                            className="keyness-continue-button"
                            disabled={!uploadedText.trim()}
                        >
                            Continue to Analysis →
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default KeynessLanding;
