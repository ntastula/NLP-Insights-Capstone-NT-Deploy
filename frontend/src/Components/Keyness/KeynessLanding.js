// frontend/src/Components/Keyness/KeynessLanding.js
import React, { useState, useEffect } from "react";
import TextInputSection from "../TextInputSection";
import KeynessAnalyser from "./KeynessAnalyser";
import "./KeynessLanding.css";

const KeynessLanding = ({ onBack, genre }) => {
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
                const url = genre
                    ? `http://localhost:8000/api/corpus-preview/?name=${encodeURIComponent(genre)}`
                    : "http://localhost:8000/api/corpus-preview/";
                const response = await fetch(url, { credentials: "include" });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                if (!cancelled) {
                    const preview = (data.preview || "").split("\n").slice(0, 4).join("\n");
                    setCorpusPreview(preview);
                }
            } catch (err) {
                if (!cancelled) {
                    console.error("Preview fetch failed:", err);
                    setCorpusPreview("");
                }
            }
        }

        fetchCorpusPreview();
        return () => {
            cancelled = true;
        };
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
    // ===== END TextInputSection block =====

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
                genre={genre} // pass selected genre through to analyser
            />
        );
    }

    return (
        <div>
            <button
                onClick={onBack}
                className="mb-6 bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded shadow"
            >
                ← Back
            </button>

            <h1 className="text-3xl font-bold mb-6">Keyness Analysis</h1>

            <div className="keyness-container">
                {/* TextInputSection block (PRESERVED) */}
                <TextInputSection
                    pastedText={pastedText}
                    handleTextPaste={handleTextPaste}
                    pastedWordCount={pastedWordCount}
                    uploadedPreview={uploadedPreview}
                    corpusPreview={corpusPreview}
                    error={error}
                    onFilesUploaded={handleFilesUploaded}
                />

                <div className="text-center">
                    <button
                        onClick={handleContinue}
                        className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-8 py-3 rounded-lg shadow-lg hover:from-purple-700 hover:to-blue-700 transform hover:-translate-y-1 transition-all"
                    >
                        Continue to Analysis →
                    </button>
                </div>
            </div>
        </div>
    );
};

export default KeynessLanding;
