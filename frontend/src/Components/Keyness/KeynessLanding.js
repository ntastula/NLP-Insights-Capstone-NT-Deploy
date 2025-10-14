import React, { useState, useEffect } from "react";
import TextInputSection from "../TextInputSection";
import KeynessAnalyser from "./KeynessAnalyser";
import "./KeynessLanding.css";

const KeynessLanding = ({
    onBack,
    genre,
    onWordDetail,
    onResults,
    comparisonMode = "corpus"
}) => {
    const [pastedText, setPastedText] = useState("");
    const [uploadedText, setUploadedText] = useState("");
    const [uploadedPreview, setUploadedPreview] = useState("");
    const [activeInput, setActiveInput] = useState("");
    const [error, setError] = useState("");
    const [analysisStarted, setAnalysisStarted] = useState(false);
    const [corpusPreview, setCorpusPreview] = useState("");
    const [pastedWordCount, setPastedWordCount] = useState(0);
    const [referenceText, setReferenceText] = useState("");
    const [referencePreview, setReferencePreview] = useState("");
    const [selectedFiles, setSelectedFiles] = useState([]);

    // Fetch corpus preview or user text preview based on comparison mode
    useEffect(() => {
        let cancelled = false;

        async function fetchPreview() {
            try {
                if (comparisonMode === "corpus") {
                    if (!genre) return;

                    const backendURL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

                    const url = `${backendURL}/api/corpus-preview-keyness/?name=${encodeURIComponent(genre)}`;

                    const response = await fetch(url, { credentials: "include" });

                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    const data = await response.json();
                    if (!cancelled) {
                        setCorpusPreview(data.preview || "");
                    }
                } else if (comparisonMode === "user_text") {
                    setCorpusPreview("");
                }
            } catch (err) {
                if (!cancelled) {
                    console.error("Error fetching preview:", err);
                    setCorpusPreview("");
                }
            }
        }

        fetchPreview();
        return () => { cancelled = true; };
    }, [genre, comparisonMode]);

    const handleTextPaste = (e) => {
        const text = e.target.value || "";
        setPastedText(text);
        setUploadedText(text);
        setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
        setActiveInput("text");

        const words = text.trim().split(/\s+/).filter(Boolean);
        setPastedWordCount(words.length);
    };

    const handleFilesUploaded = (combinedText, files) => {
        const text = combinedText || "";

        if (comparisonMode === "user_text") {
            if (!files || files.length === 0) {
                setError("Please upload at least one file.");
                return;
            }

            // Accumulate files one by one
            setSelectedFiles(prevFiles => {
                let updatedFiles = [...prevFiles, ...files];

                // Keep only first 2 files
                if (updatedFiles.length > 2) {
                    updatedFiles = updatedFiles.slice(0, 2);
                }

                // Combine the text of both files
                const combinedText = updatedFiles
                    .map(f => f.textContent || "")
                    .join("\n\n--- Next File ---\n\n");

                setUploadedText(combinedText);
                setUploadedPreview(
                    updatedFiles[1]
                        ? (updatedFiles[1].textContent || "").split("\n").slice(0, 4).join("\n")
                        : ""
                );

                // Set reference preview from first file
                if (updatedFiles[0]) {
                    const refFile = updatedFiles[0];
                    setReferenceText(refFile.textContent || "");
                    setReferencePreview(
                        (refFile.textContent || "").split("\n").slice(0, 4).join("\n")
                    );
                    setCorpusPreview(
                        (refFile.textContent || "").split("\n").slice(0, 4).join("\n")
                    );
                }
                setError(updatedFiles.length === 2 ? "" : "Please upload both reference and target texts.");
                return updatedFiles;
            });

        } else {
            setSelectedFiles(files || []);
            setUploadedText(text);
            setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
            setError("");
        }

        setActiveInput("file");
    };

    const handleContinue = () => {
        if (!uploadedText.trim()) {
            setError("Please enter or upload some text before continuing.");
            return;
        }

        if (comparisonMode === "user_text" && !referenceText.trim()) {
            setError("Please upload both reference and target texts.");
            return;
        }

        setAnalysisStarted(true);
    };

    if (analysisStarted) {
        console.log("KeynessLanding passing to KeynessAnalyser:", {
            genre,
            comparisonMode,
            referenceText: referenceText?.substring(0, 50) + "..."
        });
        return (
            <KeynessAnalyser
                uploadedText={uploadedText}
                uploadedPreview={uploadedPreview}
                corpusPreview={corpusPreview}
                onBack={() => setAnalysisStarted(false)}
                genre={genre}
                onWordDetail={onWordDetail}
                onResults={onResults}
                comparisonMode={comparisonMode}
                referenceText={referenceText}
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
                    {comparisonMode === "user_text"
                        ? "Compare two texts to find distinctive words and phrases."
                        : "Find the words that stand out most in your writing, showing what makes your voice and style different from other texts."
                    }
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
                        comparisonMode={comparisonMode}
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
                            disabled={
                                comparisonMode === "user_text"
                                    ? selectedFiles.length !== 2
                                    : !uploadedText.trim()
                            }
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