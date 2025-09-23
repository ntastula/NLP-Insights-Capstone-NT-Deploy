import React, { useState } from "react";
import TextInputSection from "../TextInputSection";
import ClusteringAnalyser from "./ClusteringAnalyser";
import "./ClusteringLanding.css";

const ClusteringLanding = ({ onBack }) => {
  const [pastedText, setPastedText] = useState("");
  const [uploadedText, setUploadedText] = useState("");
  const [uploadedPreview, setUploadedPreview] = useState("");
  const [activeInput, setActiveInput] = useState("");
  const [error, setError] = useState("");
  const [analysisStarted, setAnalysisStarted] = useState(false);
  const [pastedWordCount, setPastedWordCount] = useState(0);

  const handleTextPaste = (e) => {
    const text = e.target.value;
    setPastedText(text);
    setUploadedText(text);
    setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
    setActiveInput("text");

    const words = text.trim().split(/\s+/).filter(Boolean);
    setPastedWordCount(words.length);
  };

  const handleFilesUploaded = (combinedText, files) => {
    setUploadedText(combinedText);
    setUploadedPreview(combinedText.split("\n").slice(0, 4).join("\n"));
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
      <ClusteringAnalyser
        uploadedText={uploadedText}
        uploadedPreview={uploadedPreview}
        onBack={() => setAnalysisStarted(false)}
      />
    );
  }

  return (
    <div className="Clustering-landing-wrapper">
            <button
                onClick={onBack}
                className="Clustering-back-button"
            >
                ← Back
            </button>

            <div className="Clustering-header">
                <h1 className="Clustering-title">Clustering Analysis</h1>
                <p className="Clustering-subtitle">
                    See how your words naturally group together into clusters, highlighting the themes, styles, and repeated ideas that shape your writing.
                </p>
            </div>

            <div className="Clustering-container">
                <div className="Clustering-content-card">
                    <TextInputSection
                        pastedText={pastedText}
                        handleTextPaste={handleTextPaste}
                        pastedWordCount={pastedWordCount}
                        uploadedPreview={uploadedPreview}
                        error={error}
                        onFilesUploaded={handleFilesUploaded}
                    />

                    {error && (
                        <div className="Clustering-error-message">
                            {error}
                        </div>
                    )}

                    <div className="Clustering-continue-section">
                        <button
                            onClick={handleContinue}
                            className="Clustering-continue-button"
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

export default ClusteringLanding;
