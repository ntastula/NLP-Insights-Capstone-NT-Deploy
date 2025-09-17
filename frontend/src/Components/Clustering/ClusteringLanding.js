import React, { useState } from "react";
import TextInputSection from "../TextInputSection";
import ClusteringAnalyser from "./ClusteringAnalyser";
// import "./ClusteringLanding.css";

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
    <div>
      <button
        onClick={onBack}
        className="mb-6 bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded shadow"
      >
        ← Back
      </button>

      <h1 className="text-3xl font-bold mb-6">Clustering Analysis</h1>

      <div className="clustering-container">
        <TextInputSection
          pastedText={pastedText}
          handleTextPaste={handleTextPaste}
          pastedWordCount={pastedWordCount}
          uploadedPreview={uploadedPreview}
          error={error}
          onFilesUploaded={handleFilesUploaded}
        />

        <div className="text-center mb-12">
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

export default ClusteringLanding;


