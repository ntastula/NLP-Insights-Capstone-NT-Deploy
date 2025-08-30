import React, { useState, useEffect } from "react";
import TextInputSection from "../TextInputSection";
import KeynessAnalyser from "./KeynessAnalyser";
import './KeynessLanding.css';

const KeynessLanding = ({ onBack }) => {
  const [pastedText, setPastedText] = useState("");
  const [uploadedText, setUploadedText] = useState("");
  const [uploadedPreview, setUploadedPreview] = useState("");
  const [activeInput, setActiveInput] = useState("");
  const [error, setError] = useState("");
  const [analysisStarted, setAnalysisStarted] = useState(false);
  const [corpusPreview, setCorpusPreview] = useState("");

  useEffect(() => {
    const fetchCorpusPreview = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/corpus-preview/");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setCorpusPreview(data.preview.split("\n").slice(0, 4).join("\n"));
      } catch (err) {
        console.error(err);
      }
    };
    fetchCorpusPreview();
  }, []);

  const handleTextPaste = (e) => {
    const text = e.target.value;
    setPastedText(text);
    setUploadedText(text);
    setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
    setActiveInput("text");
  };

  const handleFilesUploaded = (combinedText, files) => {
    setUploadedText(combinedText);
    setUploadedPreview(combinedText.split("\n").slice(0, 4).join("\n"));
    setActiveInput("file");
    setError(""); // clear previous errors
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

      <TextInputSection
        pastedText={pastedText}
        handleTextPaste={handleTextPaste}
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
