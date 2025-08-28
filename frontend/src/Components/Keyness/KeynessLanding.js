import React, { useState } from "react";
import TextInputSection from "../TextInputSection";
import KeynessAnalyser from "./KeynessAnalyser";

const KeynessLanding = ({ onBack }) => {
  const [pastedText, setPastedText] = useState("");
  const [file, setFile] = useState(null);
  const [uploadedText, setUploadedText] = useState("");
  const [uploadedPreview, setUploadedPreview] = useState("");
  const [activeInput, setActiveInput] = useState("");
  const [error, setError] = useState("");
  const [analysisStarted, setAnalysisStarted] = useState(false);

  const handleTextPaste = (e) => {
    const text = e.target.value;
    setPastedText(text);
    setUploadedText(text);
    setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
    setActiveInput("text");
  };

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setActiveInput("file");

    try {
      const text = await uploadedFile.text();
      setUploadedText(text);
      setUploadedPreview(text.split("\n").slice(0, 4).join("\n"));
    } catch (err) {
      setError("Failed to read the uploaded file.");
    }
  };

  if (analysisStarted) {
    return (
      <KeynessAnalyser
        uploadedText={uploadedText}
        uploadedPreview={uploadedPreview}
        corpusPreview={null}
        onBack={() => setAnalysisStarted(false)}
      />
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <button
        onClick={onBack}
        className="mb-6 bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded shadow"
      >
        ← Back
      </button>

      <h1 className="text-3xl font-bold mb-6">Keyness Analysis</h1>
      <TextInputSection
        pastedText={pastedText}
        handleTextPaste={handleTextPaste}
        file={file}
        handleFileUpload={handleFileUpload}
        activeInput={activeInput}
        uploadedPreview={uploadedPreview}
        corpusPreview={null}
        error={error}
      />

      <div className="text-center">
        <button
          onClick={() => {
            if (!uploadedText.trim()) {
              setError("Please enter or upload some text before continuing.");
              return;
            }
            setAnalysisStarted(true);
          }}
          className="bg-gradient-to-r from-purple-600 to-blue-600 text-white px-8 py-3 rounded-lg shadow-lg hover:from-purple-700 hover:to-blue-700 transform hover:-translate-y-1 transition-all"
        >
          Continue to Analysis →
        </button>
      </div>
    </div>
  );
};

export default KeynessLanding;
