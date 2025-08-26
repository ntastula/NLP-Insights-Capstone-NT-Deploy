import React, { useState, useEffect } from "react";
import { Upload, FileText } from "lucide-react";
import KeynessAnalyser from "./KeynessAnalyser"; 
import KeynessComparison from "./KeynessComparison";

const HomePage = () => {
  const [file, setFile] = useState(null);
  const [uploadedText, setUploadedText] = useState("");
  const [pastedText, setPastedText] = useState("");
  const [uploadedPreview, setUploadedPreview] = useState("");
  const [activeInput, setActiveInput] = useState("");
  const [error, setError] = useState("");
  const [corpusPreview, setCorpusPreview] = useState("");

  useEffect(() => {
    const fetchCorpusPreview = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/corpus-preview/");
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setCorpusPreview(data.preview);
      } catch (err) {
        console.error(err);
        setError("Failed to load reference corpus preview.");
      }
    };
    fetchCorpusPreview();
  }, []);

  const updatePreview = (text, inputType) => {
    const lines = text.split("\n").filter((line) => line.trim());
    setUploadedPreview(lines.slice(0, 4).join("\n"));
    setUploadedText(text);
    setActiveInput(inputType);
  };

  const handleTextPaste = (e) => {
    const text = e.target.value;
    setPastedText(text);
    if (text.trim()) {
      updatePreview(text, "text");
      setFile(null);
      setError("");
    } else if (activeInput === "text") {
      setUploadedText("");
      setUploadedPreview("");
      setActiveInput("");
    }
  };

  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setPastedText("");
    setError("");

    try {
      if (selectedFile.type === "text/plain") {
        const text = await selectedFile.text();
        updatePreview(text, "file");
      } else if (selectedFile.name.endsWith(".doc") || selectedFile.name.endsWith(".docx")) {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch("http://localhost:8000/api/parse-document/", {
          method: "POST",
          body: formData,
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        updatePreview(data.text, "file");
      } else {
        throw new Error("Unsupported file format. Please upload .txt, .doc, or .docx files.");
      }
    } catch (err) {
      console.error(err);
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 via-purple-600 to-purple-800 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white/95 backdrop-blur-lg rounded-2xl shadow-2xl p-8">

          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-3 mb-4">
              <FileText className="w-10 h-10 text-purple-600" />
              <h1 className="text-4xl font-bold text-gray-800">Keyness Statistics Analyser</h1>
            </div>
            <p className="text-gray-600">Upload your text and analyse key vocabulary differences</p>
          </div>

          {/* Text Input */}
          <div className="bg-gray-50 rounded-xl p-6 mb-6 border-2 border-gray-300 focus-within:border-purple-400 transition-colors">
            <label className="block text-gray-700 font-bold text-lg mb-4">
              📝 Enter your text for analysis:
            </label>
            <textarea
              value={pastedText}
              onChange={handleTextPaste}
              placeholder="Type or paste your text content here for keyness analysis..."
              className="w-full h-48 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-vertical font-mono text-sm leading-relaxed"
            />
            {pastedText && (
              <div className="flex justify-between items-center mt-3 text-sm text-gray-600">
                <span>Word count: {pastedText.split(/\s+/).filter(word => word.length > 0).length}</span>
                <span className="text-green-600 font-medium">✓ Text ready for analysis</span>
              </div>
            )}
          </div>

          {/* File Upload */}
          <div className="bg-gray-50 rounded-xl p-6 mb-6 border-2 border-dashed border-gray-300 hover:border-purple-400 transition-colors">
            <label className="block text-gray-700 font-bold text-lg mb-4">📁 Or upload a document:</label>
            <div className="flex items-center justify-center">
              <label className="flex items-center gap-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white px-6 py-3 rounded-lg cursor-pointer hover:from-purple-700 hover:to-blue-700 transform hover:-translate-y-1 transition-all shadow-lg">
                <Upload className="w-5 h-5" />
                Choose File (.txt, .doc, .docx)
                <input type="file" className="hidden" accept=".txt,.doc,.docx" onChange={handleFileUpload} />
              </label>
            </div>
            {file && (
              <div className="text-center mt-4">
                <div className="inline-flex items-center gap-2 bg-green-100 text-green-800 px-4 py-2 rounded-lg">
                  <FileText className="w-5 h-5" />
                  {file.name}
                  <span className="ml-2 text-green-600 font-medium">✓ File ready for analysis</span>
                </div>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="mb-6 p-4 bg-red-100 border border-red-300 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {/* Previews */}
          {uploadedPreview && (
            <div className="mb-6 grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
                <h3 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-500" />
                  Your {activeInput === "file" ? "Uploaded" : "Entered"} Text (First 4 lines)
                </h3>
                <pre className="text-sm text-gray-600 whitespace-pre-wrap font-mono bg-gray-50 p-4 rounded-lg">
                  {uploadedPreview}
                </pre>
              </div>

              <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
                <h3 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
                  <FileText className="w-5 h-5 text-green-500" />
                  Sample Corpus (First 4 lines)
                </h3>
                <pre className="text-sm text-gray-600 whitespace-pre-wrap font-mono bg-gray-50 p-4 rounded-lg">
                  {corpusPreview}
                </pre>
              </div>
            </div>
          )}

          {/* Keyness Analysis Component */}
          <KeynessAnalyser
            uploadedText={uploadedText}
            uploadedPreview={uploadedPreview}
            corpusPreview={corpusPreview}
            method="NLTK"
          />


        </div>
      </div>
    </div>
  );
};

export default HomePage;
