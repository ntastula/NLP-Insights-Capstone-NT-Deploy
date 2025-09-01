// src/Components/TextInputSection.js
import React from "react";
import { Upload, FileText } from "lucide-react";
import KeynessAnalyser from "./Keyness/KeynessAnalyser";

const TextInputSection = ({
  pastedText,
  handleTextPaste,
  file,
  handleFileUpload,
  activeInput,
  uploadedPreview,
  corpusPreview,
  error
}) => {
  return (
    <>
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
        <label className="block text-gray-700 font-bold text-lg mb-4">
          📁 Or upload a document:
        </label>
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

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-100 border border-red-300 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      {/* Preview */}
      {uploadedPreview && (
        <div className="mb-6">
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
              <h3 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-500" />
                Your {activeInput === 'file' ? 'Uploaded' : 'Entered'} Text (First 4 lines)
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
        </div>
      )}
    </>
  );
};

export default TextInputSection;
