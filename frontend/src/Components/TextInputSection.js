import React, { useState, useRef, useEffect } from "react";
import { CheckCircle, AlertCircle, Upload, X } from "lucide-react";
import "./Keyness/KeynessLanding.css";

const TextInputSection = ({
  pastedText,
  handleTextPaste,
  pastedWordCount,
  uploadedPreview,
  corpusPreview,
  error,
  onFilesUploaded,
  comparisonMode = "corpus",
  referenceTextId = null,
  referenceTextName = null,
}) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [hover, setHover] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadErrors, setUploadErrors] = useState([]);
  const [uploadSuccess, setUploadSuccess] = useState([]);
  const [draggedFileName, setDraggedFileName] = useState("");
  const dropzoneRef = useRef(null);

  const uploadUserTextFiles = async (files) => {
    if (files.length !== 2) {
      setUploadErrors(["Please select exactly two files."]);
      return;
    }

    setUploading(true);
    setUploadErrors([]);
    setUploadSuccess([]);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("target_file", files[0]);
    formData.append("reference_file", files[1]);
    formData.append("comparison_mode", "user_text");

    try {
      const res = await fetch("http://localhost:8000/api/upload-files/", {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      const data = await res.json();

      if (!data.success) {
        setUploadErrors([data.error || "Upload failed"]);
        return;
      }

      const updatedFiles = [
        {
          name: data.target_file.filename,
          size: data.target_file.file_size,
          wordCount: data.target_file.word_count,
          textContent: data.target_file.text_content,
          processed: true,
        },
        {
          name: data.reference_file.filename,
          size: data.reference_file.file_size,
          wordCount: data.reference_file.word_count,
          textContent: data.reference_file.text_content,
          processed: true,
        },
      ];

      setSelectedFiles(updatedFiles);
      setUploadSuccess([
        `✓ Target: ${updatedFiles[0].name} (${updatedFiles[0].wordCount} words)`,
        `✓ Reference: ${updatedFiles[1].name} (${updatedFiles[1].wordCount} words)`,
      ]);

      onFilesUploaded &&
        onFilesUploaded(
          updatedFiles.map((f) => f.textContent).join("\n\n--- Next File ---\n\n"),
          updatedFiles
        );
    } catch (err) {
      setUploadErrors([err.message || "Network error"]);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  // Handle files (user_text or corpus)
  const handleFiles = (files) => {
    const fileArray = Array.from(files);

    if (comparisonMode === "user_text") {
      const newFiles = [...selectedFiles, ...fileArray];

      if (newFiles.length > 2) {
        setUploadErrors(["Please select exactly two files (target + reference)."]);
        return;
      }

      // Validate types & size
      const oversized = newFiles.filter((f) => f.size > 5 * 1024 * 1024);
      if (oversized.length > 0) {
        setUploadErrors([
          `Files too large (max 5MB): ${oversized.map((f) => f.name).join(", ")}`,
        ]);
        return;
      }

      const invalidTypes = newFiles.filter(
        (f) => !f.name.toLowerCase().match(/\.(txt|doc|docx)$/)
      );
      if (invalidTypes.length > 0) {
        setUploadErrors([
          `Invalid file types: ${invalidTypes.map((f) => f.name).join(", ")}`,
        ]);
        return;
      }

      setSelectedFiles(newFiles);
      setUploadErrors([]);
      setUploadSuccess([]);

      if (newFiles.length === 2) {
        uploadUserTextFiles(newFiles);
      }
      return;
    }

    // Corpus mode
    const existing = new Set(selectedFiles.map((f) => `${f.name}-${f.size}`));
    const newCorpusFiles = fileArray.filter((f) => !existing.has(`${f.name}-${f.size}`));
    if (newCorpusFiles.length === 0) return;

    setSelectedFiles([...selectedFiles, ...newCorpusFiles]);
    uploadCorpusFiles(newCorpusFiles);
  };

  const uploadCorpusFiles = async (files) => {
    setUploading(true);
    setUploadErrors([]);
    setUploadSuccess([]);
    setUploadProgress(0);

    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));

    try {
      const res = await fetch("http://localhost:8000/api/upload-files/", {
        method: "POST",
        body: formData,
        credentials: "include",
      });
      const data = await res.json();

      if (!data.success) {
        setUploadErrors([data.error || "Upload failed"]);
        return;
      }

      const uploadedFiles = data.files.map((f) => ({
        name: f.filename,
        size: f.file_size || f.size,
        wordCount: f.word_count,
        textContent: f.text_content,
        processed: true,
      }));

      setSelectedFiles(uploadedFiles);
      setUploadSuccess(
        uploadedFiles.map(
          (f, i) => `✓ ${i + 1}: ${f.name} (${f.wordCount} words)`
        )
      );

      onFilesUploaded &&
        onFilesUploaded(
          uploadedFiles.map((f) => f.textContent).join("\n\n--- Next File ---\n\n"),
          uploadedFiles
        );
    } catch (err) {
      setUploadErrors([err.message || "Network error"]);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  // Remove / clear files
  const removeFile = (index) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    setSelectedFiles(newFiles);
    if (newFiles.length === 0) {
      onFilesUploaded && onFilesUploaded("", []);
      setUploadSuccess([]);
      setUploadErrors([]);
    }
  };

  const clearAllFiles = () => {
    setSelectedFiles([]);
    setUploadSuccess([]);
    setUploadErrors([]);
    onFilesUploaded && onFilesUploaded("", []);
  };

  // Drag & drop handlers
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter((prev) => prev + 1);
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setHover(true);
      setDraggedFileName(e.dataTransfer.items[0].getAsFile()?.name || "");
    }
  };
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter((prev) => {
      const newCounter = prev - 1;
      if (newCounter === 0) {
        setHover(false);
        setDraggedFileName("");
      }
      return newCounter;
    });
  };
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "copy";
  };
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setHover(false);
    setDragCounter(0);
    setDraggedFileName("");

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
    e.target.value = "";
  };

  useEffect(() => {
    const handleGlobalDrop = (e) => {
      if (!dropzoneRef.current?.contains(e.target)) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    const handleGlobalDragOver = (e) => {
      if (!dropzoneRef.current?.contains(e.target)) {
        e.preventDefault();
      }
    };
    document.addEventListener("dragover", handleGlobalDragOver, false);
    document.addEventListener("drop", handleGlobalDrop, false);
    return () => {
      document.removeEventListener("dragover", handleGlobalDragOver, false);
      document.removeEventListener("drop", handleGlobalDrop, false);
    };
  }, []);

  const getComparisonLabel = () => {
    if (comparisonMode === "user_text") {
      return referenceTextName
        ? `Comparing against: ${referenceTextName}`
        : "Comparing against your selected text";
    }
    return null;
  };

  return (
    <div className="text-input-container">
      {/* Paste textarea */}
      <div className="paste-section">
        <label className="input-label">Paste Your Text</label>
        <textarea
          value={pastedText}
          onChange={handleTextPaste}
          className="keyness-textarea"
          placeholder="Paste your text here..."
        />
        {pastedText && <div className="word-count">Word count: {pastedWordCount}</div>}
      </div>

      {/* Drag & drop upload */}
      <div
        ref={dropzoneRef}
        className={`keyness-dropzone ${hover ? "hover" : ""} ${uploading ? "uploading" : ""
          }`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="dropzone-content">
          {uploading ? (
            <div className="upload-progress">
              <Upload className="upload-icon" size={24} />
              <div>Uploading… {uploadProgress}%</div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          ) : (
            <div className="dropzone-idle">
              <Upload className="upload-icon" size={32} />
              <div className="dropzone-text">Drag & drop files here</div>
              {hover && draggedFileName && (
                <div className="drag-feedback">
                  Release to upload: <strong>{draggedFileName}</strong>
                </div>
              )}
              <div className="file-info">
                Supported: .txt, .doc, .docx (max 5MB each). Max 5 files.
              </div>
            </div>
          )}
        </div>
        <input
          id="fileInput"
          type="file"
          multiple
          className="hidden-file-input"
          onChange={handleFileSelect}
          accept=".txt,.doc,.docx"
          disabled={uploading}
        />
      </div>

      {/* Select Files Button */}
      <div className="select-button-container">
        <button
          type="button"
          onClick={() => !uploading && document.getElementById("fileInput").click()}
          className="select-files-button"
          disabled={uploading}
        >
          Select Files from Computer
        </button>
      </div>

      {/* Selected files section */}
      {selectedFiles.length > 0 && (
        <div className="selected-files-section">
          <div className="files-header">
            <h4>Selected Files:</h4>
            <button onClick={clearAllFiles} className="clear-all-button">
              Clear All
            </button>
          </div>
          <div className="files-list">
            {selectedFiles.map((file, index) => (
              <div key={`${file.name}-${index}`} className="file-item">
                <div className="file-info-row">
                  <CheckCircle size={16} className="file-check" />
                  <span className="file-details">
                    {file.name} ({Math.round(file.size / 1024)}KB)
                    {file.wordCount && (
                      <span className="word-count-info"> • {file.wordCount} words</span>
                    )}
                  </span>
                </div>
                <button
                  onClick={() => removeFile(index)}
                  className="remove-file-button"
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Uploaded Text Preview */}
      {comparisonMode === "user_text" && selectedFiles.length > 1 && (
        <div className="preview-box">
          <h3 className="preview-title">Uploaded Text Preview:</h3>
          <div className="preview-content">
            {selectedFiles.slice(1).map((file, index) => {
              const previewText = file.textContent
                ? file.textContent.split("\n").slice(0, 4).join("\n")
                : "";
              return (
                <div key={index} className="file-preview">
                  <strong>{file.name}</strong>
                  {"\n"}
                  {previewText}
                  {index < selectedFiles.slice(1).length - 1 && "\n---\n"}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {comparisonMode === "corpus" && selectedFiles.length > 0 && (
        <div className="preview-box">
          <h3 className="preview-title">Uploaded Text Preview:</h3>
          <div className="preview-content">
            {selectedFiles.map((file, index) => {
              const previewText = file.textContent
                ? file.textContent.split("\n").slice(0, 4).join("\n")
                : "";
              return (
                <div key={index} className="file-preview">
                  <strong>{file.name}</strong>
                  {"\n"}
                  {previewText}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Corpus / Reference Preview */}
      {corpusPreview && (
        <div className="preview-box">
          <h3 className="preview-title">
            {comparisonMode === "user_text" ? "Reference Text Preview:" : "Corpus Preview:"}
          </h3>
          <div className="preview-content">{corpusPreview}</div>
        </div>
      )}

      {/* Success messages */}
      {uploadSuccess.length > 0 && (
        <div className="success-messages">
          {uploadSuccess.map((msg, i) => (
            <div key={i} className="success-item">
              <CheckCircle size={16} />
              {msg}
            </div>
          ))}
        </div>
      )}

      {/* Error messages */}
      {uploadErrors.length > 0 && (
        <div className="error-messages">
          {uploadErrors.map((err, i) => (
            <div key={i} className="error-item">
              <AlertCircle size={16} />
              {err}
            </div>
          ))}
        </div>
      )}

      {/* General error */}
      {error && (
        <div className="general-error">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

    </div>
  );
};

export default TextInputSection;