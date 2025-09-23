import React, { useState, useRef } from "react";
import { CheckCircle, AlertCircle, Upload, X } from "lucide-react";
import "./Keyness/KeynessLanding.css";

const TextInputSection = ({
  pastedText,
  handleTextPaste,
  pastedWordCount,
  uploadedPreview,
  corpusPreview,
  error,
  onFilesUploaded
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

  // Upload files to Django backend with progress
const uploadFilesToBackend = async (files) => {
  setUploading(true);
  setUploadErrors([]);
  setUploadSuccess([]);
  setUploadProgress(0);

  console.log("Files to upload:", files);

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));

  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:8000/api/upload-files/", true);
    xhr.withCredentials = true;

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        setUploadProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      try {
        resolve(JSON.parse(xhr.responseText));
      } catch (e) {
        setUploadErrors(["Invalid server response"]);
        resolve(null);
      }
    };

    xhr.onerror = () => {
      setUploadErrors(["Network error"]);
      resolve(null);
    };

    xhr.send(formData);
  }).then((result) => {
    setUploading(false);

    if (!result) return;

    if (result.success && result.files.length > 0) {
  const combinedText = result.files
    .map((file) => file.text_content)
    .join("\n\n--- Next File ---\n\n");

  onFilesUploaded && onFilesUploaded(combinedText, result.files); 

  setUploadSuccess(
    result.files.map((file) => {
      return `✓ ${file.filename} uploaded successfully (${file.word_count} words)`;
    })
  );

  setSelectedFiles(
    result.files.map((file) => ({
      name: file.filename,          
      processed: true,
      wordCount: file.word_count,
      charCount: file.char_count,
      textContent: file.text_content 
    }))
  );
}

    if (result.errors?.length > 0) setUploadErrors(result.errors);
    if (!result.success) setUploadErrors([result.error || "Upload failed"]);
    setUploadProgress(0);
  });
};

  // Handle files
  const handleFiles = (files) => {
    console.log("handleFiles called with:", files);
Array.from(files).forEach(f => console.log(f.name, f.size, f.type));

    const fileArray = Array.from(files);

    // Deduplicate by name + size
    const existing = new Set(selectedFiles.map((f) => `${f.name}-${f.size}`));
    const newFiles = fileArray.filter(
      (f) => !existing.has(`${f.name}-${f.size}`)
    );

    if (newFiles.length < fileArray.length) {
      setUploadErrors(["Some duplicate files were skipped"]);
    }

    if (newFiles.length === 0) return;

    if (newFiles.length > 5) {
      setUploadErrors(["Maximum 5 files allowed"]);
      return;
    }

    const oversizedFiles = newFiles.filter((file) => file.size > 5 * 1024 * 1024);
    if (oversizedFiles.length > 0) {
      setUploadErrors([
        `Files too large: ${oversizedFiles.map((f) => f.name).join(", ")}`,
      ]);
      return;
    }

    const invalidTypes = newFiles.filter(
      (file) => !file.name.toLowerCase().match(/\.(txt|doc|docx)$/)
    );
    if (invalidTypes.length > 0) {
      setUploadErrors([
        `Invalid file types: ${invalidTypes.map((f) => f.name).join(", ")}`,
      ]);
      return;
    }

    setUploadErrors([]);
    setUploadSuccess([]);
    uploadFilesToBackend(newFiles);
  };

  // Remove a selected file
  const removeFile = (indexToRemove) => {
    const newFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
    setSelectedFiles(newFiles);

    if (newFiles.length === 0) {
      onFilesUploaded && onFilesUploaded("", []);
      setUploadSuccess([]);
      setUploadErrors([]);
    }
  };

  // Clear all files
  const clearAllFiles = () => {
    setSelectedFiles([]);
    setUploadSuccess([]);
    setUploadErrors([]);
    onFilesUploaded && onFilesUploaded("", []);
  };

  // Drag events
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

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
  };

  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFiles(files);
    }
    e.target.value = "";
  };

  // Global drag handlers
  React.useEffect(() => {
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

  return (
    <div className="text-input-container">
      {/* Side by side input methods */}
      <div className="input-methods-row">
        {/* Paste textarea - Left side */}
        <div className="paste-section">
          <label className="input-label">
            Paste Your Text
          </label>
          <textarea
            value={pastedText}
            onChange={handleTextPaste}
            className="keyness-textarea"
            placeholder="Paste your text here..."
          />
          {pastedText && (
            <div className="word-count">
              Word count: {pastedWordCount}
            </div>
          )}
        </div>

        {/* Drag & Drop - Right side */}
        <div className="upload-section">
          <label className="input-label">
            Upload Files
          </label>
          <div
            ref={dropzoneRef}
            className={`keyness-dropzone ${hover ? "hover" : ""} ${
              uploading ? "uploading" : ""
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
                  <div className="dropzone-text">
                    Drag & drop files here
                  </div>
                  {hover && draggedFileName && (
                    <div className="drag-feedback">
                      Release to upload: <strong>{draggedFileName}</strong>
                    </div>
                  )}
                  <div className="file-info">
                    Supported: .txt, .doc, .docx (max 5MB each)
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
        </div>
      </div>

      {/* Select Files Button - Centered below inputs */}
      <div className="select-button-container">
        <button
          type="button"
          onClick={() =>
            !uploading && document.getElementById("fileInput").click()
          }
          className="select-files-button"
          disabled={uploading}
        >
          Select Files from Computer
        </button>
      </div>

      {/* Upload Success Messages */}
      {uploadSuccess.length > 0 && (
        <div className="success-messages">
          {uploadSuccess.map((message, index) => (
            <div key={index} className="success-item">
              <CheckCircle size={16} />
              {message}
            </div>
          ))}
        </div>
      )}

      {/* Upload Error Messages */}
      {uploadErrors.length > 0 && (
        <div className="error-messages">
          {uploadErrors.map((error, index) => (
            <div key={index} className="error-item">
              <AlertCircle size={16} />
              {error}
            </div>
          ))}
        </div>
      )}

      {/* Selected files */}
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
                    {file.processed && file.wordCount && (
                      <span className="word-count-info">
                        • {file.wordCount} words
                      </span>
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

      {/* Preview boxes - Stacked vertically */}
      <div className="preview-sections">
        {/* Uploaded Preview */}
        {selectedFiles.length > 0 && (
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
                    {index < selectedFiles.length - 1 && "\n---\n"}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Corpus Preview */}
        {corpusPreview && (
          <div className="preview-box">
            <h3 className="preview-title">Corpus Preview:</h3>
            <div className="preview-content">
              {corpusPreview}
            </div>
          </div>
        )}
      </div>

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