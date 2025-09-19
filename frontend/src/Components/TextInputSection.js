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
    <div className="keyness-container">
      {/* Paste textarea */}
      <div>
        <label className="block text-lg font-semibold mb-2">
          Paste Your Text
        </label>
        <textarea
          value={pastedText}
          onChange={handleTextPaste}
          className="keyness-textarea"
          placeholder="Paste text here..."
          style={{
            width: "100%",
            minHeight: "120px",
            padding: "12px",
            border: "1px solid #ccc",
            borderRadius: "6px",
            fontSize: "14px",
            fontFamily: "monospace",
            resize: "vertical",
          }}
        />
        {pastedText && (
          <div
            style={{ fontSize: "0.9em", color: "#666", marginTop: "6px" }}
          >
            Word count: {pastedWordCount}
          </div>
        )}
      </div>

      {/* Drag & Drop */}
      <div
        ref={dropzoneRef}
        className={`keyness-dropzone ${hover ? "hover" : ""} ${
          uploading ? "uploading" : ""
        }`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        style={{
          width: "100%",
          border: uploading ? "2px solid #007bff" : "2px dashed #ccc",
          padding: "40px 20px",
          textAlign: "center",
          borderRadius: "8px",
          backgroundColor: uploading
            ? "#f0f8ff"
            : hover
            ? "#f0f8ff"
            : "transparent",
          borderColor: uploading ? "#007bff" : hover ? "#007bff" : "#ccc",
          transition: "all 0.3s ease",
          opacity: uploading ? 0.7 : 1,
        }}
      >
        <div style={{ pointerEvents: "none" }}>
          {uploading ? (
            <div>
              <Upload className="animate-pulse mx-auto mb-2" size={24} />
              Uploading… {uploadProgress}%
              <div
                style={{
                  width: "100%",
                  height: "6px",
                  background: "#eee",
                  marginTop: "8px",
                  borderRadius: "3px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${uploadProgress}%`,
                    height: "100%",
                    background: "#007bff",
                    transition: "width 0.2s",
                  }}
                />
              </div>
            </div>
          ) : (
            <>
              Drag & drop files here
              {hover && draggedFileName && (
                <div style={{ marginTop: "8px", color: "#333" }}>
                  Release to upload: <strong>{draggedFileName}</strong>
                </div>
              )}
              <div
                style={{
                  fontSize: "0.9em",
                  color: "#666",
                  marginTop: "8px",
                }}
              >
                Supported: .txt, .doc, .docx (max 5MB each)
              </div>
            </>
          )}
        </div>
        <input
          id="fileInput"
          type="file"
          multiple
          style={{ display: "none" }}
          onChange={handleFileSelect}
          accept=".txt,.doc,.docx"
          disabled={uploading}
        />
      </div>

      {/* Select Files Button */}
      <button
        type="button"
        onClick={() =>
          !uploading && document.getElementById("fileInput").click()
        }
        style={{
          marginTop: "10px",
          padding: "6px 12px",
          background: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "4px",
          cursor: uploading ? "not-allowed" : "pointer",
        }}
      >
        Select Files
      </button>

      {/* Upload Success Messages */}
      {uploadSuccess.length > 0 && (
        <div
          style={{
            backgroundColor: "#d1f2eb",
            border: "1px solid #52c41a",
            borderRadius: "6px",
            padding: "12px",
            marginTop: "12px",
          }}
        >
          {uploadSuccess.map((message, index) => (
            <div
              key={index}
              style={{ color: "#389e0d", marginBottom: "4px" }}
            >
              {message}
            </div>
          ))}
        </div>
      )}

      {/* Upload Error Messages */}
      {uploadErrors.length > 0 && (
        <div
          style={{
            backgroundColor: "#fff2f0",
            border: "1px solid #ff4d4f",
            borderRadius: "6px",
            padding: "12px",
            marginTop: "12px",
          }}
        >
          {uploadErrors.map((error, index) => (
            <div
              key={index}
              style={{ color: "#cf1322", marginBottom: "4px" }}
            >
              <AlertCircle
                size={16}
                style={{ display: "inline", marginRight: "6px" }}
              />
              {error}
            </div>
          ))}
        </div>
      )}

      {/* Selected files */}
      {selectedFiles.length > 0 && (
        <div
          className="keyness-file-list"
          style={{
            width: "100%",
            padding: "16px",
            backgroundColor: "#f8f9fa",
            borderRadius: "6px",
            border: "1px solid #e9ecef",
            marginTop: "12px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "12px",
            }}
          >
            <h4 style={{ margin: 0, color: "#495057" }}>Selected Files:</h4>
            <button
              onClick={clearAllFiles}
              style={{
                background: "none",
                border: "none",
                color: "#6c757d",
                cursor: "pointer",
                fontSize: "0.9em",
              }}
            >
              Clear All
            </button>
          </div>
          {selectedFiles.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "8px 0",
                borderBottom:
                  index < selectedFiles.length - 1
                    ? "1px solid #dee2e6"
                    : "none",
              }}
            >
              <div style={{ display: "flex", alignItems: "center" }}>
                <CheckCircle
                  size={16}
                  style={{ marginRight: "8px", color: "green" }}
                />
                <span>
                  {file.name} ({Math.round(file.size / 1024)}KB)
                  {file.processed && file.wordCount && (
                    <span
                      style={{
                        color: "#6c757d",
                        fontSize: "0.9em",
                      }}
                    >
                      {" "}
                      • {file.wordCount} words
                    </span>
                  )}
                </span>
              </div>
              <button
                onClick={() => removeFile(index)}
                style={{
                  background: "none",
                  border: "none",
                  color: "#dc3545",
                  cursor: "pointer",
                  padding: "4px",
                }}
              >
                <X size={16} />
              </button>
            </div>
          ))}
        </div>
      )}

{/* Uploaded Preview */}
{selectedFiles.length > 0 && (
  <div
    className="keyness-preview"
    style={{
      width: "100%",
      padding: "16px",
      backgroundColor: "#f8f9fa",
      borderRadius: "6px",
      border: "1px solid #e9ecef",
      marginTop: "12px",
    }}
  >
    <h3
      className="font-semibold mb-2"
      style={{ marginBottom: "12px", color: "#495057" }}
    >
      Uploaded Text Preview:
    </h3>
    <pre
      style={{
        whiteSpace: "pre-wrap",
        maxHeight: "200px",   // keeps box scrollable
        overflowY: "auto",
        backgroundColor: "#ffffff",
        padding: "12px",
        borderRadius: "4px",
        border: "1px solid #dee2e6",
        fontSize: "13px",
        fontFamily: "monospace",
        margin: 0,
      }}
    >
      {selectedFiles.map((file, index) => {
        const previewText = file.textContent
          ? file.textContent.split("\n").slice(0, 4).join("\n") // first 4 lines
          : "";

        return (
          <div key={index} style={{ marginBottom: "1em" }}>
            <strong>{file.name}</strong>
            {"\n"}
            {previewText}
            {"\n---\n"}
          </div>
        );
      })}
    </pre>
  </div>
)}





      {/* Corpus Preview */}
{corpusPreview && (
  <div
    className="keyness-preview"
    style={{
      width: "100%",
      padding: "16px",
      backgroundColor: "#f8f9fa",
      borderRadius: "6px",
      border: "1px solid #e9ecef",
      marginTop: "12px",
    }}
  >
    <h3
      className="font-semibold mb-2"
      style={{ marginBottom: "12px", color: "#495057" }}
    >
      Corpus Preview:
    </h3>
    <pre
      style={{
        whiteSpace: "pre-wrap",
        maxHeight: "200px",
        overflowY: "auto",
        backgroundColor: "#ffffff",
        padding: "12px",
        borderRadius: "4px",
        border: "1px solid #dee2e6",
        fontSize: "13px",
        fontFamily: "monospace",
        margin: 0,
      }}
    >
      {corpusPreview}
    </pre>
  </div>
)}



      {/* General error */}
      {error && (
        <div
          className="keyness-error"
          style={{
            width: "100%",
            color: "#dc3545",
            padding: "12px",
            backgroundColor: "#f8d7da",
            border: "1px solid #f5c6cb",
            borderRadius: "6px",
            marginTop: "12px",
          }}
        >
          {error}
        </div>
      )}
    </div>
  );
};

export default TextInputSection;