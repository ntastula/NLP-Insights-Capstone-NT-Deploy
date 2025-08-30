import React, { useState, useRef } from "react";
import { CheckCircle } from "lucide-react";
import "./Keyness/KeynessLanding.css";

const TextInputSection = ({
  pastedText,
  handleTextPaste,
  uploadedPreview,
  corpusPreview,
  error,
  onFilesUploaded
}) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [hover, setHover] = useState(false);
  const [dragCounter, setDragCounter] = useState(0);
  const dropzoneRef = useRef(null);

  // Handle files when dropped or selected
  const handleFiles = (files) => {
    console.log('handleFiles called with:', files);
    const fileArray = Array.from(files);
    console.log('fileArray:', fileArray);

    // Limit to 5 files max
    if (fileArray.length + selectedFiles.length > 5) {
      alert("Maximum 5 files allowed");
      return;
    }

    // Read all files and combine their content
    const readPromises = fileArray.map(file => {
      console.log('Reading file:', file.name);
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          console.log('File read successfully:', file.name);
          resolve(reader.result);
        };
        reader.onerror = () => {
          console.error('Error reading file:', file.name, reader.error);
          reject(reader.error);
        };
        reader.readAsText(file);
      });
    });

    // Wait for all files to be read, then update state and call onFilesUploaded
    Promise.all(readPromises)
      .then(textContents => {
        console.log('All files read, textContents:', textContents);
        const combinedText = textContents.join('\n\n');
        const newFiles = [...selectedFiles, ...fileArray];
        
        console.log('Setting selectedFiles to:', newFiles);
        setSelectedFiles(newFiles);
        
        console.log('Calling onFilesUploaded');
        onFilesUploaded && onFilesUploaded(combinedText, newFiles);
      })
      .catch(error => {
        console.error('Error reading files:', error);
      });
  };

  // Drag events with counter to handle nested elements properly
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('dragEnter, counter:', dragCounter);
    console.log('dragEnter - dataTransfer.items:', e.dataTransfer.items);
    console.log('dragEnter - dataTransfer.files:', e.dataTransfer.files);
    console.log('dragEnter - dataTransfer.types:', e.dataTransfer.types);
    
    setDragCounter(prev => prev + 1);
    
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setHover(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('dragLeave, counter before:', dragCounter);
    
    setDragCounter(prev => {
      const newCounter = prev - 1;
      if (newCounter === 0) {
        setHover(false);
      }
      return newCounter;
    });
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
    console.log('dragOver - dataTransfer.items:', e.dataTransfer.items);
    console.log('dragOver - dataTransfer.files:', e.dataTransfer.files);
    console.log('dragOver - dataTransfer.types:', e.dataTransfer.types);
  };

  const handleDrop = (e) => {
    console.log('handleDrop called');
    e.preventDefault();
    e.stopPropagation();
    
    setHover(false);
    setDragCounter(0);
    
    console.log('dataTransfer:', e.dataTransfer);
    console.log('files:', e.dataTransfer.files);
    console.log('items:', e.dataTransfer.items);
    
    const files = e.dataTransfer.files;
    
    if (files && files.length > 0) {
      console.log('Found files in dataTransfer.files');
      handleFiles(files);
    } else {
      console.log('No files found');
    }
  };

  const handleFileSelect = (e) => {
    console.log('handleFileSelect called');
    const files = e.target.files;
    console.log('Files from file input:', files);
    if (files && files.length > 0) {
      handleFiles(files);
    }
    // Reset the input so the same file can be selected again if needed
    e.target.value = '';
  };

  // Global drag handlers to prevent default browser behavior
  React.useEffect(() => {
    const preventDefaults = (e) => {
      e.preventDefault();
      e.stopPropagation();
    };

    // Only prevent default behavior outside of our dropzone
    const handleGlobalDrop = (e) => {
      // Check if the drop target is our dropzone or a child of it
      if (!dropzoneRef.current?.contains(e.target)) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    const handleGlobalDragOver = (e) => {
      // Only prevent default outside our dropzone
      if (!dropzoneRef.current?.contains(e.target)) {
        e.preventDefault();
      }
    };

    // Add event listeners to prevent default drag behavior on the entire page
    document.addEventListener('dragover', handleGlobalDragOver, false);
    document.addEventListener('drop', handleGlobalDrop, false);

    return () => {
      document.removeEventListener('dragover', handleGlobalDragOver, false);
      document.removeEventListener('drop', handleGlobalDrop, false);
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
        />
      </div>

      {/* Drag & Drop */}
      <div
        ref={dropzoneRef}
        className={`keyness-dropzone ${hover ? "hover" : ""}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => document.getElementById("fileInput").click()}
        style={{
          border: '2px dashed #ccc',
          padding: '40px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          borderRadius: '8px',
          backgroundColor: hover ? '#f0f8ff' : 'transparent',
          borderColor: hover ? '#007bff' : '#ccc',
          transition: 'all 0.3s ease'
        }}
      >
        <div style={{ pointerEvents: 'none' }}>
          Drag & drop files here or click to select
          {hover && <div>Release to upload files</div>}
        </div>
        <input
          id="fileInput"
          type="file"
          multiple
          style={{ display: "none" }}
          onChange={handleFileSelect}
          accept=".txt,.doc,.docx,.pdf"
        />
      </div>

      {/* Selected files */}
      {selectedFiles.length > 0 && (
        <div className="keyness-file-list">
          <h4>Selected Files:</h4>
          {selectedFiles.map((file, index) => (
            <div key={`${file.name}-${index}`} className="keyness-file-item">
              <CheckCircle size={16} style={{ marginRight: '8px', color: 'green' }} />
              {file.name} ({Math.round(file.size / 1024)}KB)
            </div>
          ))}
        </div>
      )}

      {/* Uploaded text preview */}
      {uploadedPreview && (
        <div className="keyness-preview">
          <h3 className="font-semibold mb-2">Uploaded Text Preview:</h3>
          <pre style={{ whiteSpace: 'pre-wrap', maxHeight: '200px', overflow: 'auto' }}>
            {uploadedPreview}
          </pre>
        </div>
      )}

      {/* Corpus preview */}
      {corpusPreview && (
        <div className="keyness-corpus-preview">
          <h3 className="font-semibold mb-2">Corpus Preview:</h3>
          <pre style={{ whiteSpace: 'pre-wrap', maxHeight: '200px', overflow: 'auto' }}>
            {corpusPreview}
          </pre>
        </div>
      )}

      {/* Error messages */}
      {error && <div className="keyness-error" style={{ color: 'red', padding: '10px' }}>{error}</div>}
    </div>
  );
};

export default TextInputSection;