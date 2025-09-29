import React, { useState, useEffect, useMemo } from "react";
import ClusteringCharts from "./ClusteringCharts";
import CreativeClusteringAnalysis from "./CreativeClusteringAnalysis";
import './ClusteringAnalyser.css';

/**
 * Lightweight progress bar for clustering analysis.
 */
const ProgressBar = ({ loading }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let timer;
    if (loading) {
      setProgress(0);
      timer = setInterval(() => {
        setProgress((old) => (old < 90 ? old + Math.random() * 3 : old));
      }, 200);
    } else {
      setProgress(100);
      const reset = setTimeout(() => setProgress(0), 500);
      return () => clearTimeout(reset);
    }
    return () => clearInterval(timer);
  }, [loading]);

  return (
    <div className="progress-container-wrapper">
      <div className="progress-container">
      <div
          className="progress-fill"
        style={{ width: `${progress}%` }}
      ></div>
        <div className="progress-text">
        {Math.floor(progress)}%
      </div>
    </div>
    </div>
  );
};

const ClusteringAnalyser = ({ uploadedText, onBack }) => {
  const [clusters, setClusters] = useState([]);
  const [topTerms, setTopTerms] = useState({});
  const [themes, setThemes] = useState({});
  const [numClusters, setNumClusters] = useState(null);
  const [numDocs, setNumDocs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedCluster, setSelectedCluster] = useState("all");
  const [embedding, setEmbedding] = useState("conceptnet");
  const [showEmbeddingOptions, setShowEmbeddingOptions] = useState(true);

  // Toggles
  const [showTopTerms, setShowTopTerms] = useState(false);
  const [showDocs, setShowDocs] = useState(false);

  // Embedding options with descriptions
  const embeddingOptions = [
    {
      id: "conceptnet",
      name: "ConceptNet",
      title: "Discover thematic and conceptual connections",
      description: "Choose ConceptNet when you want to uncover deeper thematic relationships and conceptual connections in your writing. This method groups text based on semantic meaning and common-sense knowledge, helping you identify how different parts of your work connect on a conceptual level. Perfect for understanding the underlying themes, motifs, and conceptual threads that run through your creative work."
    },
    {
      id: "spacy",
      name: "spaCy",
      title: "Find linguistic patterns and structural similarities",
      description: "Choose spaCy when you want to discover patterns based on linguistic structure, grammar, and word relationships. This method focuses on how you actually use language - sentence structure, grammatical patterns, and word choice similarities. Ideal for understanding your writing style, identifying recurring linguistic habits, and seeing how different sections of your work share similar structural approaches."
    }
  ];

  // Parse uploaded text into document array
  const parseTextDocuments = (text) => {
    if (!text) return [];
    
    // Split by double line breaks (common document separator)
    let documents = text.split(/\n\s*\n/).filter(doc => doc.trim());
    
    // If no double line breaks, split by single line breaks
    if (documents.length === 1) {
      documents = text.split(/\n/).filter(doc => doc.trim());
    }
    
    // If still just one document, split by sentences (for very long text)
    if (documents.length === 1 && text.length > 1000) {
      documents = text.match(/[^.!?]+[.!?]+/g) || [text];
    }
    
    return documents.map(doc => doc.trim()).filter(doc => doc.length > 0);
  };

  // Create the textDocuments array
  const textDocuments = parseTextDocuments(uploadedText);

  const runAnalysis = async () => {
    if (!uploadedText) return;

    try {
      setLoading(true);
      setError("");
      const response = await fetch("http://localhost:8000/api/clustering-analysis/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: uploadedText, embedding }),
      });

      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
      const data = await response.json();

      // Add PCA coordinates to clusters for plotting
      const clustersWithCoords = data.clusters.map((c, idx) => ({
        ...c,
        x: c.pca_x ?? idx, // fallback in case pca_x not returned
        y: c.pca_y ?? idx,
      }));

      setClusters(data.clusters || []);
      setTopTerms(data.top_terms || {});
      setThemes(data.suggested_themes || {});
      setNumClusters(data.num_clusters || null);
      setNumDocs(data.num_docs || null);
      setSelectedCluster("all"); // reset filter
      setShowEmbeddingOptions(false); // Hide embedding options when results are displayed

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify({ clusters, topTerms, themes }, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "clustering_results.json";
    link.click();
  };

  const handleEmbeddingChange = (embeddingId) => {
    setEmbedding(embeddingId);
  };

  return (
    <div className="clustering-container">
      <button
        onClick={onBack}
        className="clustering-back-button"
      >
        ← Back
      </button>

      <div className="clustering-card">
        <h1 className="clustering-title">
          Clustering Analysis
        </h1>

        {/* Embedding Selection Section or Collapsed State */}
        {showEmbeddingOptions ? (
          <div className="embedding-selection">
            <h2 className="embedding-selection-title">Choose Your Analysis Method</h2>
            <div className="embedding-container">
              {embeddingOptions.map((option) => (
                <div 
                  key={option.id} 
                  className={`embedding-card ${embedding === option.id ? 'selected' : ''}`}
                  onClick={() => handleEmbeddingChange(option.id)}
                >
                  <div className="embedding-card-content">
                    {/* Left side - Description */}
                    <div className="embedding-description">
                      <h3 className="embedding-title">
                        {option.name}: {option.title}
                      </h3>
                      <p className="embedding-text">
                        {option.description}
                      </p>
        </div>

                    {/* Right side - Radio button */}
                    <div className="embedding-radio-container">
                      <input
                        type="radio"
                        name="embedding"
                        value={option.id}
                        checked={embedding === option.id}
                        onChange={(e) => handleEmbeddingChange(e.target.value)}
                        className="embedding-radio"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="collapsed-embedding-selection">
            <div className="current-embedding-info">
              <span className="current-embedding-text">
                Using <strong>{embeddingOptions.find(opt => opt.id === embedding)?.name || embedding}</strong> embeddings
              </span>
              <button 
                onClick={() => setShowEmbeddingOptions(true)}
                className="change-embedding-button"
                disabled={loading}
              >
                Change Method
              </button>
            </div>
          </div>
        )}

        {/* Run Analysis button */}
        <div className="text-center">
          <button
            onClick={runAnalysis}
            disabled={loading || !uploadedText}
            className="analysis-button"
          >
            {loading ? "Analysing..." : "Run Analysis"}
          </button>
        </div>

        {/* Progress Bar */}
          {loading && <ProgressBar loading={loading} />}

        {/* Error */}
        {error && (
          <div className="error-message">
            Error: {error}
          </div>
        )}

        {/* Metadata */}
        {!loading && !error && numClusters && numDocs && (
          <p className="analysis-metadata">
            Automatically grouped into {numClusters} clusters based on {numDocs} text segments.
          </p>
        )}

        {/* Results */}
        {!loading && !error && clusters.length > 0 && (
          <div className="results-section">
          <CreativeClusteringAnalysis
            clusters={clusters}
            topTerms={topTerms}
            themes={themes}
            textDocuments={textDocuments}
          />
          </div>
        )}
      </div>
    </div>
  );
};

export default ClusteringAnalyser;
