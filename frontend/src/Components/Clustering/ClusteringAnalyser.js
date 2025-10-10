import React, { useState, useEffect, useMemo } from "react";
import ClusteringCharts from "./ClusteringCharts";
import CreativeClusteringAnalysis from "./CreativeClusteringAnalysis";
import '../ProgressBar.css';
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

  const [showTopTerms, setShowTopTerms] = useState(false);
  const [showDocs, setShowDocs] = useState(false);

  const parseTextDocuments = (text) => {
    if (!text) return [];
    let documents = text.split(/\n\s*\n/).filter(doc => doc.trim());
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
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/clustering-analysis/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: uploadedText }),
      });

      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
      const data = await response.json();

      const clustersWithCoords = data.clusters.map((c, idx) => ({
        ...c,
        x: c.pca_x ?? idx, 
        y: c.pca_y ?? idx,
      }));

      setClusters(data.clusters || []);
      setTopTerms(data.top_terms || {});
      setThemes(data.suggested_themes || {});
      setNumClusters(data.num_clusters || null);
      setNumDocs(data.num_docs || null);
      setSelectedCluster("all"); 

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

        <div className="analysis-description">
          <p className="description-text">
            This analysis uses ConceptNet embeddings to discover thematic and conceptual connections in your writing. 
            Text segments will be grouped based on semantic meaning and common-sense knowledge, helping you identify 
            how different parts of your work connect on a conceptual level.
          </p>
        </div>

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
