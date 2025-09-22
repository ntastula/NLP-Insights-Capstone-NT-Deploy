import React, { useState, useEffect } from "react";
import ClusteringCharts from "./ClusteringCharts";
import CreativeClusteringAnalysis from "./CreativeClusteringAnalysis";
import '../ProgressBar.css';


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
    <div className="progress-container bg-gray-200 rounded overflow-hidden relative h-6 w-full mb-4">
      <div
        className="progress-fill bg-blue-600 h-full transition-all duration-200"
        style={{ width: `${progress}%` }}
      ></div>
      <div className="progress-text absolute w-full text-center top-0 left-0 font-medium text-white">
        {Math.floor(progress)}%
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

  // Toggles
  const [showTopTerms, setShowTopTerms] = useState(false);
  const [showDocs, setShowDocs] = useState(false);

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
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <button
        onClick={onBack}
        className="mb-6 bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded shadow"
      >
        ← Back
      </button>

      <div className="bg-white rounded-2xl shadow-lg p-10 max-w-5xl w-full">
        <h1 className="text-3xl font-bold text-green-600 mb-6 text-center">
          Clustering Analysis
        </h1>

        {/* Embedding selection */}
        <div className="mb-4">
          <label className="mr-2 font-medium">Embeddings:</label>
          <select
            value={embedding}
            onChange={(e) => setEmbedding(e.target.value)}
            className="border rounded p-1"
          >
            <option value="conceptnet">ConceptNet</option>
            <option value="spacy">spaCy</option>
          </select>
        </div>

        {/* Run Analysis button */}
        <div className="mb-6">
          <button
            onClick={runAnalysis}
            disabled={loading || !uploadedText}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded shadow"
          >
            {loading ? "Analysing..." : "Run Analysis"}
          </button>
        </div>

        {/* Progress Bar */}
          {loading && <ProgressBar loading={loading} />}

        {/* Error */}
        {error && <p className="text-red-500 mb-4">Error: {error}</p>}

        {/* Metadata */}
        {!loading && !error && numClusters && numDocs && (
          <p className="text-gray-600 text-center mb-6">
            Automatically grouped into {numClusters} clusters based on {numDocs} text segments.
          </p>
        )}

        {/* Results */}
        {!loading && !error && clusters.length > 0 && (
          <CreativeClusteringAnalysis
            clusters={clusters}
            topTerms={topTerms}
            themes={themes}
          />
        )}
      </div>
    </div>
  );
};

export default ClusteringAnalyser;
