import React, { useState } from "react";
import ClusteringCharts from "./ClusteringCharts";
import "./CreativeClusteringAnalysis.css";

const CreativeClusteringAnalysis = ({ clusters, topTerms, themes }) => {
    const [showChart, setShowChart] = useState(true);
    const [selectedCluster, setSelectedCluster] = useState("all");
    const [showTopTerms, setShowTopTerms] = useState(false);
    const [showDocuments, setShowDocuments] = useState(false);

    // Unique cluster labels for dropdown
  const clusterOptions = Array.from(new Set(clusters.map(c => c.label))).sort(
    (a, b) => a - b
  );

const handleDownload = () => {
    const dataStr = JSON.stringify({ clusters, topTerms, themes }, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clustering_results.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  // Download results as JSON
  const downloadResults = () => {
    const data = {
      clusters,
      topTerms,
      themes,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clustering_results.json";
    a.click();
    URL.revokeObjectURL(url);
  };

    // Filtered clusters based on dropdown
    const displayedClusters =
        selectedCluster === "all"
            ? clusters
            : clusters.filter(c => c.label === Number(selectedCluster));

    const handleViewChange = (view) => {
        setShowChart(view === 'chart');
        setShowTopTerms(view === 'terms');
        setShowDocuments(view === 'documents');
    };

  return (
        <div className="clustering-results-container">
            {/* View Controls */}
            <div className="clustering-view-controls">
    <button
                    className={`clustering-btn btn-chart ${showChart ? 'active' : ''}`}
                    onClick={() => handleViewChange('chart')}
    >
      Show Chart
    </button>

                <button
                    className={`clustering-btn btn-terms ${showTopTerms ? 'active' : ''}`}
                    onClick={() => handleViewChange('terms')}
                >
                    Show Top Terms
                </button>

                <button
                    className={`clustering-btn btn-documents ${showDocuments ? 'active' : ''}`}
                    onClick={() => handleViewChange('documents')}
                >
                    Show Clustered Documents
                </button>

                <button
                    className="clustering-btn btn-download"
                    onClick={handleDownload}
                >
                    Download Results
                </button>
            </div>

            {/* Cluster Filter */}
            {clusters.length > 0 && (
                <div className="cluster-filter-section">
                    <label className="cluster-filter-label">Filter Cluster:</label>
                    <select
                        value={selectedCluster}
                        onChange={e => setSelectedCluster(e.target.value)}
                        className="cluster-filter-select"
                    >
                        <option value="all">All Clusters</option>
                        {clusterOptions.map(label => (
                            <option key={label} value={label}>
                                Cluster {label}
                            </option>
                        ))}
                    </select>
                </div>
            )}

            {/* Chart View */}
            {showChart && clusters.length > 0 && (
                <div className="chart-section">
                    <ClusteringCharts
                        clusters={clusters}
                        selectedCluster={selectedCluster}
                    />
                </div>
            )}

            {/* Top Terms View */}
            {showTopTerms && Object.keys(topTerms).length > 0 && (
                <div className="top-terms-grid">
                    {Object.entries(topTerms).map(([cluster, terms]) => (
                        <div key={cluster} className="cluster-term-card">
                            <h3 className="cluster-term-title">
                                Cluster {cluster}
                            </h3>
                            <div className="cluster-terms-list">
                                {terms.join(", ")}
                            </div>
                            {themes[cluster] && (
                                <div className="cluster-theme">
                                    Suggested theme: {themes[cluster]}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Documents View */}
            {showDocuments && displayedClusters.length > 0 && (
                <div className="documents-section">
                    <h2 className="documents-title">Clustered Documents</h2>
                    <div className="documents-list">
                        {displayedClusters.map((item, idx) => (
                            <div key={idx} className="document-item">
                                <span className="document-cluster-label">
                                    Cluster {item.label}:
                                </span>
                                <span className="document-text">
                                    {item.doc}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Empty State */}
            {!showChart && !showTopTerms && !showDocuments && (
                <div className="no-data-message">
                    Select a view to see your clustering results
                </div>
            )}
        </div>
    );
};

export default CreativeClusteringAnalysis;

