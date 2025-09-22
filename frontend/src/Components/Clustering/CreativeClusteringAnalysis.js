import React, { useState } from "react";
import ClusteringCharts from "./ClusteringCharts";

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

  return (
  <div className="space-y-6">
  {/* Buttons */}
  <div className="mt-4 flex gap-4">
    <button
      className="bg-yellow-600 text-white px-4 py-2 rounded hover:bg-yellow-700"
      onClick={() => {
        setShowChart(true);
        setShowTopTerms(false);
        setShowDocuments(false);
      }}
    >
      Show Chart
    </button>

    <button
      className="bg-purple-600 text-white px-4 py-2 rounded hover:bg-purple-700"
      onClick={() => {
        setShowTopTerms(true);
        setShowChart(false);
        setShowDocuments(false);
      }}
    >
      Show Top Terms
    </button>

    <button
      className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
      onClick={() => {
        setShowDocuments(true);
        setShowChart(false);
        setShowTopTerms(false);
      }}
    >
      Show Clustered Documents
    </button>

    <button
      className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      onClick={downloadResults}
    >
      Download Results
    </button>
  </div>

    {/* Cluster selection dropdown */}
    {clusters.length > 0 && (
      <div className="mb-4">
        <label className="mr-2 font-medium">Filter Cluster:</label>
        <select
          value={selectedCluster}
          onChange={e => setSelectedCluster(e.target.value)}
          className="border rounded p-1"
        >
          <option value="all">All</option>
          {clusterOptions.map(label => (
            <option key={label} value={label}>
              Cluster {label}
            </option>
          ))}
        </select>
      </div>
    )}

    {/* Scatterplot chart */}
    {showChart && clusters.length > 0 && (
      <div className="mb-6">
        <ClusteringCharts 
  clusters={clusters} 
  selectedCluster={selectedCluster} 
/>
      </div>
    )}

    {/* Top terms per cluster */}
    {showTopTerms && Object.keys(topTerms).length > 0 && (
      <div className="grid grid-cols-2 gap-4 mb-6">
        {Object.entries(topTerms).map(([cluster, terms]) => (
          <div key={cluster} className="bg-gray-50 p-4 rounded shadow">
            <h3 className="font-bold text-purple-600">Cluster {cluster}</h3>
            <p>{terms.join(", ")}</p>
            {themes[cluster] && (
              <p className="text-sm text-green-600 mt-1">
                Suggested theme: {themes[cluster]}
              </p>
            )}
          </div>
        ))}
      </div>
    )}

    {/* Clustered documents */}
    {showDocuments && displayedClusters.length > 0 && (
      <div>
        <h2 className="text-xl font-semibold mb-2">Clustered Documents</h2>
        <ul className="space-y-2">
          {displayedClusters.map((item, idx) => (
            <li key={idx} className="bg-gray-50 p-3 rounded shadow">
              <span className="font-bold text-blue-600">Cluster {item.label}:</span>{" "}
              {item.doc}
            </li>
          ))}
        </ul>
      </div>
    )}
  </div>
);

};

export default CreativeClusteringAnalysis;

