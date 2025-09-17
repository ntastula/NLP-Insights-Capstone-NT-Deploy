import React, { useEffect, useState } from "react";

const ClusteringAnalyser = ({ uploadedText, onBack }) => {
  const [clusters, setClusters] = useState([]);
  const [topTerms, setTopTerms] = useState({});
  const [themes, setThemes] = useState({});
  const [numClusters, setNumClusters] = useState(null);
  const [numDocs, setNumDocs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedCluster, setSelectedCluster] = useState("all"); // <-- dropdown state

  const runAnalysis = async () => {
    try {
      setLoading(true);
      setError("");
      const response = await fetch("http://localhost:8000/api/clustering-analysis/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: uploadedText }),
      });

      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
      const data = await response.json();

      // Save results into state (match backend field names)
      setClusters(data.clusters || []);
      setTopTerms(data.top_terms || {});
      setThemes(data.suggested_themes || {});
      setNumClusters(data.num_clusters || null);
      setNumDocs(data.num_docs || null);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (uploadedText) runAnalysis();
  }, [uploadedText]);

  // Compute cluster options and filtered clusters
  const clusterOptions = Array.from(new Set(clusters.map(c => c.label))).sort((a, b) => a - b);
  const displayedClusters =
    selectedCluster === "all"
      ? clusters
      : clusters.filter(c => c.label === Number(selectedCluster));

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <button
        onClick={onBack}
        className="mb-6 bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-2 rounded shadow"
      >
        ← Back
      </button>

      <div className="bg-white rounded-2xl shadow-lg p-10 max-w-3xl w-full">
        <h1 className="text-3xl font-bold text-green-600 mb-6 text-center">
          Clustering Analysis
        </h1>

        {/* Loading & errors */}
        {loading && <p className="text-gray-600">Analysing text…</p>}
        {error && <p className="text-red-500">Error: {error}</p>}

        {/* Metadata */}
        {!loading && !error && numClusters && numDocs && (
          <p className="text-gray-600 text-center mb-6">
            Automatically grouped into {numClusters} clusters based on {numDocs} text segments.
          </p>
        )}

        {/* Results */}
        {!loading && !error && clusters.length > 0 && (
          <>
            {/* Top terms per cluster */}
            <h2 className="text-xl font-semibold mb-4">Top Terms per Cluster</h2>
            <div className="grid grid-cols-2 gap-4 mb-6">
              {Object.entries(topTerms).map(([cluster, terms]) => (
                <div key={cluster} className="bg-gray-50 p-4 rounded shadow">
                  <h3 className="font-bold text-purple-600">Cluster {cluster}</h3>
                  <p className="text-gray-700">{terms.join(", ")}</p>
                  {themes[cluster] && (
                    <p className="text-sm text-green-600 mt-1">
                      Suggested theme: {themes[cluster]}
                    </p>
                  )}
                </div>
              ))}
            </div>

            {/* Clustered documents with dropdown filter */}
            <h2 className="text-xl font-semibold mb-4">Clustered Documents</h2>

            {/* Dropdown */}
            <div className="mb-4">
              <label className="mr-2 font-medium">Select Cluster:</label>
              <select
                value={selectedCluster}
                onChange={(e) => setSelectedCluster(e.target.value)}
                className="border rounded p-1"
              >
                <option value="all">All</option>
                {clusterOptions.map((label) => (
                  <option key={label} value={label}>
                    Cluster {label}
                  </option>
                ))}
              </select>
            </div>

            {/* Filtered documents */}
            <ul className="space-y-2 text-left">
              {displayedClusters.map((item, idx) => (
                <li key={idx} className="bg-gray-50 p-3 rounded shadow">
                  <span className="font-bold text-blue-600">Cluster {item.label}:</span>{" "}
                  {item.doc}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
};

export default ClusteringAnalyser;

