import React, { useState, useMemo, useEffect } from "react";
import Plot from "react-plotly.js";

const ClusteringCharts = ({ clusters, selectedCluster = "all" }) => {
  const [maxWords, setMaxWords] = useState(5);
  const [showWordsOnChart, setShowWordsOnChart] = useState(false);

  // Debug logs
  useEffect(() => {
    console.log("Raw clusters prop:", clusters);
    if (clusters && clusters.length > 0) {
      console.log("First 3 clusters:", clusters.slice(0, 3));
    }
  }, [clusters]);

  // Filter clusters for selectedCluster
  const displayedClusters =
    selectedCluster === "all"
      ? clusters
      : clusters.filter((c) => c.label === Number(selectedCluster));

      useEffect(() => {
    console.log("Displayed clusters after filter:", displayedClusters.slice(0, 3));
  }, [displayedClusters]);

  // Group clusters by label
  const grouped = useMemo(() => {
    const g = {};
    displayedClusters.forEach((c) => {
      if (!g[c.label]) g[c.label] = [];
      g[c.label].push(c);
    });
    return g;
  }, [displayedClusters]);

  useEffect(() => {
    console.log("Grouped clusters:", grouped);
  }, [grouped]);

  // Prepare Plotly traces
const traceData = useMemo(() => {
  const grouped = {};
  displayedClusters.forEach((c) => {
    if (!grouped[c.label]) grouped[c.label] = [];
    grouped[c.label].push(c);
  });

  return Object.entries(grouped).map(([label, points]) => {
    const clusterColor = `hsl(${(label * 60) % 360}, 70%, 50%)`;

    return {
      x: points.map((p) => p.x),
      y: points.map((p) => p.y),
      text: points.map((p) => {
        const words = p.words || (p.doc ? p.doc.split(/\s+/) : []);
        return Array.isArray(words)
          ? words.slice(0, maxWords).join(", ")
          : words;
      }),
      mode: showWordsOnChart ? "markers+text" : "markers",
      type: "scatter",
      name: `Cluster ${label}`,
      marker: {
        size: 10,
        color: clusterColor,
        line: { width: 1, color: "#333" },
      },
      textfont: {
        color: clusterColor, // text now matches the cluster colour
        size: 12,
      },
      hovertemplate: "%{text}<extra>Cluster " + label + "</extra>",
    };
  });
}, [displayedClusters, maxWords, showWordsOnChart]);



  // Dynamic chart title
  const chartTitle =
    selectedCluster === "all"
      ? "Cluster Scatterplot (All Clusters)"
      : `Cluster Scatterplot (Cluster ${selectedCluster})`;

  return (
    <div className="bg-white p-4 rounded-2xl shadow-lg w-full mb-6">
      {/* Controls */}
      <div className="mb-4 flex flex-wrap gap-4 items-center">
        <label>
          Max words on hover:
          <input
            type="number"
            min={1}
            max={20}
            value={maxWords}
            onChange={(e) => setMaxWords(Number(e.target.value))}
            className="ml-2 border rounded p-1 w-16"
          />
        </label>

        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={showWordsOnChart}
            onChange={(e) => setShowWordsOnChart(e.target.checked)}
          />
          Show words on chart
        </label>
      </div>

      {/* Plot */}
      <Plot
        data={traceData}
        layout={{
          title: { text: chartTitle, font: { size: 24 }, x: 0.5 },
          xaxis: { title: "PCA 1" },
          yaxis: { title: "PCA 2" },
          showlegend: true,
        }}
        style={{ width: "80vw", height: "80vh" }}
        useResizeHandler={true}
      />
    </div>
  );
};

export default ClusteringCharts;
