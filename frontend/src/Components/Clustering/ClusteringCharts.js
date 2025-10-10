import React, { useState, useMemo, useEffect } from "react";
import Plot from "react-plotly.js";
import "./ClusteringCharts.css";

const ClusteringCharts = ({ clusters, selectedCluster = "all" }) => {
  const [maxWords, setMaxWords] = useState(5);
  const [showWordsOnChart, setShowWordsOnChart] = useState(false);

  const clusterColors = [
    '#ef4444', // Red
    '#3b82f6', // Blue
    '#10b981', // Green
    '#f59e0b', // Amber
    '#8b5cf6', // Violet
    '#ec4899', // Pink
    '#06b6d4', // Cyan
    '#84cc16', // Lime
    '#f97316', // Orange
    '#6366f1', // Indigo
    '#14b8a6', // Teal
    '#eab308', // Yellow
    '#f43f5e', // Rose
    '#8b5cf6', // Purple
    '#06b6d4', // Sky
  ];

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

    return Object.entries(grouped).map(([label, points], index) => {
      const clusterColor = clusterColors[index % clusterColors.length] ||
        `hsl(${(label * 137.5) % 360}, 70%, 55%)`;

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
          size: 12,
          color: clusterColor,
          line: {
            width: 2,
            color: "rgba(255, 255, 255, 0.8)"
          },
          opacity: 0.8,
        },
        textfont: {
          color: clusterColor,
          size: 11,
          family: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        },
        textposition: "middle right",
        hovertemplate:
          "<b>%{text}</b><br>" +
          "X: %{x:.2f}<br>" +
          "Y: %{y:.2f}<br>" +
          "<extra>Cluster " + label + "</extra>",
        hoverlabel: {
          bgcolor: clusterColor,
          bordercolor: "white",
          font: { color: "white", size: 12 }
        }
      };
    });
  }, [displayedClusters, maxWords, showWordsOnChart, clusterColors]);

  // Dynamic chart title
  const chartTitle =
    selectedCluster === "all"
      ? "Cluster Scatterplot (All Clusters)"
      : `Cluster Scatterplot (Cluster ${selectedCluster})`;

  // Layout configuration with fixed legend positioning
  const plotLayout = {
    title: {
      text: chartTitle,
      font: {
        size: 24,
        family: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        color: "#1e293b"
      },
      x: 0.5,
      xanchor: "center",
      y: 0.98,
      yanchor: "top"
    },
    xaxis: {
      title: {
        text: "PCA Component 1",
        font: { size: 14, color: "#475569" }
      },
      gridcolor: "rgba(226, 232, 240, 0.6)",
      zerolinecolor: "rgba(148, 163, 184, 0.8)",
      tickfont: { color: "#64748b", size: 11 }
    },
    yaxis: {
      title: {
        text: "PCA Component 2",
        font: { size: 14, color: "#475569" }
      },
      gridcolor: "rgba(226, 232, 240, 0.6)",
      zerolinecolor: "rgba(148, 163, 184, 0.8)",
      tickfont: { color: "#64748b", size: 11 }
    },
    showlegend: true,
    legend: {
      orientation: "v",
      x: 1.02,
      xanchor: "left",
      y: 1,
      yanchor: "top",
      bgcolor: "rgba(255, 255, 255, 0.95)",
      bordercolor: "rgba(226, 232, 240, 0.8)",
      borderwidth: 1,
      font: { size: 11, color: "#475569" }
    },
    plot_bgcolor: "rgba(248, 250, 252, 0.5)",
    paper_bgcolor: "white",
    margin: { l: 70, r: 150, t: 80, b: 70 },
    autosize: true,
    hovermode: "closest"
  };

  // Responsive config that adjusts margins based on container
  const plotConfig = {
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'cluster_scatterplot',
      height: 800,
      width: 1200,
      scale: 2
    }
  };

  return (
    <div className="clustering-charts-container">
      {/* Controls */}
      <div className="controls-section">
        <div className="control-group">
          <label htmlFor="max-words">Max words on hover:</label>
          <input
            id="max-words"
            type="number"
            min={1}
            max={20}
            value={maxWords}
            onChange={(e) => setMaxWords(Number(e.target.value))}
            className="control-input"
          />
        </div>

        <label className="checkbox-control">
          <input
            type="checkbox"
            checked={showWordsOnChart}
            onChange={(e) => setShowWordsOnChart(e.target.checked)}
            className="checkbox-input"
          />
          <span>Show words on chart</span>
        </label>
      </div>

      {/* Plot */}
      <div className="plot-container">
        <Plot
          data={traceData}
          layout={plotLayout}
          config={plotConfig}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler={true}
        />
      </div>
    </div>
  );
};

export default ClusteringCharts;
