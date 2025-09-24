import React, { useState, useMemo } from "react";
import Plot from "react-plotly.js";
import { TrendingUp, BarChart3, ScatterChart, ToggleLeft, ToggleRight } from "lucide-react";
import "./Charts.css";

const Charts = ({ results, method = "nltk" }) => {
  const [chartType, setChartType] = useState("primary");
  const [topN, setTopN] = useState(20);
  const topResults = results ? results.slice(0, topN) : [];


  // Chart configurations for each method
  const chartConfigs = {
    nltk: {
      primary: {
        type: "bar",
        title: "Log-Likelihood Analysis",
        xKey: "log_likelihood",
        xLabel: "Log-Likelihood",
        description: "Higher values indicate stronger association with your text"
      },
      secondary: {
        type: "scatter",
        title: "Effect Size vs Log-Likelihood",
        xKey: "log_likelihood",
        yKey: "effect_size",
        xLabel: "Log-Likelihood",
        yLabel: "Effect Size",
        description: "Relationship between statistical significance and practical significance"
      }
    },
    sklearn: {
      primary: {
        type: "bar",
        title: "Chi-Square Analysis",
        xKey: "chi2",
        xLabel: "Chi² Score",
        description: "Higher Chi² values indicate stronger feature importance"
      },
      secondary: {
        type: "scatter",
        title: "Chi² vs P-value",
        xKey: "chi2",
        yKey: "p_value",
        xLabel: "Chi² Score",
        yLabel: "P-value (log scale)",
        description: "Statistical significance visualization"
      }
    },
    gensim: {
      primary: {
        type: "bar",
        title: "TF-IDF Analysis",
        xKey: "tfidf_score",
        xLabel: "TF-IDF Score",
        description: "Term frequency-inverse document frequency scores"
      }
    },
    spacy: {
      primary: {
        type: "bar",
        title: "Log-Likelihood Analysis",
        xKey: "log_likelihood",
        xLabel: "Log-Likelihood",
        description: "Primary keyness measure for linguistic analysis"
      },
      secondary: {
        type: "scatter",
        title: "Effect Size vs Chi²",
        xKey: "chi2",
        yKey: "effect_size",
        xLabel: "Chi² Score",
        yLabel: "Effect Size",
        description: "Practical vs statistical significance"
      }
    }
  };

  const currentConfig = chartConfigs[method.toLowerCase()] || chartConfigs.nltk;
  const hasSecondaryChart = currentConfig.secondary !== undefined;
  const activeConfig = currentConfig[chartType] || currentConfig.primary;

  // Prepare chart data
  const chartData = useMemo(() => {
  if (!results || results.length === 0) return [];

  return topResults.map(r => ({
    word: r.word,
    uploaded: r.uploaded_count ?? r.count_a ?? 0,
    sample: r.sample_count ?? r.count_b ?? 0,
    keyness_score: r.keyness_score ?? r.keyness ?? 0,
    direction: r.direction ?? "Neutral",
    x: r[activeConfig.xKey] ?? r.keyness_score ?? 0,  
    y: r[activeConfig.yKey] ?? r.effect_size ?? 0     
  }));
}, [topResults, activeConfig, results]);



  if (!results || results.length === 0) {
    return <p className="no-data-message">No chart data available.</p>;
  }

  // Create bar chart with uploaded vs corpus counts
const createBarChart = () => {
  return [
    {
      x: chartData.map(d => d.word),
      y: chartData.map(d => d.uploaded),
      name: "Uploaded Text",
      type: "bar",
      marker: { color: "#3b82f6", line: { color: "#1e40af", width: 1 } },
      hovertemplate: "<b>%{x}</b><br>Uploaded: %{y}<extra></extra>"
    },
    {
      x: chartData.map(d => d.word),
      y: chartData.map(d => d.sample),
      name: "Corpus",
      type: "bar",
      marker: { color: "#10b981", line: { color: "#059669", width: 1 } },
      hovertemplate: "<b>%{x}</b><br>Corpus: %{y}<extra></extra>"
    }
  ];
};


// Create scatter chart with x/y keyness fields
const createScatterChart = () => [
  {
    x: chartData.map(d => d.x),
    y: chartData.map(d => d.y),
    text: chartData.map(d => d.word),
    mode: "markers+text",
    type: "scatter",
    marker: { size: 12, color: "#3b82f6", line: { color: "#1e40af", width: 2 }, opacity: 0.8 },
    textposition: "top center",
    textfont: { size: 10, color: "#1e40af" },
    hovertemplate: "<b>%{text}</b><br>" +
                   `${activeConfig.xLabel}: %{x:.3f}<br>` +
                   `${activeConfig.yLabel}: %{y:.3f}<extra></extra>`
  }
];



  const plotData = activeConfig.type === "bar" ? createBarChart() : createScatterChart();

  const getLayout = () => {
    const baseLayout = {
      title: { text: activeConfig.title, font: { size: 24, family: "Inter, sans-serif", color: "#1e293b" }, x: 0.5 },
      paper_bgcolor: "rgba(255, 255, 255, 0.95)",
      plot_bgcolor: "rgba(248, 250, 252, 0.8)",
      margin: { l: 80, r: 80, t: 100, b: 120 },
      showlegend: true,
      legend: {
        orientation: "h",
        x: 0.5,
        xanchor: "center",
        y: -0.2,
        bgcolor: "rgba(255, 255, 255, 0.9)",
        bordercolor: "#e2e8f0",
        borderwidth: 1
      },
      hovermode: "closest"
    };

    if (activeConfig.type === "bar") {
      return {
        ...baseLayout,
        xaxis: {
          title: { text: "Words", font: { size: 14, color: "#475569" } },
          tickangle: -45,
          tickfont: { size: 11, color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)"
        },
        yaxis: {
          title: { text: activeConfig.xLabel, font: { size: 14, color: "#475569" } },
          tickfont: { color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)"
        },
        barmode: "group"
      };
    } else {
      return {
        ...baseLayout,
        xaxis: {
          title: { text: activeConfig.xLabel, font: { size: 14, color: "#475569" } },
          tickfont: { color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)",
          type: activeConfig.yKey === "p_value" ? "linear" : "linear"
        },
        yaxis: {
          title: { text: activeConfig.yLabel, font: { size: 14, color: "#475569" } },
          tickfont: { color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)",
          type: activeConfig.yKey === "p_value" ? "log" : "linear"
        }
      };
    }
  };

  return (
    <div className="charts-container">
      <div className="chart-header">
        <h3 className="chart-title">
          <TrendingUp className="chart-icon" />
          Keyness Analysis ({method.toUpperCase()})
        </h3>

        {hasSecondaryChart && (
          <div className="chart-toggle">
            <button
              onClick={() => setChartType(chartType === "primary" ? "secondary" : "primary")}
              className={`toggle-button ${chartType}`}
            >
              {chartType === "primary" ? (
                <>
                  <BarChart3 className="toggle-icon" />
                  Switch to Scatter
                  <ToggleRight className="toggle-indicator" />
                </>
              ) : (
                <>
                  <ScatterChart className="toggle-icon" />
                  Switch to Bar
                  <ToggleLeft className="toggle-indicator" />
                </>
              )}
            </button>
          </div>
        )}
      </div>

      <div className="chart-description">
        <p>{activeConfig.description}</p>
      </div>

      <div className="chart-slider">
  <label htmlFor="topNSlider">
    Number of results: <strong>{topN}</strong>
  </label>
  <input
    id="topNSlider"
    type="range"
    min={5}
    max={50}
    step={1}
    value={topN}
    onChange={(e) => setTopN(parseInt(e.target.value))}
  />
</div>

      <div className="chart-wrapper">
        <Plot
          data={plotData}
          layout={getLayout()}
          style={{ width: "100%", height: "600px" }}
          useResizeHandler={true}
          config={{
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            toImageButtonOptions: {
              format: 'png',
              filename: `${method}_keyness_analysis`,
              height: 800,
              width: 1200,
              scale: 2
            }
          }}
        />
      </div>

      <div className="chart-footer">
        <div className="method-info">
          <strong>Method:</strong> {method.toUpperCase()} | 
          <strong> Chart Type:</strong> {activeConfig.type === "bar" ? "Bar Chart" : "Scatter Plot"}
        </div>
      </div>
    </div>
  );
};

export default Charts;
