import React, { useState, useMemo, useEffect } from "react";
import Plot from "react-plotly.js";
import { TrendingUp, BarChart3, ScatterChart, ToggleLeft, ToggleRight, Info } from "lucide-react";
import "./Charts.css";

const Charts = ({ results, method = "nltk" }) => {
  const [chartType, setChartType] = useState("primary");
  const [topN, setTopN] = useState(20);
  const topResults = results ? results.slice(0, topN) : [];
  const [summary, setSummary] = useState("");
  const [loadingSummary, setLoadingSummary] = useState("");
  const [chartSummary, setChartSummary] = useState("");

  useEffect(() => {
  if (!results || results.length === 0) return;

  const fetchSummary = async () => {
    setLoadingSummary(true);
    try {
      const response = await fetch("/summarise-chart/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: `${method.toUpperCase()} Keyness Analysis`,
          chart_type: chartType === "primary" ? "bar" : "scatter",
          chart_data: results.slice(0, 20).map((r) => ({
            label: r.word,
            value: r.keyness ?? r.log_likelihood ?? r.chi2 ?? r.tfidf_score ?? 0,
          })),
        }),
      });
      const data = await response.json();
      setChartSummary(data.analysis || "No summary available.");
    } catch (err) {
      console.error("Error fetching chart summary:", err);
      setChartSummary("Failed to fetch chart summary.");
    } finally {
      setLoadingSummary(false);
    }
  };

  fetchSummary();
}, [chartType, results, method]);


  // Chart configurations for each method
  const chartConfigs = {
    nltk: {
      primary: {
        type: "bar",
        title: "Log-Likelihood Analysis",
        xKey: "log_likelihood",
        xLabel: "Log-Likelihood Score",
        description: "Higher log-likelihood values indicate stronger statistical association with your text compared to the reference corpus",
        color: "#10b981"
      },
      secondary: {
        type: "scatter",
        title: "Effect Size vs Log-Likelihood",
        xKey: "log_likelihood",
        yKey: "effect_size",
        xLabel: "Log-Likelihood Score",
        yLabel: "Effect Size",
        description: "Relationship between statistical significance (log-likelihood) and practical significance (effect size)",
        color: "#8b5cf6"
      }
    },
    sklearn: {
      primary: {
        type: "bar",
        title: "Chi-Square Analysis",
        xKey: "chi2",
        xLabel: "Chi² Score",
        description: "Higher Chi² values indicate stronger statistical independence between word frequency and text type",
        color: "#3b82f6"
      },
      secondary: {
        type: "scatter",
        title: "Chi² vs P-value",
        xKey: "chi2",
        yKey: "p_value",
        xLabel: "Chi² Score",
        yLabel: "P-value (log scale)",
        description: "Statistical significance visualization - lower p-values (bottom) with higher Chi² (right) are most significant",
        color: "#f59e0b"
      }
    },
    gensim: {
      primary: {
        type: "bar",
        title: "TF-IDF Analysis",
        xKey: "tfidf_score",
        xLabel: "TF-IDF Score",
        description: "Term Frequency-Inverse Document Frequency scores show words frequent in your text but rare in the corpus",
        color: "#ef4444"
      }
    },
    spacy: {
      primary: {
        type: "bar",
        title: "SpaCy Log-Likelihood Analysis",
        xKey: "log_likelihood",
        xLabel: "Log-Likelihood Score",
        description: "Primary keyness measure combining multiple statistical indicators for comprehensive linguistic analysis",
        color: "#10b981"
      },
      secondary: {
        type: "scatter",
        title: "Effect Size vs Chi²",
        xKey: "chi2",
        yKey: "effect_size",
        xLabel: "Chi² Score",
        yLabel: "Effect Size",
        description: "Comparison of practical significance (effect size) against statistical significance (Chi²)",
        color: "#8b5cf6"
      }
    }
  };

  const currentConfig = chartConfigs[method.toLowerCase()] || chartConfigs.nltk;
  const hasSecondaryChart = currentConfig.secondary !== undefined;
  const activeConfig = currentConfig[chartType] || currentConfig.primary;

  const chartData = useMemo(() => {
    if (!results || results.length === 0) return [];

    return topResults.map(r => {
      const uploaded = r.uploaded_count ?? r.count_a ?? r.uploaded_freq ?? 0;
      const sample = r.sample_count ?? r.count_b ?? r.sample_freq ?? 0;
      const word = r.word ?? 'Unknown';
      
      let xValue = 0;
      let yValue = 0;
      
      if (activeConfig.xKey) {
        xValue = r[activeConfig.xKey] ?? r.keyness_score ?? r.keyness ?? 0;
      }
      
      if (activeConfig.yKey) {
        yValue = r[activeConfig.yKey] ?? r.effect_size ?? 0;
      }

      return {
        word,
        uploaded,
        sample,
        keyness_score: r.keyness_score ?? r.keyness ?? 0,
        direction: r.direction ?? "Neutral",
        x: xValue,
        y: yValue,
        chi2: r.chi2 ?? 0,
        p_value: r.p_value ?? 1,
        effect_size: r.effect_size ?? 0,
        log_likelihood: r.log_likelihood ?? 0,
        tfidf_score: r.tfidf_score ?? 0
      };
    });
  }, [topResults, activeConfig, results]);

  if (!results || results.length === 0) {
    return (
      <div className="charts-container">
        <div className="no-data-container">
          <Info className="no-data-icon" />
          <h3 className="no-data-title">No Chart Data Available</h3>
          <p className="no-data-message">
            Charts will appear here once your keyness analysis is complete.
          </p>
        </div>
      </div>
    );
  }

  const createBarChart = () => {
    const colors = {
      uploaded: "#3b82f6",
      sample: "#10b981"
    };

    return [
      {
        x: chartData.map(d => d.word),
        y: chartData.map(d => d.uploaded),
        name: "Your Text",
        type: "bar",
        marker: { 
          color: colors.uploaded, 
          line: { color: "#1e40af", width: 1 },
          opacity: 0.8
        },
        hovertemplate: "<b>%{x}</b><br>" +
                      "Your Text: %{y}<br>" +
                      `${activeConfig.xLabel}: %{customdata:.3f}<extra></extra>`,
        customdata: chartData.map(d => d.x)
      },
      {
        x: chartData.map(d => d.word),
        y: chartData.map(d => d.sample),
        name: "Reference Corpus",
        type: "bar",
        marker: { 
          color: colors.sample, 
          line: { color: "#059669", width: 1 },
          opacity: 0.8
        },
        hovertemplate: "<b>%{x}</b><br>" +
                      "Corpus: %{y}<br>" +
                      `${activeConfig.xLabel}: %{customdata:.3f}<extra></extra>`,
        customdata: chartData.map(d => d.x)
      }
    ];
  };

  const createScatterChart = () => {
    const getPointColor = (point) => {
      if (method.toLowerCase() === 'sklearn' && point.p_value < 0.05) {
        return "#ef4444"; 
      } else if (point.keyness_score > 0) {
        return "#10b981"; 
      }
      return activeConfig.color;
    };

    return [{
      x: chartData.map(d => d.x),
      y: chartData.map(d => d.y),
      text: chartData.map(d => d.word),
      mode: "markers+text",
      type: "scatter",
      marker: { 
        size: chartData.map(d => Math.max(8, Math.min(16, d.uploaded / 2 + 8))),
        color: chartData.map(d => getPointColor(d)),
        line: { color: "#1e40af", width: 1.5 },
        opacity: 0.7,
        colorscale: "Viridis"
      },
      textposition: "top center",
      textfont: { size: 9, color: "#1e40af", family: "Inter, sans-serif" },
      hovertemplate: "<b>%{text}</b><br>" +
                     `${activeConfig.xLabel}: %{x:.3f}<br>` +
                     `${activeConfig.yLabel}: %{y:.3f}<br>` +
                     "Your Text: %{customdata.uploaded}<br>" +
                     "Corpus: %{customdata.sample}<br>" +
                     "<extra></extra>",
      customdata: chartData.map(d => ({
        uploaded: d.uploaded,
        sample: d.sample,
        keyness: d.keyness_score
      }))
    }];
  };

  const plotData = activeConfig.type === "bar" ? createBarChart() : createScatterChart();

  const getLayout = () => {
    const baseLayout = {
      title: { 
        text: activeConfig.title, 
        font: { size: 20, family: "Inter, sans-serif", color: "#1e293b" }, 
        x: 0.5,
        y: 0.95
      },
      paper_bgcolor: "rgba(255, 255, 255, 0.95)",
      plot_bgcolor: "rgba(248, 250, 252, 0.8)",
      margin: { l: 80, r: 80, t: 80, b: 100 },
      showlegend: true,
      legend: {
        orientation: "h",
        x: 0.5,
        xanchor: "center",
        y: -0.15,
        bgcolor: "rgba(255, 255, 255, 0.95)",
        bordercolor: "#e2e8f0",
        borderwidth: 1,
        font: { size: 12 }
      },
      hovermode: "closest",
      font: { family: "Inter, sans-serif" }
    };

      if (activeConfig.type === "bar") {
      return {
        ...baseLayout,
        xaxis: {
          title: { text: "Keywords", font: { size: 14, color: "#475569" } },
          tickangle: -45,
          tickfont: { size: 10, color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)",
          linecolor: "#e2e8f0"
        },
        yaxis: {
          title: { text: "Frequency Count", font: { size: 14, color: "#475569" } },
          tickfont: { color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)",
          linecolor: "#e2e8f0"
        },
        barmode: "group",
        bargap: 0.4,
        bargroupgap: 0.1
      };
    } else {
      return {
        ...baseLayout,
        xaxis: {
          title: { text: activeConfig.xLabel, font: { size: 14, color: "#475569" } },
          tickfont: { color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)",
          linecolor: "#e2e8f0",
          zeroline: true,
          zerolinecolor: "rgba(0, 0, 0, 0.3)"
        },
        yaxis: {
          title: { text: activeConfig.yLabel, font: { size: 14, color: "#475569" } },
          tickfont: { color: "#64748b" },
          gridcolor: "rgba(226, 232, 240, 0.6)",
          linecolor: "#e2e8f0",
          type: activeConfig.yKey === "p_value" ? "log" : "linear",
          zeroline: true,
          zerolinecolor: "rgba(0, 0, 0, 0.3)"
        }
      };
    }
  };

  return (
    <div className="charts-container">
      <div className="chart-header">
        <h3 className="chart-title">
          <TrendingUp className="chart-icon" />
          Keyness Visualization ({method.toUpperCase()})
        </h3>

        {hasSecondaryChart && (
          <div className="chart-toggle">
          <button
              onClick={() => setChartType(chartType === "primary" ? "secondary" : "primary")}
              className={`toggle-button ${chartType}`}
              title={`Switch to ${chartType === "primary" ? "scatter plot" : "bar chart"}`}
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
        <Info className="description-icon" />
        <p>{activeConfig.description}</p>
      </div>

      <div className="chart-controls">
        <div className="chart-slider">
          <label htmlFor="topNSlider">
            Displaying top <strong>{topN}</strong> keywords
          </label>
          <input
            id="topNSlider"
            type="range"
            min={5}
            max={Math.min(50, results.length)}
            step={1}
            value={topN}
            onChange={(e) => setTopN(parseInt(e.target.value))}
            className="slider"
          />
          <span className="slider-value">{topN}/{results.length}</span>
        </div>
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
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            toImageButtonOptions: {
              format: 'png',
              filename: `${method}_keyness_${chartType}_analysis`,
              height: 800,
              width: 1200,
              scale: 2
            },
            responsive: true
          }}
        />
                    </div>

      {/* summary */}
      <div className="chart-summary">
        <h4>AI Chart Summary</h4>
        <p>{summary}</p>
                  </div>

      <div className="chart-footer">
        <div className="method-info">
          <strong>Analysis:</strong> {method.toUpperCase()} | 
          <strong> Visualization:</strong> {activeConfig.type === "bar" ? "Frequency Comparison" : "Statistical Relationship"} |
          <strong> Data Points:</strong> {chartData.length}
    </div>
  </div>
    </div>
  );
};

export default Charts;