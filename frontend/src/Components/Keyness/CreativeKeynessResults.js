import React, { useState, useMemo, useEffect } from "react";
import Charts from "./Charts";
import ResultsTable from "./ResultsTable";
import ResultsSummary from "./ResultsSummary";
import KeynessResultsGrid from "./KeynessResultsGrid";
import "./CreativeKeynessResults.css";
import { exportKeynessToXlsx } from "./ExportXlsx";
import axios from "axios";

const posColors = {
  NOUN: "noun",
  VERB: "verb",
  ADJ: "adj",
  ADV: "adv",
  OTHER: "other",
};

const CreativeKeynessResults = ({ results, stats, method, uploadedText, genre, onWordDetail, onChangeMethod, loading }) => {
  console.log("onChangeMethod prop:", onChangeMethod);
  console.log("onChangeMethod type:", typeof onChangeMethod);
  const [activeView, setActiveView] = useState("keywords");
  const [summary, setSummary] = useState("");
  const [summaryLoading, setSummaryLoading] = useState(false);

  const [chartSummaries, setChartSummaries] = useState({
    primary: { summary: "", loading: false, error: null },
    secondary: { summary: "", loading: false, error: null }
  });
  const [activeChartType, setActiveChartType] = useState("primary"); 

  const safeResults = Array.isArray(results) ? results : [];

  // Compute chart data for both primary and secondary charts
  const chartData = useMemo(() => {
    if (!safeResults || safeResults.length === 0) return { primary: [], secondary: [] };

    const primaryData = safeResults.slice(0, 20).map((r) => ({
      label: r.word,
      value: r.keyness ?? r.log_likelihood ?? r.chi2 ?? r.tfidf_score ?? 0,
    }));

    const secondaryData = safeResults.slice(0, 30).map((r) => ({
      label: r.word,
      x: r.frequency || r.count || 0,
      y: r.keyness ?? r.log_likelihood ?? r.chi2 ?? r.tfidf_score ?? 0,
    }));

    return { primary: primaryData, secondary: secondaryData };
  }, [safeResults]);

  const fetchChartSummary = async (chartType, data, forceRefresh = false) => {
    if (chartSummaries[chartType].summary && !forceRefresh) return;

    setChartSummaries(prev => ({
      ...prev,
      [chartType]: { ...prev[chartType], loading: true, error: null }
    }));

    try {
      const payload = chartType === "primary"
        ? {
          title: `${method.toUpperCase()} Keyness Analysis - Top Keywords`,
          chart_type: "bar",
          chart_data: data,
        }
        : {
          title: `${method.toUpperCase()} Keyness Analysis - Frequency vs Keyness`,
          chart_type: "scatter",
          chart_data: data,
        };

      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/summarise-keyness-chart/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const responseData = await response.json();

      setChartSummaries(prev => ({
        ...prev,
        [chartType]: {
          summary: responseData.analysis || "No summary available.",
          loading: false,
          error: null
        }
      }));
    } catch (err) {
      console.error(`Error fetching ${chartType} chart summary:`, err);
      setChartSummaries(prev => ({
        ...prev,
        [chartType]: {
          summary: "",
          loading: false,
          error: `Failed to fetch ${chartType} chart summary.`
        }
      }));
    }
  };

  // ✅ Effect to fetch chart summaries when Charts tab becomes active
  useEffect(() => {
    if (activeView !== "charts" || chartData.primary.length === 0) return;

    // Fetch primary chart summary immediately
    fetchChartSummary("primary", chartData.primary);

    // Pre-fetch secondary chart summary in background (optional)
    if (chartData.secondary.length > 0) {
      setTimeout(() => {
        fetchChartSummary("secondary", chartData.secondary);
      }, 1000); // Delay to not overwhelm the API
    }
  }, [activeView, chartData, method]);

  // Group by POS and filter to only words from uploaded text
  const uploadedWordsSet = useMemo(() => {
    if (!uploadedText) return new Set();
    return new Set(
      uploadedText
        .toLowerCase()
        .match(/\b\w+\b/g)
    );
  }, [uploadedText]);

  const posGroups = useMemo(() => {
    const groups = {};
    safeResults.forEach((r) => {
      if (!uploadedWordsSet.has(r.word.toLowerCase())) return;
      if (r.pos === "PROPN") return; // Skip proper nouns

      const pos = (r.pos || r.pos_tag || "OTHER").toUpperCase();
      if (!groups[pos]) groups[pos] = [];
      groups[pos].push(r);
    });
    return groups;
  }, [safeResults, uploadedWordsSet]);

  // Fetch summary for summary tab
  const fetchSummary = async () => {
    setSummaryLoading(true);
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/get-keyness-summary/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keyness_results: results }),
      });
      const data = await response.json();
      setSummary(data.summary || "No summary available");
    } catch (err) {
      setSummary("Error retrieving summary");
    } finally {
      setSummaryLoading(false);
    }
  };

  React.useEffect(() => {
    if (activeView === "summary" && !summary && !summaryLoading) {
      fetchSummary();
    }
  }, [activeView]);

  // Handle clicking on a keyword
  const handleKeywordClick = (w) => {
    if (!w) return;
    const wordData = safeResults.find(item => item.word?.toLowerCase() === w.word.toLowerCase());
    if (wordData && onWordDetail) {
      onWordDetail({
        word: w.word,
        wordData,
        uploadedText,
        method,
        results: safeResults
      });
    }
  };

  // Handle chart type change
  const handleChartTypeChange = (chartType) => {
    setActiveChartType(chartType);

    // Fetch summary for the newly active chart if not already loaded
    const currentChartData = chartType === "primary" ? chartData.primary : chartData.secondary;
    if (!chartSummaries[chartType].summary && !chartSummaries[chartType].loading && currentChartData.length > 0) {
      fetchChartSummary(chartType, currentChartData);
    }
  };

  if (!results || Object.keys(results).length === 0) {
    return (
      <div className="results-container">
        <div className="no-results">
          No significant keywords found.
        </div>
      </div>
    );
  }

  const chartDataForExport = results?.slice(0, 20).map((r) => ({
    label: r.word,
    value: r.keyness ?? r.log_likelihood ?? r.chi2 ?? r.tfidf_score ?? 0,
  }));
  console.log("onChangeMethod prop:", onChangeMethod);


  const viewLabels = {
    keywords: "Keywords",
    charts: "Charts",
    table: "Table",
    summary: "Summary"
  };

  return (
    <div className="results-container">
      <div className="current-analysis-info">
        <span className="current-analysis-text">
          Analysing with <strong>{method?.toUpperCase()}</strong>
        </span>
        <button
          onClick={() => onChangeMethod && onChangeMethod()}
          className="change-method-button"
          disabled={loading}
        >
          Change Method
        </button>
      </div>

      <ResultsSummary
        stats={stats}
        selectedMethod={method}
        comparisonResults={safeResults}
        genre={genre}
      />

      {/* Main View Toggle Buttons */}
      <div className="view-controls">
        {Object.keys(viewLabels).map((view) => (
          <button
            key={view}
            className={`btn ${activeView === view ? "bg-blue-500 text-white" : ""}`}
            onClick={() => setActiveView(view)}
          >
            {viewLabels[view]}
          </button>
        ))}

        {/* Download button */}
        <button
          className="btn bg-green-500 text-white"
          onClick={() =>
            exportKeynessToXlsx(
              safeResults,
              method,
              stats,
              posGroups,
              [],
              chartDataForExport
            )
          }
        >
          Download XLSX
        </button>
      </div>

      {/* Summary View */}
      {activeView === "summary" && (
        <div className="keyness-summary">
          {summaryLoading
            ? "Loading summary..."
            : summary.split(/\n{2,}|(?<=\.)\s+/).map((p, i) => (
              <p key={i}>{p.trim()}</p>
            ))}
        </div>
      )}

      {/* Keywords View */}
      {activeView === "keywords" && (
        <div className="creative-results">
          <div className="keywords-header">
            <h2>Top 50 Most Significant Keywords from Your Text</h2>
            <p>Click on a word to see detailed analysis including sentences, synonyms, and more.</p>
          </div>

          {Object.keys(posGroups).length === 0 ? (
            <div className="no-keywords">No significant keywords found</div>
          ) : (
            <>
              {Object.entries(posGroups).map(([pos, words]) => {
                const posFullNames = {
                  ADV: "Adverbs",
                  NOUN: "Nouns",
                  VERB: "Verbs",
                  ADJ: "Adjectives",
                  OTHER: "Other Words",
                };
                const posLabel = posFullNames[pos] || pos;
                const posKey = pos.toLowerCase();

                return (
                  <div key={pos} className="pos-section">
                    <h3 data-pos={posKey}>{posLabel}</h3>
                    <div className="word-list">
                      {words.map((w, idx) => (
                        <span
                          key={`${w.word}-${idx}`}
                          className={`keyword keyword-pill ${posColors[w.pos] || posColors.OTHER}`}
                          onClick={() => handleKeywordClick(w)}
                          tabIndex={0}
                          role="button"
                          onKeyPress={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.preventDefault();
                              handleKeywordClick(w);
                            }
                          }}
                          title={`Click for detailed analysis of "${w.word}"`}
                        >
                          {w.word}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}

      {/* Charts View */}
      {activeView === "charts" && (
        <div className="charts-container">
          {/* Charts Component with callback for chart type changes */}
          <Charts
            results={safeResults}
            method={method}
            onChartTypeChange={handleChartTypeChange}
          />

          {/* Enhanced AI Summary Section */}
          <div className="chart-summary-section mt-6">
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6 shadow-sm">
              <div className="flex items-center mb-4">
                <div className="w-2 h-6 bg-blue-500 rounded-full mr-3"></div>
                <h4 className="font-semibold text-xl text-gray-800">
                  What this chart shows: {activeChartType === "primary" ? "Top Keywords Chart" : "Frequency vs Keyness Chart"}
                </h4>
              </div>

              <div className="chart-summary-content">
                {chartSummaries[activeChartType].loading ? (
                  <div className="flex items-center space-x-2 text-blue-600">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <p>Analysing chart data...</p>
                  </div>
                ) : chartSummaries[activeChartType].error ? (
                  <div className="bg-red-50 border border-red-200 rounded-md p-4">
                    <p className="text-red-700">{chartSummaries[activeChartType].error}</p>
                    <button
                      className="mt-2 text-sm bg-red-100 hover:bg-red-200 text-red-800 px-3 py-1 rounded"
                      onClick={() => fetchChartSummary(activeChartType, activeChartType === "primary" ? chartData.primary : chartData.secondary, true)}
                    >
                      Retry Analysis
                    </button>
                  </div>
                ) : chartSummaries[activeChartType].summary ? (
                  <div className="prose prose-sm max-w-none">
                    <div className="text-gray-700 leading-relaxed whitespace-pre-line">
                      {chartSummaries[activeChartType].summary}
                    </div>
                  </div>
                ) : (
                  <p className="text-gray-500 italic">No analysis available yet.</p>
                )}
              </div>

              {/* Refresh button */}
              {chartSummaries[activeChartType].summary && !chartSummaries[activeChartType].loading && (
                <div className="mt-4 pt-4 border-t border-blue-100">
                  <button
                    className="text-sm bg-blue-100 hover:bg-blue-200 text-blue-800 px-4 py-2 rounded-md transition-colors"
                    onClick={() => fetchChartSummary(activeChartType, activeChartType === "primary" ? chartData.primary : chartData.secondary, true)}
                  >
                    Refresh Analysis
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Table View */}
      {activeView === "table" && <ResultsTable results={safeResults} method={method} />}
    </div>
  );
};

export default CreativeKeynessResults;