import React, { useState, useEffect } from "react";
import ClusteringCharts from "./ClusteringCharts";
import "./CreativeClusteringAnalysis.css";

const CreativeClusteringAnalysis = ({ clusters, topTerms, themes, textDocuments }) => {
    const [showChart, setShowChart] = useState(true);
    const [selectedCluster, setSelectedCluster] = useState("all");
    const [showTopTerms, setShowTopTerms] = useState(false);
    const [showDocuments, setShowDocuments] = useState(false);
    const [showSummary, setShowSummary] = useState(false);
    const [showThemes, setShowThemes] = useState(false);
    const [themeAnalysisData, setThemeAnalysisData] = useState(null);
    const [isLoadingThemeAnalysis, setIsLoadingThemeAnalysis] = useState(false);
    const [themeAnalysisError, setThemeAnalysisError] = useState(null);
    const [chartSummaryData, setChartSummaryData] = useState(null);
    const [isLoadingChartSummary, setIsLoadingChartSummary] = useState(false);
    const [chartSummaryError, setChartSummaryError] = useState(null);

    // Function to call the clustering summary API
    const generateChartSummary = async () => {
        setIsLoadingChartSummary(true);
        setChartSummaryError(null);
        
        try {
            const response = await fetch("http://localhost:8000/api/summarise-clustering-chart/", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    clusters: clusters,
                    top_terms: topTerms,
                    themes: themes,
                    selected_cluster: selectedCluster,
                    title: `Clustering Analysis - ${selectedCluster === 'all' ? 'All Clusters' : `Cluster ${selectedCluster}`}`
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            setChartSummaryData(data);
            
        } catch (error) {
            console.error('Error generating summary:', error);
            setChartSummaryError(error.message);
        } finally {
            setIsLoadingChartSummary(false);
        }
    };

    const generateThemeAnalysis = async () => {
    setIsLoadingThemeAnalysis(true);
    setThemeAnalysisError(null);
    
    try {
        const response = await fetch("http://localhost:8000/api/analyse-themes/", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text_documents: textDocuments,
                clusters: clusters,
                top_terms: topTerms,
                themes: themes,
                title: 'Document Collection Theme Analysis'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setThemeAnalysisData(data);
        
    } catch (error) {
        console.error('Error generating theme analysis:', error);
        setThemeAnalysisError(error.message);
    } finally {
        setIsLoadingThemeAnalysis(false);
    }
};

useEffect(() => {
    console.log('Theme tab shown:', showThemes);
    console.log('Text documents count:', textDocuments?.length);
    console.log('Theme analysis data:', themeAnalysisData);
    console.log('Loading:', isLoadingThemeAnalysis);
    console.log('Error:', themeAnalysisError);
    if (showThemes && !themeAnalysisData && !isLoadingThemeAnalysis && !themeAnalysisError) {
        generateThemeAnalysis();
    }
}, [showThemes]);

    // Auto-generate chart summary when showing chart or cluster selection changes
    useEffect(() => {
        if (showChart && clusters.length > 0) {
            generateChartSummary();
        }
    }, [showChart, selectedCluster, clusters.length]);

    // Function for general summary (placeholder for later)
    const generateGeneralSummary = async () => {
       
    };

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
        setShowThemes(view === 'themes');
        setShowSummary(view === 'summary');

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
                    className={`clustering-btn btn-themes ${showThemes ? 'active' : ''}`}
                    onClick={() => handleViewChange('themes')}
                >
                    Themes
                </button>

                {/* Summary Button */}
                <button
                    className={`clustering-btn btn-summary ${showSummary ? 'active' : ''}`}
                    onClick={() => handleViewChange('summary')}
                >
                    General Summary
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

            {/* Chart View with integrated summary */}
            {showChart && clusters.length > 0 && (
                <div className="chart-section">
                    <ClusteringCharts 
                        clusters={clusters} 
                        selectedCluster={selectedCluster} 
                    />
                    
                    {/* Chart Summary Section */}
                    <div className="chart-summary-section">
                        <div className="chart-summary-header">
                            <h3 className="chart-summary-title">Chart Analysis</h3>
                            <button
                                className={`clustering-btn btn-refresh-small ${isLoadingChartSummary ? 'loading' : ''}`}
                                onClick={generateChartSummary}
                                disabled={isLoadingChartSummary}
                                title="Regenerate analysis"
                            >
                                {isLoadingChartSummary ? '⟳' : '↻'}
                            </button>
                        </div>
                        
                        {isLoadingChartSummary && (
                            <div className="chart-summary-loading">
                                <div className="loading-spinner"></div>
                                <p>Analyzing clustering results...</p>
                            </div>
                        )}
                        
                        {chartSummaryError && (
                            <div className="chart-summary-error">
                                <p>Error generating analysis: {chartSummaryError}</p>
                                <button onClick={generateChartSummary} className="clustering-btn btn-small">
                                    Try Again
                                </button>
                            </div>
                        )}
                        
                        {chartSummaryData && !isLoadingChartSummary && (
                            <div className="chart-summary-content">
                                <div className="chart-summary-meta">
                                    <span>Scope: {chartSummaryData.analysis_scope}</span>
                                    <span>Documents: {chartSummaryData.total_documents}</span>
                                    <span>Clusters: {chartSummaryData.num_clusters}</span>
                                </div>
                                <div className="chart-summary-text">
                                    <pre>{chartSummaryData.analysis}</pre>
                                </div>
                            </div>
                        )}
                    </div>
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

            {showThemes && (
    <div className="themes-section">
        <div className="themes-header">
            <h2 className="themes-title">Theme Analysis</h2>
            <button
                className={`clustering-btn btn-refresh ${isLoadingThemeAnalysis ? 'loading' : ''}`}
                onClick={generateThemeAnalysis}
                disabled={isLoadingThemeAnalysis}
            >
                {isLoadingThemeAnalysis ? 'Analyzing...' : 'Regenerate'}
            </button>
        </div>
        
        {isLoadingThemeAnalysis && (
            <div className="themes-loading">
                <div className="loading-spinner"></div>
                <p>Analyzing themes and topics in your document collection...</p>
            </div>
        )}
        
        {themeAnalysisError && (
            <div className="themes-error">
                <p>Error generating theme analysis: {themeAnalysisError}</p>
                <button onClick={generateThemeAnalysis} className="clustering-btn">
                    Try Again
                </button>
            </div>
        )}
        
        {themeAnalysisData && !isLoadingThemeAnalysis && (
            <div className="themes-content">
                <div className="themes-meta">
                    <span>Data Source: {themeAnalysisData.data_source}</span>
                    <span>Total Documents: {themeAnalysisData.total_documents}</span>
                    <span>Analyzed: {themeAnalysisData.documents_analyzed}</span>
                    {themeAnalysisData.has_clustering_context && (
                        <span>Clustering Context: Available</span>
                    )}
                </div>
                <div className="themes-analysis">
                    <pre>{themeAnalysisData.analysis}</pre>
                </div>
            </div>
        )}
        
        {!themeAnalysisData && !themeAnalysisError && !isLoadingThemeAnalysis && (
            <div className="themes-placeholder">
                <p>Theme analysis will automatically generate when you first view this tab.</p>
                <p>This analysis identifies dominant themes, topics, and conceptual patterns in your document collection.</p>
            </div>
        )}
    </div>
)}

            {/* General Summary View (placeholder) */}
            {showSummary && (
                <div className="summary-section">
                    <div className="summary-header">
                        <h2 className="summary-title">General Analysis Summary</h2>
                        <button
                            className="clustering-btn btn-generate"
                            onClick={generateGeneralSummary}
                        >
                            Coming Soon
                        </button>
                    </div>
                    
                    <div className="summary-placeholder">
                        <p>General summary functionality will be implemented here.</p>
                        <p>This will provide an overall analysis of your clustering results across all views.</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CreativeClusteringAnalysis;

