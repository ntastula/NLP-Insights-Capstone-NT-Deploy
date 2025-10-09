import React, { useState, useEffect } from "react";
import ClusteringCharts from "./ClusteringCharts";
import "./CreativeClusteringAnalysis.css";

const CreativeClusteringAnalysis = ({ clusters, topTerms, themes, textDocuments = [] }) => {
    const [showChart, setShowChart] = useState(true);
    const [selectedCluster, setSelectedCluster] = useState("all");
    const [showTopTerms, setShowTopTerms] = useState(false);
    const [showDocuments, setShowDocuments] = useState(false);
    const [showThemes, setShowThemes] = useState(false);
    const [showThematicFlow, setShowThematicFlow] = useState(false);
    const [showOverusedThemes, setShowOverusedThemes] = useState(false);
    const [themeAnalysisData, setThemeAnalysisData] = useState(null);
    const [isLoadingThemeAnalysis, setIsLoadingThemeAnalysis] = useState(false);
    const [themeAnalysisError, setThemeAnalysisError] = useState(null);
    const [thematicFlowData, setThematicFlowData] = useState(null);
    const [isLoadingThematicFlow, setIsLoadingThematicFlow] = useState(false);
    const [thematicFlowError, setThematicFlowError] = useState(null);
    const [overusedThemesData, setOverusedThemesData] = useState(null);
    const [isLoadingOverusedThemes, setIsLoadingOverusedThemes] = useState(false);
    const [overusedThemesError, setOverusedThemesError] = useState(null);
    const [chartSummaryData, setChartSummaryData] = useState(null);
    const [isLoadingChartSummary, setIsLoadingChartSummary] = useState(false);
    const [chartSummaryError, setChartSummaryError] = useState(null);

    const generateChartSummary = async () => {
        setIsLoadingChartSummary(true);
        setChartSummaryError(null);

        try {
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/summarise-clustering-chart/`, {
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

    // Auto-generate chart summary when showing chart or cluster selection changes
    useEffect(() => {
        if (showChart && clusters.length > 0) {
            generateChartSummary();
        }
    }, [showChart, selectedCluster, clusters.length]);

    // Function for general summary (placeholder for later)
    const generateGeneralSummary = async () => {
    };

    const generateThemeAnalysis = async () => {
        setIsLoadingThemeAnalysis(true);
        setThemeAnalysisError(null);

        try {
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/analyse-themes/`, {
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
        if (showThemes && !themeAnalysisData && !isLoadingThemeAnalysis && !themeAnalysisError) {
            generateThemeAnalysis();
        }
    }, [showThemes]);

    // Function for thematic flow analysis
    const generateThematicFlow = async () => {
        setIsLoadingThematicFlow(true);
        setThematicFlowError(null);

        try {
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/analyse-thematic-flow/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text_documents: textDocuments,
                    clusters: clusters,
                    top_terms: topTerms,
                    themes: themes,
                    title: 'Thematic Flow Analysis'
                })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setThematicFlowData(data);
        } catch (error) {
            console.error('Error generating thematic flow:', error);
            setThematicFlowError(error.message);
        } finally {
            setIsLoadingThematicFlow(false);
        }
    };

    // Auto-generate thematic flow when tab is first viewed
    useEffect(() => {
        if (showThematicFlow && !thematicFlowData && !isLoadingThematicFlow && !thematicFlowError) {
            generateThematicFlow();
        }
    }, [showThematicFlow]);

    // Function for overused themes analysis
    const generateOverusedThemes = async () => {
        setIsLoadingOverusedThemes(true);
        setOverusedThemesError(null);

        try {
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/analyse-overused-themes/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text_documents: textDocuments,
                    clusters: clusters,
                    top_terms: topTerms,
                    themes: themes,
                    title: 'Overused/Underused Analysis'
                })
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            setOverusedThemesData(data);
        } catch (error) {
            console.error('Error generating overused themes analysis:', error);
            setOverusedThemesError(error.message);
        } finally {
            setIsLoadingOverusedThemes(false);
        }
    };

    // Auto-generate overused themes when tab is first viewed
    useEffect(() => {
        if (showOverusedThemes && !overusedThemesData && !isLoadingOverusedThemes && !overusedThemesError) {
            generateOverusedThemes();
        }
    }, [showOverusedThemes]);

    // Unique cluster labels for dropdown
    const clusterOptions = Array.from(new Set(clusters.map(c => c.label))).sort(
        (a, b) => a - b
    );

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
        setShowThematicFlow(view === 'flow');
        setShowOverusedThemes(view === 'overused');
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

                {/* Themes Button */}
                <button
                    className={`clustering-btn btn-themes ${showThemes ? 'active' : ''}`}
                    onClick={() => handleViewChange('themes')}
                >
                    Themes
                </button>

                {/* Thematic Flow Button */}
                <button
                    className={`clustering-btn btn-flow ${showThematicFlow ? 'active' : ''}`}
                    onClick={() => handleViewChange('flow')}
                >
                    Thematic Flow
                </button>

                {/* Overused Themes Button */}
                <button
                    className={`clustering-btn btn-overused ${showOverusedThemes ? 'active' : ''}`}
                    onClick={() => handleViewChange('overused')}
                >
                    Overused Themes
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
                                <p>Analysing clustering results...</p>
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

            {/* Themes View */}
            {showThemes && (
                <div className="themes-section">
                    <div className="themes-header">
                        <h2 className="themes-title">Theme Analysis</h2>
                        <button
                            className={`clustering-btn btn-refresh ${isLoadingThemeAnalysis ? 'loading' : ''}`}
                            onClick={generateThemeAnalysis}
                            disabled={isLoadingThemeAnalysis}
                        >
                            {isLoadingThemeAnalysis ? 'Analysing...' : 'Regenerate'}
                        </button>
                    </div>

                    {isLoadingThemeAnalysis && (
                        <div className="themes-loading">
                            <div className="loading-spinner"></div>
                            <p>Analysing themes and topics in your document collection...</p>
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
                                <span>Analysed: {themeAnalysisData.documents_analysed}</span>
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

            {/* Thematic Flow View */}
            {showThematicFlow && (
                <div className="flow-section">
                    <div className="flow-header">
                        <h2 className="flow-title">Thematic Flow Analysis</h2>
                        <button
                            className={`clustering-btn btn-refresh ${isLoadingThematicFlow ? 'loading' : ''}`}
                            onClick={generateThematicFlow}
                            disabled={isLoadingThematicFlow}
                        >
                            {isLoadingThematicFlow ? 'Analysing...' : 'Regenerate'}
                        </button>
                    </div>

                    {isLoadingThematicFlow && (
                        <div className="flow-loading">
                            <div className="loading-spinner"></div>
                            <p>Analysing thematic relationships and flow patterns...</p>
                        </div>
                    )}

                    {thematicFlowError && (
                        <div className="flow-error">
                            <p>Error generating thematic flow analysis: {thematicFlowError}</p>
                            <button onClick={generateThematicFlow} className="clustering-btn">
                                Try Again
                            </button>
                        </div>
                    )}

                    {thematicFlowData && !isLoadingThematicFlow && (
                        <div className="flow-content">
                            <div className="flow-meta">
                                <span>Data Source: {thematicFlowData.data_source}</span>
                                <span>Total Documents: {thematicFlowData.total_documents}</span>
                                <span>Analysed: {thematicFlowData.documents_analysed}</span>
                                {thematicFlowData.has_clustering_context && (
                                    <span>Clustering Context: Available</span>
                                )}
                            </div>
                            <div className="flow-analysis">
                                <pre>{thematicFlowData.analysis}</pre>
                            </div>
                        </div>
                    )}

                    {!thematicFlowData && !thematicFlowError && !isLoadingThematicFlow && (
                        <div className="flow-placeholder">
                            <p>Thematic flow analysis will automatically generate when you first view this tab.</p>
                            <p>This analysis examines how themes interconnect, develop, and flow throughout your documents.</p>
                        </div>
                    )}
                </div>
            )}

            {/* Overused Themes View */}
            {showOverusedThemes && (
                <div className="overused-section">
                    <div className="overused-header">
                        <h2 className="overused-title">Overused/Underused Analysis</h2>
                        <button
                            className={`clustering-btn btn-refresh ${isLoadingOverusedThemes ? 'loading' : ''}`}
                            onClick={generateOverusedThemes}
                            disabled={isLoadingOverusedThemes}
                        >
                            {isLoadingOverusedThemes ? 'Analysing...' : 'Regenerate'}
                        </button>
                    </div>

                    {isLoadingOverusedThemes && (
                        <div className="overused-loading">
                            <div className="loading-spinner"></div>
                            <p>Analysing patterns of overuse and underuse...</p>
                        </div>
                    )}

                    {overusedThemesError && (
                        <div className="overused-error">
                            <p>Error generating overused themes analysis: {overusedThemesError}</p>
                            <button onClick={generateOverusedThemes} className="clustering-btn">
                                Try Again
                            </button>
                        </div>
                    )}

                    {overusedThemesData && !isLoadingOverusedThemes && (
                        <div className="overused-content">
                            <div className="overused-meta">
                                <span>Data Source: {overusedThemesData.data_source}</span>
                                <span>Total Documents: {overusedThemesData.total_documents}</span>
                                <span>Analysed: {overusedThemesData.documents_analysed}</span>
                                {overusedThemesData.has_clustering_context && (
                                    <span>Clustering Context: Available</span>
                                )}
                            </div>
                            <div className="overused-analysis">
                                <pre>{overusedThemesData.analysis}</pre>
                            </div>
                        </div>
                    )}

                    {!overusedThemesData && !overusedThemesError && !isLoadingOverusedThemes && (
                        <div className="overused-placeholder">
                            <p>Overused/underused analysis will automatically generate when you first view this tab.</p>
                            <p>This analysis identifies repetitive patterns, overused words/phrases, and areas needing more development.</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default CreativeClusteringAnalysis;

