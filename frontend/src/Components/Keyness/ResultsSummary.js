import React, { useState, useEffect } from "react";
import { BarChart3, BookOpen, X, Info } from "lucide-react";
import "./ResultsSummary.css";

const ResultsSummary = ({ stats, selectedMethod, comparisonResults, genre }) => {
  const [showCorpusModal, setShowCorpusModal] = useState(false);
  const [corpusData, setCorpusData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const openCorpusModal = async () => {

    if (!genre) {
      setError("No genre specified");
      setShowCorpusModal(true);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const backendURL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

      const url = `${backendURL}/api/corpus-meta-keyness/?name=${encodeURIComponent(genre)}`;

      const response = await fetch(url, { credentials: "include" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      setCorpusData(data);
      setShowCorpusModal(true);
    } catch (err) {
      setError(err.message);
      setShowCorpusModal(true);
    } finally {
      setLoading(false);
    }
  };


  const closeModal = () => {
    setShowCorpusModal(false);
    setCorpusData(null);
    setError(null);
  };

  if (!stats) return null;

  return (
    <>
      <div className="results-summary">
        <div className="summary-header">
          <h3 className="summary-title">
            <BarChart3 className="summary-icon" />
            {selectedMethod ? `${selectedMethod} Keyness Analysis Results` : "Keyness Analysis Results"}
          </h3>
        </div>

        <div className="stats-grid">
          <div className="stat-card user-text-card">
            <div className="stat-number user-text-number">
              {stats.uploadedTotal?.toLocaleString()}
            </div>
            <div className="stat-label">Words in your text</div>
          </div>

          <div className="stat-card corpus-card">
            <div className="stat-number corpus-number">
              {stats.corpusTotal?.toLocaleString()}
            </div>
            <div className="stat-label">
              Words in sample{" "}
              <button
                className="corpus-button"
                onClick={openCorpusModal}
                disabled={loading}
                title="Click to see corpus details"
              >
                <BookOpen className="corpus-icon" />
                {loading ? "Loading..." : "Corpus"}
              </button>
            </div>
          </div>

          <div className="stat-card keywords-card">
            <div className="stat-number keywords-number">
              {(stats.totalSignificant || 0).toLocaleString()}
            </div>
            <div className="stat-label">Significant keywords</div>
          </div>
        </div>
      </div>

      {/* Corpus Modal */}
      {showCorpusModal && (
        <div className="corpus-modal-overlay" onClick={closeModal}>
          <div className="corpus-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h4 className="modal-title">
                <BookOpen className="modal-icon" />
                Sample Corpus Details
              </h4>
              <button className="modal-close" onClick={closeModal} title="Close">
                <X />
              </button>
            </div>

            <div className="modal-content">
              {error ? (
                <div className="error-message">
                  <Info className="error-icon" />
                  <p>Error loading corpus data: {error}</p>
                </div>
              ) : corpusData ? (
                <>
                  <div className="corpus-info">
                    <p className="corpus-genre">
                      <strong>Genre:</strong>{" "}
                      {corpusData.genre
                        ? corpusData.genre
                          .replace(/_keyness$/, "")
                          .replace(/_/g, " ")
                          .replace(/\b\w/g, (c) => c.toUpperCase())
                        : "General"}
                    </p>
                  </div>

                  <div className="books-list">
                    <h5 className="books-title">Books in this corpus:</h5>
                    <div className="books-grid">
                      {corpusData.previews?.map((book, index) => (
                        <div key={index} className="book-item">
                          <div className="book-title">{book.title}</div>
                          <div className="book-author">by {book.author}</div>
                        </div>
                      )) || <p className="no-books">No books found in this corpus.</p>}
                    </div>
                  </div>
                </>
              ) : (
                <div className="loading-message">
                  <div className="loading-spinner"></div>
                  <p>Loading corpus information...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ResultsSummary;