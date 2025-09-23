import React, { useEffect, useMemo, useState } from "react";
import "./HomePage.css";

const HomePage = ({ onSelect, selectedGenre, onSelectGenre, onProceed }) => {
  const [corpora, setCorpora] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const [localGenre, setLocalGenre] = useState("");
  const [analysisType, setAnalysisType] = useState("");

  // Fetch corpora list from backend
  useEffect(() => {
    if (!analysisType) return; // don't fetch before user selects analysis type

    let cancelled = false;

    (async () => {
      try {
        setLoading(true);
        setErr("");
        const r = await fetch(`http://localhost:8000/api/corpora/?analysis=${analysisType}`, { credentials: "include" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (cancelled) return;

        const list = Array.isArray(d.corpora) ? d.corpora : [];
        setCorpora(list);

        // Set default genre if none selected yet
        if (!localGenre && list.length) {
          let defaultFile = list[0];
          if (analysisType === "keyness") {
            defaultFile = list.find(f => f.startsWith("general_english")) || list[0];
          }
          setLocalGenre(defaultFile);
        }
      } catch (e) {
        if (!cancelled) setErr(String(e.message || e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [analysisType]);

  const handleAnalysisChange = (e) => {
    const type = e.target.value;
    setAnalysisType(type);
    setLocalGenre(""); // reset genre when analysis changes
  };

  const handleGenreChange = (e) => {
    setLocalGenre(e.target.value);
    if (typeof onSelectGenre === "function") onSelectGenre(e.target.value);
  };

  const filteredCorpora = useMemo(() => {
    if (!Array.isArray(corpora)) return [];

    return corpora.filter(file => {
      if (analysisType === "keyness") return true; // all valid
      if (analysisType === "sentiment" || analysisType === "sensorimotor") return true;
      return false; // clustering: no dropdown
    });
  }, [analysisType, corpora]);

  return (
    <div className="homepage-container">
      <div className="homepage-card">
        <h1 className="homepage-title">Welcome to TTC Writing Analysis</h1>

        {/* Analysis Selection */}
        <div className="homepage-section">
          <label className="homepage-label">
            What type of analysis would you like to do?
          </label>
          <select
            onChange={handleAnalysisChange}
            value={analysisType}
            className="homepage-select"
          >
            <option value="" disabled>Select analysis type…</option>
            <option value="keyness">Keyness</option>
            <option value="sentiment">Sentiment</option>
            <option value="clustering">Clustering</option>
            <option value="sensorimotor">Sensorimotor Norms</option>
          </select>
        </div>

        {/* Genre Selection + Button for non-clustering */}
        {analysisType && analysisType !== "clustering" && (
          <>
            <div className="homepage-section">
              <label className="homepage-label">
                Genre Corpus
              </label>
              {loading ? (
                <div className="homepage-loading">Loading genres...</div>
              ) : err ? (
                <div className="homepage-error">Error loading genres: {err}</div>
              ) : (
                <select
                  value={localGenre}
                  onChange={handleGenreChange}
                  className="homepage-select"
                  disabled={loading || !!err || filteredCorpora.length === 0}
                >
                  {filteredCorpora.map((file) => {
                    // Remove "_keyness" and ".json"
                    let displayName = file.replace(/_keyness$/, "").replace(/\.json$/, "");
                    
                    // Replace underscores with spaces
                    displayName = displayName.replace(/_/g, " ");
                    
                    // Capitalise each word
                    displayName = displayName.replace(/\b\w/g, (c) => c.toUpperCase());

                    return (
                      <option key={file} value={file}>
                        {displayName}
                      </option>
                    );
                  })}
                </select>
              )}
            </div>

            <button
              onClick={() => onProceed({ analysisType, genre: localGenre })}
              className="homepage-button"
              disabled={!localGenre || loading || !!err}
            >
              Go to {analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} Analysis
            </button>
          </>
        )}

        {/* Button for clustering */}
        {analysisType === "clustering" && (
          <button
            onClick={() => onProceed({ analysisType })}
            className="homepage-button"
          >
            Go to Clustering Analysis
          </button>
        )}
      </div>
    </div>
  );
};

export default HomePage;