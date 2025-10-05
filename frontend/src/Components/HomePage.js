import React, { useEffect, useMemo, useState } from "react";
import CreativeKeynessResults from "./Keyness/CreativeKeynessResults";
import GenreCorpusSelector from "./GenreCorpusSelector";
import "./HomePage.css";

const HomePage = ({ onSelect, selectedGenre, onSelectGenre, onProceed }) => {
  const [corpora, setCorpora] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [localGenre, setLocalGenre] = useState("");
  const [analysisType, setAnalysisType] = useState("");
  const [analysisDone, setAnalysisDone] = useState(false);
  const [comparisonMode, setComparisonMode] = useState("");
  const [selectedMethod, setSelectedMethod] = useState("");

  // Fetch corpora list from backend (only for corpus mode)
  useEffect(() => {
    if (!analysisType) return;
    if (analysisType === "keyness" && comparisonMode === "user_text") {
      setLoading(false);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        setLoading(true);
        setErr("");
        const r = await fetch(
          `http://localhost:8000/api/corpora/?analysis=${analysisType}`,
          { credentials: "include" }
        );
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (cancelled) return;

        const list = Array.isArray(d.corpora) ? d.corpora : [];
        setCorpora(list);

        if (!list.length) {
          setErr("No corpora found for this analysis type.");
          setLocalGenre("");
          return;
        }

        let defaultFile = localGenre || list[0];
        if (analysisType === "keyness" && comparisonMode === "corpus") {
          defaultFile =
            list.find((f) => f.startsWith("general_fiction")) || list[0];
        }
        setLocalGenre(defaultFile);
      } catch (e) {
        if (!cancelled) setErr(String(e.message || e));
        setLocalGenre("");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [analysisType, comparisonMode]);

  const handleAnalysisChange = (e) => {
    const type = e.target.value;
    setAnalysisType(type);
    setLocalGenre("");
    setComparisonMode("");
  };

  const handleComparisonModeChange = (e) => {
    const mode = e.target.value;
    setComparisonMode(mode);
    setLocalGenre("");
  };

  const handleGenreChange = (e) => {
    setLocalGenre(e.target.value);
    if (typeof onSelectGenre === "function") onSelectGenre(e.target.value);
  };

  const filteredCorpora = useMemo(() => {
    if (!Array.isArray(corpora)) return [];

    return corpora.filter((file) => {
      if (analysisType === "keyness" && comparisonMode === "corpus") return true;
      if (analysisType === "sentiment" || analysisType === "sensorimotor") return true;
      return false; // clustering or user_text: no dropdown
    });
  }, [analysisType, comparisonMode, corpora]);


  const formatDisplayName = (file) => {
    let displayName = file.replace(/_keyness$/, "").replace(/\.json$/, "");
    displayName = displayName.replace(/_/g, " ");
    displayName = displayName.replace(/\b\w/g, (c) => c.toUpperCase());
    return displayName;
  };

  const handleUploadedFiles = (combinedText, files, tempGenre = null) => {
    if (tempGenre) {
      setLocalGenre(tempGenre);
      onSelectGenre(tempGenre);
    }
  };


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
            <option value="" disabled>
              Select analysis type…
            </option>
            <option value="keyness">Keyness</option>
            <option value="sentiment">Sentiment</option>
            <option value="clustering">Clustering</option>
            <option value="sensorimotor">Sensorimotor Norms</option>
          </select>
        </div>

        {/* Comparison Mode Selection (only for keyness) */}
        {analysisType === "keyness" && (
          <div className="homepage-section">
            <label className="homepage-label">
              Compare your text against:
            </label>
            <select
              onChange={handleComparisonModeChange}
              value={comparisonMode}
              className="homepage-select"
            >
              <option value="">-- Select comparison mode --</option>
              <option value="corpus">Texts from a genre corpus</option>
              <option value="user_text">
                Another of Your Texts (Two-File Upload)
              </option>
            </select>

            {comparisonMode === "user_text" && (
              <div className="info-message">
                ℹ️ You'll upload two texts in the next step:
                <ul>
                  <div>
                    A <strong>Reference text</strong>: The text to compare against; and
                  </div>
                  <div>
                    A <strong>Target text</strong>: The text being analysed
                  </div>
                </ul>
                <em>
                  Note: Texts are not stored and will be lost when you close the
                  page.
                </em>
              </div>
            )}
          </div>
        )}

        {/* Genre selector + button for corpus-based analyses */}
        {(
          (analysisType === "keyness" && comparisonMode === "corpus") ||
          analysisType === "sentiment" ||
          analysisType === "sensorimotor"
        ) && (
            <>
              <GenreCorpusSelector
                loading={loading}
                err={err}
                localGenre={localGenre}
                onGenreChange={handleGenreChange}
                filteredCorpora={filteredCorpora}
                formatDisplayName={formatDisplayName}
              />

              <button
                onClick={() => {
                  onProceed({
                    analysisType,
                    genre: localGenre,
                    comparisonMode,
                  });
                  setAnalysisDone(true);
                }}
                className="homepage-button"
                disabled={!localGenre || loading || !!err}
              >
                Go to{" "}
                {analysisType.charAt(0).toUpperCase() + analysisType.slice(1)}{" "}
                Analysis
              </button>
            </>
          )}

        {/* Special case: Keyness with user_text (no genre needed) */}
        {analysisType === "keyness" && comparisonMode === "user_text" && (
          <button
            onClick={() => {
              onProceed({
                analysisType: "keyness",
                genre: null,           
                comparisonMode: comparisonMode
              });
            }}
            className="homepage-button"
            disabled={loading || !!err} 
          >
            Go to Keyness Analysis
          </button>
        )}

        {/* Clustering button */}
        {analysisType === "clustering" && (
          <button
            onClick={() => onProceed({ analysisType })}
            className="homepage-button"
          >
            Go to Clustering Analysis
          </button>
        )}
      </div>

      {/* Render CreativeKeynessResults below the selection UI (only for corpus mode) */}
      {analysisDone && localGenre && comparisonMode === "corpus" && (
        <CreativeKeynessResults
          genre={localGenre}
          onSelect={onSelect}
          selectedGenre={localGenre}
          onProceed={onProceed}
        />
      )}
    </div>
  );
};

export default HomePage;