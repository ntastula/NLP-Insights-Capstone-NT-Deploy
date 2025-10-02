// src/components/GenreCorpusSelector.js
import React from "react";

const GenreCorpusSelector = ({
  loading,
  err,
  localGenre,
  onGenreChange,
  filteredCorpora,
  formatDisplayName
}) => {
  return (
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
          onChange={(e) => onGenreChange(e)}
          className="homepage-select"
          disabled={loading || !!err || filteredCorpora.length === 0}
        >
          {filteredCorpora.map((file) => (
            <option key={file} value={file}>
              {formatDisplayName(file)}
            </option>
          ))}
        </select>
      )}
    </div>
  );
};

export default GenreCorpusSelector;
