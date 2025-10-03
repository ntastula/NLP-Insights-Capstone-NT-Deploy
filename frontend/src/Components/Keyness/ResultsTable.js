import React from "react";
import { formatNumber } from "../../Utils";
import "./ResultsTable.css";

const ResultsTable = ({ results = [], method = "nltk" }) => {
  if (!Array.isArray(results) || results.length === 0) return null;

  const methodUpper = method.toUpperCase();
  const isSklearn = methodUpper === "SKLEARN";
  const isGensim = methodUpper === "GENSIM";
  const isSpacy = methodUpper === "SPACY";
  const isNltk = methodUpper === "NLTK";

  // Statistics descriptions based on method
  const getStatisticsDescription = () => {
    const baseStats = {
      "Your Text Freq": "The number of times this word appears in your uploaded text",
      "Sample Freq": "The number of times this word appears in the comparison sample text"
    };

    if (isSklearn || isSpacy) {
      return {
        ...baseStats,
        "Chi² (Chi-squared)": "A statistical test that measures how much a word's frequency differs from what we'd expect by chance. Higher values indicate more significant differences.",
        "p-value": "The probability that the observed difference occurred by chance. Values below 0.05 are typically considered statistically significant."
      };
    }

    if (isGensim) {
      return {
        ...baseStats,
        "TF-IDF Score": "Term Frequency-Inverse Document Frequency score. Higher values indicate words that are frequent in your text but rare in the comparison sample."
      };
    }

    if (isNltk || isSpacy) {
      return {
        ...baseStats,
        "Effect Size": "A measure of how practically significant the difference is, regardless of sample size. Larger absolute values indicate stronger effects.",
        "Log-Likelihood": "A statistical measure of how unlikely the observed word frequency would be if both texts came from the same source. Higher values indicate more distinctive words.",
        "Keyness": "An overall measure of how characteristic or 'key' this word is to your text compared to the sample. Higher values indicate more distinctive words."
      };
    }

    return baseStats;
  };

  const statisticsDescriptions = getStatisticsDescription();

  return (
    <div className="results-table-container">
      {/* Statistics Explanation Section */}
      <div className="statistics-explanation">
        <h3 className="explanation-title">Understanding Your Results</h3>
        <p className="explanation-intro">
          This table shows words that are statistically distinctive in your text compared to a reference sample.
          Here's what each column means:
        </p>
        <div className="statistics-grid">
          {Object.entries(statisticsDescriptions).map(([stat, description]) => (
            <div key={stat} className="statistic-item">
              <strong className="statistic-name">{stat}:</strong>
              <span className="statistic-description">{description}</span>
            </div>
          ))}
        </div>
        <div className="interpretation-note">
          <strong>💡 Interpretation Tip:</strong> Words with higher statistical values are more characteristic of your text and may represent key themes or distinctive language patterns.
        </div>
      </div>

      {/* Results Table */}
      <div className="table-section">
        <h3 className="table-title">Detailed Keyword Analysis Results</h3>
        <div className="table-wrapper">
          <table className="results-table">
            <thead>
              <tr>
                <th className="word-column">Word</th>
                <th className="freq-column">Your Text Freq</th>
                <th className="freq-column">Sample Freq</th>

                {/* Sklearn and Spacy show Chi² and p-value */}
                {(isSklearn || isSpacy) && (
                  <>
                    <th className="stat-column">Chi²</th>
                    <th className="stat-column">p-value</th>
                  </>
                )}

                {/* Gensim */}
                {isGensim && <th className="stat-column">TF-IDF Score</th>}

                {/* Nltk and Spacy show Effect Size / Log-Likelihood / Keyness */}
                {(isNltk || isSpacy) && (
                  <>
                    <th className="stat-column">Effect Size</th>
                    <th className="stat-column">Log-Likelihood</th>
                    <th className="stat-column">Keyness</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {results.map((row, index) => {
                const word = row.word ?? "-";
                const uploaded = row.uploaded_count ?? row.uploaded_freq ?? 0;
                const sample = row.sample_count ?? row.sample_freq ?? 0;

                return (
                  <tr key={index} className={index % 2 === 0 ? "row-even" : "row-odd"}>
                    <td className="word-cell">{word}</td>
                    <td className="freq-cell">{uploaded}</td>
                    <td className="freq-cell">{sample}</td>

                    {(isSklearn || isSpacy) && (
                      <>
                        <td className="stat-cell">{formatNumber(row.chi2)}</td>
                        <td className="stat-cell p-value-cell">{formatNumber(row.p_value, 2)}</td>
                      </>
                    )}

                    {isGensim && (
                      <td className="stat-cell">{formatNumber(row.tfidf_score)}</td>
                    )}

                    {(isNltk || isSpacy) && (
                      <>
                        <td className="stat-cell">{formatNumber(row.effect_size)}</td>
                        <td className="stat-cell">{formatNumber(row.log_likelihood)}</td>
                        <td className="stat-cell keyness-cell">{row.keyness_score ?? "-"}</td>
                      </>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ResultsTable;
