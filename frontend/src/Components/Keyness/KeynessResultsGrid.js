import React from "react";
import "./KeynessResultsGrid.css";

const KeynessResultsGrid = ({ results = [], method, onWordDetail, word }) => {
  if (!results || results.length === 0) return <p>No results available.</p>;

  const methodUpper = method.toUpperCase();
  const isSklearn = methodUpper === "SKLEARN";
  const isGensim = methodUpper === "GENSIM";
  const isSpacy = methodUpper === "SPACY";
  const isNltk = methodUpper === "NLTK";

  // Get method-specific explanation
  const getMethodExplanation = () => {
    if (isSklearn) {
      return {
        title: "Chi-Square Analysis Results",
        description: "These keywords show the strongest statistical association with your text using chi-square testing.",
        focus: "Look for words with high Chi² values and low p-values (< 0.05) for the most significant keywords."
      };
    }

    if (isGensim) {
      return {
        title: "TF-IDF Analysis Results",
        description: "These keywords are identified as distinctive using Term Frequency-Inverse Document Frequency scoring.",
        focus: "Higher TF-IDF scores indicate words that are frequent in your text but rare in the comparison sample."
      };
    }

    if (isSpacy) {
      return {
        title: "Multi-Method Analysis Results",
        description: "These keywords combine multiple statistical measures to identify the most distinctive words in your text.",
        focus: "Consider words with high keyness scores and significant statistical values across multiple measures."
      };
    }

    return {
      title: "Log-Likelihood Analysis Results",
      description: "These keywords show the strongest statistical distinctiveness using log-likelihood testing.",
      focus: "Higher keyness and log-likelihood values indicate more distinctive words in your text."
    };
  };

  const methodInfo = getMethodExplanation();
  const filteredResults = word
    ? results.filter((r) => r.word === word)
    : results;

  return (
    <div className="keyness-results-container">
      {/* Method Explanation Section */}
      <div className="method-explanation">
        <h3 className="method-title">{methodInfo.title}</h3>
        <p className="method-description">{methodInfo.description}</p>
        <div className="method-focus">
          <strong>💡 What to look for:</strong> {methodInfo.focus}
        </div>
      </div>

      {/* Results Section */}
      <div className="results-section">
        {!word && (
          <h3 className="results-title">
            Top Distinctive Keywords
            <span className="results-count">
              ({filteredResults.length} words found)
            </span>
          </h3>
        )}

        <div className="keywords-grid">
          {filteredResults.map((r, idx) => (
            <div
              key={idx}
              className="keyword-card"
              onClick={() =>
                onWordDetail &&
                onWordDetail({
                  word: r.word,
                  wordData: r,
                  method,
                  results, 
                })
              }
            >
              <div className="keyword-header">
                <h4 className="keyword-word">{r.word}</h4>
                {!word && <div className="keyword-rank">#{idx + 1}</div>}
              </div>

              <div className="keyword-stats">
                {/* Method-specific section */}
                {isSklearn && (
                  <>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Your Text:</span>
                      <span className="stat-value">
                        {r.uploaded_count ?? r.count_a}
                      </span>
                    </div>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Corpus:</span>
                      <span className="stat-value">
                        {r.sample_count ?? r.count_b}
                      </span>
                    </div>
                    <div className="stat-item primary-stat">
                      <span className="stat-label">Chi²:</span>
                      <span className="stat-value">{r.chi2?.toFixed(3)}</span>
                    </div>
                    <div className="stat-item significance-stat">
                      <span className="stat-label">p-value:</span>
                      <span className="stat-value p-value">
                        {r.p_value?.toExponential(2)}
                      </span>
                    </div>
                  </>
                )}

                {isGensim && (
                  <>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Your Text:</span>
                      <span className="stat-value">{r.uploaded_count}</span>
                    </div>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Corpus:</span>
                      <span className="stat-value">{r.sample_count}</span>
                    </div>
                    <div className="stat-item primary-stat">
                      <span className="stat-label">TF-IDF Score:</span>
                      <span className="stat-value">
                        {r.tfidf_score?.toFixed(3)}
                      </span>
                    </div>
                  </>
                )}

                {isSpacy && (
                  <>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Your Text:</span>
                      <span className="stat-value">
                        {r.uploaded_count ?? r.count_a}
                      </span>
                    </div>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Corpus:</span>
                      <span className="stat-value">
                        {r.sample_count ?? r.count_b}
                      </span>
                    </div>
                    {r.chi2 !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">Chi²:</span>
                        <span className="stat-value">{r.chi2?.toFixed(3)}</span>
                      </div>
                    )}
                    {r.p_value !== undefined && (
                      <div className="stat-item significance-stat">
                        <span className="stat-label">p-value:</span>
                        <span className="stat-value p-value">
                          {r.p_value?.toExponential(2)}
                        </span>
                      </div>
                    )}
                    {r.tfidf_score !== undefined && (
                      <div className="stat-item">
                        <span className="stat-label">TF-IDF:</span>
                        <span className="stat-value">
                          {r.tfidf_score?.toFixed(3)}
                        </span>
                      </div>
                    )}
                    <div className="stat-item">
                      <span className="stat-label">Log-Likelihood:</span>
                      <span className="stat-value">{r.log_likelihood}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Effect Size:</span>
                      <span className="stat-value">{r.effect_size}</span>
                    </div>
                    <div className="stat-item primary-stat">
                      <span className="stat-label">Keyness:</span>
                      <span className="stat-value keyness-value">
                        {r.keyness_score}
                      </span>
                    </div>
                  </>
                )}

                {isNltk && (
                  <>
                    <div className="stat-item primary-stat">
                      <span className="stat-label">Log-Likelihood:</span>
                      <span className="stat-value">{r.log_likelihood}</span>
                    </div>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Your Text:</span>
                      <span className="stat-value">
                        {r.uploaded_count ?? r.count_a}
                      </span>
                    </div>
                    <div className="stat-item frequency-stat">
                      <span className="stat-label">Corpus:</span>
                      <span className="stat-value">
                        {r.sample_count ?? r.count_b}
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Effect Size:</span>
                      <span className="stat-value">{r.effect_size}</span>
                    </div>
                    <div className="stat-item primary-stat">
                      <span className="stat-label">Keyness:</span>
                      <span className="stat-value keyness-value">
                        {r.keyness_score}
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default KeynessResultsGrid;