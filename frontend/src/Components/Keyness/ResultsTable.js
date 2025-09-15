import React from "react";
import { formatNumber } from "../../Utils";

const ResultsTable = ({ results = [], method = "nltk" }) => {
  if (!Array.isArray(results) || results.length === 0) return null;

  const methodUpper = method.toUpperCase();
  const isSklearn = methodUpper === "SKLEARN";
  const isGensim = methodUpper === "GENSIM";
  const isSpacy = methodUpper === "SPACY";
  const isNltk = methodUpper === "NLTK";

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mt-6 overflow-x-auto">
      <h3 className="font-bold text-gray-800 mb-4 text-xl">Detailed Keyword Table</h3>
      <table className="min-w-full table-auto text-left border-collapse">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-4 py-2 border-b">Word</th>
            <th className="px-4 py-2 border-b">Your Text Freq</th>
            <th className="px-4 py-2 border-b">Sample Freq</th>

            {/* Sklearn and Spacy show Chi² and p-value */}
            {(isSklearn || isSpacy) && (
              <>
                <th className="px-4 py-2 border-b">Chi²</th>
                <th className="px-4 py-2 border-b">p-value</th>
              </>
            )}

            {/* Gensim */}
            {isGensim && <th className="px-4 py-2 border-b">TF-IDF Score</th>}

            {/* Nltk and Spacy show Effect Size / Log-Likelihood / Keyness */}
            {(isNltk || isSpacy) && (
              <>
                <th className="px-4 py-2 border-b">Effect Size</th>
                <th className="px-4 py-2 border-b">Log-Likelihood</th>
                <th className="px-4 py-2 border-b">Keyness</th>
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
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-4 py-2 border-b">{word}</td>
                <td className="px-4 py-2 border-b">{uploaded}</td>
                <td className="px-4 py-2 border-b">{sample}</td>

                {(isSklearn || isSpacy) && (
                  <>
                    <td className="px-4 py-2 border-b">{formatNumber(row.chi2)}</td>
                    <td className="px-4 py-2 border-b">{formatNumber(row.p_value, 2)}</td>
                  </>
                )}

                {isGensim && (
                  <td className="px-4 py-2 border-b">{formatNumber(row.tfidf_score)}</td>
                )}

                {(isNltk || isSpacy) && (
  <>
    <td className="px-4 py-2 border-b">{formatNumber(row.effect_size)}</td>
    <td className="px-4 py-2 border-b">{formatNumber(row.log_likelihood)}</td>
    <td className="px-4 py-2 border-b">{row.keyness ?? "-"}</td>
  </>
)}

              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;
