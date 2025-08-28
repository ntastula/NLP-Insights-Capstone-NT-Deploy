import React from "react";

const ResultsTable = ({ results, method }) => {
  if (!results || results.length === 0) return null;

  const methodLower = method?.toLowerCase();
  const isSklearn = methodLower === "sklearn";
  const isGensim = methodLower === "gensim";
  const isSpacy = methodLower === "spacy";
  const isNltk = methodLower === "nltk";

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mt-6 overflow-x-auto">
      <h3 className="font-bold text-gray-800 mb-4 text-xl">Detailed Keyword Table</h3>
      <table className="min-w-full table-auto text-left border-collapse">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-4 py-2 border-b">Word</th>
            <th className="px-4 py-2 border-b">Your Text Freq</th>
            <th className="px-4 py-2 border-b">Sample Freq</th>

            {isSklearn && (
              <>
                <th className="px-4 py-2 border-b">Chi²</th>
                <th className="px-4 py-2 border-b">p-value</th>
              </>
            )}

            {isGensim && <th className="px-4 py-2 border-b">TF-IDF Score</th>}

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
          {results.map((row, index) => (
            <tr key={index} className="hover:bg-gray-50">
              <td className="px-4 py-2 border-b">{row.word}</td>
              <td className="px-4 py-2 border-b">{row.uploaded_count ?? row.count_a ?? row.uploaded_freq ?? 0}</td>
              <td className="px-4 py-2 border-b">{row.sample_count ?? row.count_b ?? row.sample_freq ?? 0}</td>

              {isSklearn && (
                <>
                  <td>{row.chi2?.toFixed(3)}</td>
                  <td>{row.p_value?.toExponential(2)}</td>
                </>
              )}

              {isGensim && <td>{row.tfidf_score?.toFixed(3)}</td>}

              {(isNltk || isSpacy) && (
                <>
                  <td>{row.effect_size}</td>
                  <td>{row.log_likelihood}</td>
                  <td>{row.keyness}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;
