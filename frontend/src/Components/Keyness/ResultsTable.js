// src/Components/ResultsTable.js
import React from "react";

const ResultsTable = ({ results, method }) => {
  if (!results || results.length === 0) return null;

  const isSklearn = method?.toLowerCase() === "sklearn";

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mt-6 overflow-x-auto">
      <h3 className="font-bold text-gray-800 mb-4 text-xl">Detailed Keyword Table</h3>
      <table className="min-w-full table-auto text-left border-collapse">
        <thead>
          <tr className="bg-gray-100">
            <th className="px-4 py-2 border-b">Word</th>
            <th className="px-4 py-2 border-b">Your Text Freq</th>
            <th className="px-4 py-2 border-b">Sample Freq</th>
            {isSklearn ? (
              <>
                <th className="px-4 py-2 border-b">Chi²</th>
                <th className="px-4 py-2 border-b">p-value</th>
              </>
            ) : (
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
              <td className="px-4 py-2 border-b">{row.uploaded_count ?? row.uploaded_freq}</td>
              <td className="px-4 py-2 border-b">{row.sample_count ?? row.sample_freq}</td>

              {method === "SKLEARN" ? (
  <>
    <td>{row.chi2.toFixed(3)}</td>
    <td>{row.p_value.toExponential(2)}</td>
    <td>-</td> {/* No keyness label for sklearn */}
  </>
) : (
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
