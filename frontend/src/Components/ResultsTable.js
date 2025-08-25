// src/Components/ResultsTable.js
import React from "react";

const ResultsTable = ({ results }) => {
  if (!results) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6 overflow-x-auto">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr>
            <th className="px-4 py-2 border-b">Word</th>
            <th className="px-4 py-2 border-b">Your Text Count</th>
            <th className="px-4 py-2 border-b">Corpus Count</th>
            <th className="px-4 py-2 border-b">Effect Size</th>
            <th className="px-4 py-2 border-b">Log-Likelihood</th>
            <th className="px-4 py-2 border-b">Keyness</th>
          </tr>
        </thead>
        <tbody>
          {results.results.map((row, idx) => (
            <tr key={idx} className="hover:bg-gray-50">
              <td className="px-4 py-2 border-b">{row.word}</td>
              <td className="px-4 py-2 border-b">{row.uploaded_count}</td>
              <td className="px-4 py-2 border-b">{row.sample_count}</td>
              <td className="px-4 py-2 border-b">{row.effect_size}</td>
              <td className="px-4 py-2 border-b">{row.log_likelihood}</td>
              <td className="px-4 py-2 border-b">{row.keyness}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;
