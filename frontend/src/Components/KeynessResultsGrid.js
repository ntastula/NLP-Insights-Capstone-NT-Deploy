// src/Components/KeynessResultsGrid.js
import React from "react";

const KeynessResultsGrid = ({ results }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    <h3>Significant Keywords:</h3>
    {results.map((r, idx) => (
      <div key={idx} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
        
        <h4 className="font-bold text-gray-800 text-lg">{r.word}</h4>
        <p className="text-gray-600">Log-likelihood: {r.log_likelihood.toFixed(2)}</p>
        <p className="text-gray-600">Uploaded text count: {r.count_a}</p>
        <p className="text-gray-600">Corpus count: {r.count_b}</p>
      </div>
    ))}
  </div>
);


export default KeynessResultsGrid;
