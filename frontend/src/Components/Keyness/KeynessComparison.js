import React from "react";
import { BarChart3 } from "lucide-react";
import KeynessResultsGrid from "./KeynessResultsGrid";

const KeynessComparison = ({ comparisonResults, method }) => {
  if (!comparisonResults) return null;

  const renderMethodGrid = (methodName, results) => {
    if (!results || results.length === 0) return <p>No results for {methodName}</p>;

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
        {comparisonResults.map((r, idx) => (
          <div key={idx} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
            <h4 className="font-bold text-gray-800 text-lg">{r.word}</h4>
            <p className="text-gray-600">Log-likelihood: {r.log_likelihood.toFixed(2)}</p>
            <p className="text-gray-600">Uploaded text count: {r.count_a}</p>
            <p className="text-gray-600">Sample corpus count: {r.count_b}</p>
          </div>
        ))}
      </div>
    );
  };
};

export default KeynessComparison;