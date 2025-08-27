// src/Components/ResultsSummary.js
// src/Components/ResultsSummary.js
import React from "react";
import { BarChart3 } from "lucide-react";

const ResultsSummary = ({ stats, selectedMethod, comparisonResults }) => {
  if (!stats) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6">
  <h3 className="font-bold text-gray-800 mb-6 text-xl flex items-center gap-2">
    <BarChart3 className="w-6 h-6 text-purple-500" />
    {selectedMethod ? `${selectedMethod} Keyness Analysis Results` : "Keyness Analysis Results"}
  </h3>
  <div className="grid md:grid-cols-3 gap-4 text-center">
    <div>
      <div className="text-2xl font-bold text-blue-600">{stats.uploadedTotal}</div>
      <div className="text-sm text-gray-600">Words in your text</div>
    </div>
    <div>
      <div className="text-2xl font-bold text-green-600">{stats.corpusTotal}</div>
      <div className="text-sm text-gray-600">Words in sample corpus</div>
    </div>
    <div>
      <div className="text-2xl font-bold text-purple-600">
  {comparisonResults.slice(0, 20).length} 
</div>

      <div className="text-sm text-gray-600">Significant keywords</div>
    </div>
  </div>
</div>

  );
};

export default ResultsSummary;
