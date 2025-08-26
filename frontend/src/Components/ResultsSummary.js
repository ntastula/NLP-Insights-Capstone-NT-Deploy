// src/Components/ResultsSummary.js
import React from "react";
import { BarChart3 } from "lucide-react";
import KeynessAnalyser from "./KeynessAnalyser";

const ResultsSummary = ({ uploadedTotal, corpusTotal, sigKeywords, method }) => {
  if (!uploadedTotal && !corpusTotal) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6">
      <h3 className="font-bold text-gray-800 mb-6 text-xl flex items-center gap-2">
        <BarChart3 className="w-6 h-6 text-purple-500" />
        {method} Keyness Analysis Results
      </h3>

      <div className="mb-6 p-4 bg-blue-50 rounded-lg">
        <div className="grid md:grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600">{uploadedTotal}</div>
            <div className="text-sm text-gray-600">Words in your text</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">{corpusTotal}</div>
            <div className="text-sm text-gray-600">Words in sample corpus</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-600">{sigKeywords}</div>
            <div className="text-sm text-gray-600">Significant keywords</div>
          </div>
        </div>
      </div>
    </div>
  );
};


export default ResultsSummary;
