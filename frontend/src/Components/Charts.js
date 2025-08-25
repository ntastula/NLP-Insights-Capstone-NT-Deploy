// src/Components/Charts.js
import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine
} from "recharts";
import { TrendingUp } from "lucide-react";

const Charts = ({ results }) => {
  if (!results) return null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6">
      <h3 className="font-bold text-gray-800 mb-6 text-xl flex items-center gap-2">
        <TrendingUp className="w-6 h-6 text-green-500" />
        Keyword Frequency Comparison
      </h3>

      {/* Top Keywords Bar Chart */}
      <div className="mb-8">
        <h4 className="font-semibold text-gray-700 mb-4">
          Top 15 Most Significant Keywords (Frequency per 1000 words)
        </h4>
        <div style={{ width: "100%", height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={results.results.slice(0, 15)}
              margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="word" angle={-45} textAnchor="end" height={80} fontSize={12} />
              <YAxis label={{ value: "Frequency per 1000 words", angle: -90, position: "insideLeft" }} />
              <Tooltip
                formatter={(value, name) => [value, name === "uploaded_freq" ? "Your Text" : "Sample Corpus"]}
                labelFormatter={(word) => `Word: ${word}`}
              />
              <Legend />
              <Bar dataKey="uploaded_freq" name="Your Text" fill="#3b82f6" radius={[2, 2, 0, 0]} />
              <Bar dataKey="sample_freq" name="Sample Corpus" fill="#10b981" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Keyness Scatter Plot */}
      <div className="mb-8">
        <h4 className="font-semibold text-gray-700 mb-4">Keyness Analysis Scatter Plot</h4>
        <div style={{ width: "100%", height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart data={results.results} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="log_likelihood" name="Log-Likelihood" label={{ value: "Log-Likelihood", position: "insideBottom", offset: -10 }} />
              <YAxis dataKey="effect_size" name="Effect Size" label={{ value: "Effect Size", angle: -90, position: "insideLeft" }} />
              <ReferenceLine x={3.84} stroke="#ef4444" strokeDasharray="5 5" />
              <Tooltip
                formatter={(value, name) => [value, name]}
                labelFormatter={(_, payload) =>
                  payload?.[0]?.payload ? `Word: ${payload[0].payload.word} (${payload[0].payload.keyness} Keyness)` : ""
                }
                contentStyle={{ backgroundColor: "#f8fafc", border: "1px solid #e2e8f0" }}
              />
              <Scatter name="Positive Keyness" data={results.results.filter(r => r.keyness === "Positive")} fill="#3b82f6" />
              <Scatter name="Negative Keyness" data={results.results.filter(r => r.keyness === "Negative")} fill="#ef4444" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="flex justify-center gap-6 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded"></div>
            <span>Positive Keyness (More frequent in your text)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span>Negative Keyness (More frequent in sample corpus)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Charts;
