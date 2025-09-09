import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ReferenceLine
} from "recharts";
import { TrendingUp } from "lucide-react";

const Charts = ({ results, method = "nltk" }) => {
  if (!results || results.length === 0) {
    return <p className="text-gray-500 italic">No chart data available.</p>;
  }

  const topResults = results.slice(0, 20);

  // Determine axis labels and data keys dynamically
let xKey = "log_likelihood";
let xLabel = "Log-Likelihood";
let yKey = "effect_size";
let yLabel = "Effect Size";
let showReferenceLine = false;

if (method.toLowerCase() === "nltk") {
  xKey = "log_likelihood";
  xLabel = "Log-Likelihood";
  yKey = "effect_size";
  yLabel = "Effect Size";
  showReferenceLine = true;
} else if (method.toLowerCase() === "sklearn") {
  xKey = "chi2";
  xLabel = "Chi²";
  yKey = null; // No scatter for sklearn
} else if (method.toLowerCase() === "gensim") {
  xKey = "tfidf_score";
  xLabel = "TF-IDF Score";
  yKey = null; // No scatter for gensim
} else if (method.toLowerCase() === "spacy") {
  xKey = "chi2";
  xLabel = "Chi²";
  yKey = "effect_size";
  yLabel = "Effect Size";
  showReferenceLine = true;
}

  // Prepare bar chart data
  const barData = topResults.map(r => ({
    word: r.word,
    uploaded: r.uploaded_count ?? r.count_a ?? 0,
    sample: r.sample_count ?? r.count_b ?? 0,
    score: r.log_likelihood ?? r.chi2 ?? r.tfidf_score ?? 0,
  }));

  // Determine if scatter chart should be displayed
  const showScatter = ["nltk", "spacy"].includes(method.toLowerCase());

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-6">
      <h3 className="font-bold text-gray-800 mb-6 text-xl flex items-center gap-2">
        <TrendingUp className="w-6 h-6 text-green-500" />
        Keyword Frequency Comparison ({method.toUpperCase()})
      </h3>

      {/* Bar Chart */}
      <div className="mb-8" style={{ width: "100%", height: 400 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={barData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="word" angle={-45} textAnchor="end" height={80} fontSize={12} />
            <YAxis label={{ value: yLabel, angle: -90, position: "insideLeft" }} />
            <Tooltip formatter={value => value} labelFormatter={word => `Word: ${word}`} />
            <Legend />
            <Bar dataKey="uploaded" name="Your Text" fill="#3b82f6" radius={[2, 2, 0, 0]} />
            <Bar dataKey="sample" name="Sample Corpus" fill="#10b981" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Scatter Chart */}
{ xKey && yKey && (
  <div className="mb-8" style={{ width: "100%", height: 400 }}>
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart data={topResults} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey={xKey}
          name={xLabel}
          label={{ value: xLabel, position: "insideBottom", offset: -10 }}
        />
        <YAxis
          dataKey={yKey}
          name={yLabel}
          label={{ value: yLabel, angle: -90, position: "insideLeft" }}
        />
        {showReferenceLine && <ReferenceLine x={3.84} stroke="#ef4444" strokeDasharray="5 5" />}
        <Tooltip
          formatter={(value) => value}
          labelFormatter={(_, payload) =>
            payload?.[0]?.payload ? `Word: ${payload[0].payload.word} (${payload[0].payload.keyness})` : ""
          }
          contentStyle={{ backgroundColor: "#f8fafc", border: "1px solid #e2e8f0" }}
        />
        <Scatter
          name="Positive Keyness"
          data={results.filter(r => r.keyness === "Positive")}
          fill="#3b82f6"
        />
        <Scatter
          name="Negative Keyness"
          data={results.filter(r => r.keyness === "Negative")}
          fill="#ef4444"
        />
      </ScatterChart>
    </ResponsiveContainer>
  </div>
)}

    </div>
  );
};

export default Charts;
