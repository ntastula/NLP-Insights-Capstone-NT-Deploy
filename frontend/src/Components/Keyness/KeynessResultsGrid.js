import React from "react";

const KeynessResultsGrid = ({ results, method }) => {
  if (!results || results.length === 0) return null;

  return (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    <h3 className="col-span-full font-bold text-gray-800 mb-4">Significant Keywords</h3>

    {results
      .filter(r => r.word && r.word.match(/\w/)) // Only include words with letters/numbers
      .map((r, idx) => (
        <div key={idx} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <h4 className="font-bold text-gray-800 text-lg">{r.word}</h4>

          {method.toUpperCase() === "SKLEARN" ? (
            <>
              <p className="text-gray-600">Your Text Count: {r.uploaded_count ?? r.count_a}</p>
              <p className="text-gray-600">Corpus Count: {r.sample_count ?? r.count_b}</p>
              <p className="text-gray-600">Chi²: {r.chi2?.toFixed(3)}</p>
              <p className="text-gray-600">p-value: {r.p_value?.toExponential(2)}</p>
            </>
          ) : method.toUpperCase() === "GENSIM" ? (
            <>
              <p className="text-gray-600">Your Text Count: {r.uploaded_count}</p>
              <p className="text-gray-600">Corpus Count: {r.sample_count}</p>
              <p className="text-gray-600">TF-IDF Score: {r.tfidf_score?.toFixed(3)}</p>
            </>
          ) : method.toUpperCase() === "SPACY" ? (
            <>
              <p className="text-gray-600">Your Text Count: {r.uploaded_count ?? r.count_a}</p>
              <p className="text-gray-600">Corpus Count: {r.sample_count ?? r.count_b}</p>
              {r.chi2 !== undefined && <p className="text-gray-600">Chi²: {r.chi2?.toFixed(3)}</p>}
              {r.p_value !== undefined && <p className="text-gray-600">p-value: {r.p_value?.toExponential(2)}</p>}
              {r.tfidf_score !== undefined && <p className="text-gray-600">TF-IDF Score: {r.tfidf_score?.toFixed(3)}</p>}
              <p className="text-gray-600">Log-Likelihood: {r.log_likelihood}</p>
              <p className="text-gray-600">Effect Size: {r.effect_size}</p>
              <p className="text-gray-600">Keyness: {r.keyness}</p>
            </>  
          ) : (
            <>
              <p className="text-gray-600">Log-Likelihood: {r.log_likelihood}</p>
              <p className="text-gray-600">Uploaded Text Count: {r.uploaded_count ?? r.count_a}</p>
              <p className="text-gray-600">Corpus Count: {r.sample_count ?? r.count_b}</p>
              <p className="text-gray-600">Effect Size: {r.effect_size}</p>
              <p className="text-gray-600">Keyness: {r.keyness}</p>
            </>
          )}
        </div>
      ))}
  </div>
);

};

export default KeynessResultsGrid;

