import React, { useEffect, useMemo, useState } from "react";

const HomePage = ({ onSelect, selectedGenre, onSelectGenre, onProceed }) => {
  const [corpora, setCorpora] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  const [localGenre, setLocalGenre] = useState("");
  const [analysisType, setAnalysisType] = useState("");

  // Fetch corpora list from backend
  useEffect(() => {
  if (!analysisType) return; // don't fetch before user selects analysis type

  let cancelled = false;

  (async () => {
    try {
      setLoading(true);
      setErr("");
      const r = await fetch(`http://localhost:8000/api/corpora/?analysis=${analysisType}`, { credentials: "include" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      if (cancelled) return;

      const list = Array.isArray(d.corpora) ? d.corpora : [];
      setCorpora(list);

      // Set default genre if none selected yet
      if (!localGenre && list.length) {
  let defaultFile = list[0];
  if (analysisType === "keyness") {
    defaultFile = list.find(f => f.startsWith("general_english")) || list[0];
  }
  setLocalGenre(defaultFile);
}
    } catch (e) {
      if (!cancelled) setErr(String(e.message || e));
    } finally {
      if (!cancelled) setLoading(false);
    }
  })();

  return () => { cancelled = true; };
}, [analysisType]);



  const handleAnalysisChange = (e) => {
    const type = e.target.value;
    setAnalysisType(type);
    setLocalGenre(""); // reset genre when analysis changes
  };

  const handleGenreChange = (e) => {
    setLocalGenre(e.target.value);
    if (typeof onSelectGenre === "function") onSelectGenre(e.target.value);
  };

  const filteredCorpora = useMemo(() => {
  if (!Array.isArray(corpora)) return [];

  return corpora.filter(file => {
    if (analysisType === "keyness") return true; // all valid
    if (analysisType === "sentiment" || analysisType === "sensorimotor") return true;
    return false; // clustering: no dropdown
  });
}, [analysisType, corpora]);


  ;

  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-8 bg-gradient-to-br from-blue-500 via-purple-600 to-purple-800 p-6">
      <div className="bg-white/95 backdrop-blur-lg rounded-2xl shadow-2xl p-8 text-center max-w-lg w-full">
        <h1 className="text-4xl font-bold text-gray-800 mb-6">Welcome to TTC Writing Analysis</h1>

        {/* Analysis Selection */}
        <label className="block text-left text-sm font-medium text-gray-700 mb-1">
          What type of analysis would you like to do?
        </label>
        <select
          onChange={handleAnalysisChange}
          value={analysisType}
          className="w-full p-3 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 mb-4"
        >
          <option value="" disabled>Select analysis type…</option>
          <option value="keyness">Keyness</option>
          <option value="sentiment">Sentiment</option>
          <option value="clustering">Clustering</option>
          <option value="sensorimotor">Sensorimotor Norms</option>
        </select>

        {/* Genre Selection + Button for non-clustering */}
{analysisType && analysisType !== "clustering" && (
  <>
    <label className="block text-left text-sm font-medium text-gray-700 mb-1">
      Genre Corpus
    </label>
<select
  value={localGenre}
  onChange={handleGenreChange}
  className="w-full p-3 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 mb-4"
  disabled={loading || !!err || filteredCorpora.length === 0}
>
  {filteredCorpora.map((file) => {
    // Remove "_keyness" and ".json"
    let displayName = file.replace(/_keyness$/, "").replace(/\.json$/, "");
    
    // Replace underscores with spaces
    displayName = displayName.replace(/_/g, " ");
    
    // Capitalise each word
    displayName = displayName.replace(/\b\w/g, (c) => c.toUpperCase());

    return (
      <option key={file} value={file}>
        {displayName}
      </option>
    );
  })}
</select>




    <button
      onClick={() => onProceed({ analysisType, genre: localGenre })}
      className="w-full bg-purple-600 text-white p-3 rounded-lg font-medium hover:bg-purple-700 transition"
    >
      Go to {analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} Analysis
    </button>
  </>
)}

{/* Button for clustering */}
{analysisType === "clustering" && (
  <button
    onClick={() => onProceed({ analysisType })}
    className="w-full bg-purple-600 text-white p-3 rounded-lg font-medium hover:bg-purple-700 transition"
  >
    Go to Clustering Analysis
  </button>
)}



      </div>
    </div>
  );
};

export default HomePage;
