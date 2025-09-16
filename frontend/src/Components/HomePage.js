import React, { useEffect, useMemo, useState } from "react";

const HomePage = ({ onSelect, selectedGenre, onSelectGenre }) => {
  // Genre corpora state (from your genre branch)
  const [corpora, setCorpora] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");

  // Local fallback if parent doesn't control selectedGenre
  const [localGenre, setLocalGenre] = useState("");
  const effectiveGenre = (selectedGenre ?? localGenre) || "";

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        setLoading(true);
        setErr("");
        const r = await fetch("http://localhost:8000/api/corpora/", {
          credentials: "include",
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        if (cancelled) return;

        const list = Array.isArray(d.corpora) ? d.corpora : [];
        setCorpora(list);

        // If nothing selected yet, pick the first corpus
        if (!effectiveGenre && list.length) {
          if (typeof onSelectGenre === "function") {
            onSelectGenre(list[0]);
          } else {
            setLocalGenre(list[0]);
          }
        }
      } catch (e) {
        if (!cancelled) setErr(String(e.message || e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // We intentionally don't depend on effectiveGenre to avoid loops.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleGenreChange(e) {
    const val = e.target.value;
    if (typeof onSelectGenre === "function") {
      onSelectGenre(val);
    } else {
      setLocalGenre(val);
    }
  }

  const genreOptions = useMemo(
    () => corpora.map((c) => (
      <option key={c} value={c}>
        {c}
      </option>
    )),
    [corpora]
  );

  // --- UI from main branch, with genre picker inserted ---
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-8 bg-gradient-to-br from-blue-500 via-purple-600 to-purple-800 p-6">
      <div className="bg-white/95 backdrop-blur-lg rounded-2xl shadow-2xl p-8 text-center max-w-lg w-full">
        <h1 className="text-4xl font-bold text-gray-800 mb-6">
          Welcome to TTC Writing Analysis
        </h1>
        <p className="text-gray-600 mb-6">
          Select the type of analysis you would like to perform:
        </p>

        {/* Genre Corpus */}
        <label className="block text-left text-sm font-medium text-gray-700 mb-1">
          Genre Corpus
        </label>
        <select
          className="w-full p-3 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 mb-4 disabled:opacity-60"
          value={effectiveGenre}
          onChange={handleGenreChange}
          disabled={loading || !!err || !corpora.length}
        >
          {loading && <option>Loading…</option>}
          {!loading && corpora.length === 0 && <option value="">No corpora found</option>}
          {!loading && corpora.length > 0 && genreOptions}
        </select>
        {err && (
          <div className="text-sm text-red-600 mb-2">
            Failed to load corpora: {err}
          </div>
        )}

        {/* Analysis selection */}
        <select
          onChange={(e) => onSelect(e.target.value)}
          className="w-full p-3 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          defaultValue=""
        >
          <option value="" disabled>
            What type of analysis would you like to do?
          </option>
          <option value="keyness">Keyness</option>
          <option value="sentiment">Sentiment</option>
          <option value="clustering">Clustering</option>
          <option value="sensorimotor">Sensorimotor Norms</option>
        </select>
      </div>
    </div>
  );
};

export default HomePage;
