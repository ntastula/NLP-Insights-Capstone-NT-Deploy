import React, { useEffect, useMemo, useState } from "react";
import SentimentResults from "./SentimentResults";

const API_URL = `${process.env.REACT_APP_BACKEND_URL}/api/analyse-sentiment/`;

export default function SentimentAnalyser({ uploadedText, uploadedPreview, corpusPreview, onBack, genre }) {
    const [data, setData] = useState(null);
    const [state, setState] = useState({ loading: true, error: "" });

    const previews = useMemo(() => {
        const out = [];
        if (uploadedPreview) out.push({ label: "Your text (preview)", body: uploadedPreview });
        if (corpusPreview) {
            const label = genre ? `Corpus preview (${genre})` : "Corpus preview";
            out.push({ label, body: corpusPreview });
        }
        return out;
    }, [uploadedPreview, corpusPreview, genre]);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            setState({ loading: true, error: "" });
            try {
                const res = await fetch(API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        uploaded_text: uploadedText || "",
                        corpus_name: genre || "",          // NEW: tell backend which genre corpus to use
                    }),
                });
                const json = await res.json().catch(() => ({}));
                if (!res.ok) throw new Error(json?.error || `HTTP ${res.status}`);
                if (!cancelled) setData(json);
            } catch (e) {
                if (!cancelled) setState({ loading: false, error: e.message || "Request failed" });
                return;
            }
            if (!cancelled) setState({ loading: false, error: "" });
        })();
        return () => { cancelled = true; };
    }, [uploadedText, genre]);   // re-run if user switches genre

    return (
        <div className="min-h-screen bg-gray-50 p-6">
            <div className="max-w-5xl mx-auto">
                <div className="flex items-center justify-between mb-6">
                    <h1 className="text-2xl md:text-3xl font-bold">Sentiment</h1>
                    <button onClick={onBack} className="px-3 py-2 text-sm rounded-md bg-gray-200 hover:bg-gray-300 text-gray-800">
                        ← Back
                    </button>
                </div>

                {state.loading && (
                    <div className="bg-white border rounded-2xl p-8 shadow-sm text-center">
                        <div className="animate-pulse text-gray-500">Analysing your text…</div>
                    </div>
                )}

                {state.error && (
                    <div className="bg-red-50 border border-red-200 rounded-2xl p-6 text-red-700">
                        Error: {state.error}. Is the API available at <code className="font-mono">{API_URL}</code>?
                    </div>
                )}

                {!state.loading && !state.error && (
                    <>
                        {previews.length > 0 && (
                            <div className="grid md:grid-cols-2 gap-4 mb-6">
                                {previews.map((b, i) => (
                                    <div key={i} className="bg-white border rounded-xl p-4 shadow-sm">
                                        <div className="text-xs font-semibold text-gray-600 mb-2">{b.label}</div>
                                        <pre className="text-sm text-gray-800 whitespace-pre-wrap leading-snug max-h-48 overflow-auto">{b.body}</pre>
                                    </div>
                                ))}
                            </div>
                        )}
                        <SentimentResults data={data} />
                    </>
                )}
            </div>
        </div>
    );
}
