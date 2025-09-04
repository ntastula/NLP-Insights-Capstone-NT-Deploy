import React from "react";
import SentimentSummary from "./SentimentSummary";
import EmotionBars from "./EmotionBars";
import SentimentWordList, { OOVList } from "./SentimentWordList";

export default function SentimentResults({ data }) {
    const summary = data?.summary || {};
    const emotions = data?.emotions || {};
    const pos = data?.top_contributors?.positive || [];
    const neg = data?.top_contributors?.negative || [];
    const oov = summary?.oov_examples || [];

    return (
        <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-white border rounded-2xl p-6 shadow-sm">
                    <SentimentSummary summary={summary} />
                </div>
                <div className="bg-white border rounded-2xl p-6 shadow-sm">
                    <h3 className="font-semibold mb-4">Emotion Averages</h3>
                    <EmotionBars emotions={emotions} />
                </div>
            </div>

            <div className="grid lg:grid-cols-3 gap-4">
                <div className="bg-white border rounded-xl p-4 shadow-sm">
                    <SentimentWordList title="Top Positive" rows={pos} sign="pos" />
                </div>
                <div className="bg-white border rounded-xl p-4 shadow-sm">
                    <SentimentWordList title="Top Negative" rows={neg} sign="neg" />
                </div>
                <div className="bg-white border rounded-xl p-4 shadow-sm">
                    <OOVList pairs={oov} />
                </div>
            </div>
        </div>
    );
}
