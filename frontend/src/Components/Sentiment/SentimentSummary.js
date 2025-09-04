import React from "react";

const polarityLabel = (p) => (typeof p === "number" ? (p > 0 ? "Positive" : p < 0 ? "Negative" : "Neutral") : "—");
const formatPct = (x) => `${(100 * (x || 0)).toFixed(1)}%`;
const nice = (x, d = 3) => (typeof x === "number" ? x.toFixed(d) : "—");

export default function SentimentSummary({ summary }) {
    const {
        polarity,
        coverage,
        token_count,
        matched_token_count,
        sentiment_score_mean,
        magnitude,
        stddev,
        positive_ratio,
        negative_ratio,
        neutral_ratio,
    } = summary || {};

    return (
        <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100">
                    Polarity: {polarityLabel(polarity)}
                </span>
                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-indigo-100 text-indigo-700">
                    Coverage: {formatPct(coverage)}
                </span>
                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100">
                    Tokens: {token_count ?? 0} (matched {matched_token_count ?? 0})
                </span>
            </div>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <div>
                    <div className="text-xs text-gray-500">Mean Sentiment</div>
                    <div className="text-lg font-semibold">{nice(sentiment_score_mean)}</div>
                </div>
                <div>
                    <div className="text-xs text-gray-500">Magnitude</div>
                    <div className="text-lg font-semibold">{nice(magnitude)}</div>
                </div>
                <div>
                    <div className="text-xs text-gray-500">Std Dev</div>
                    <div className="text-lg font-semibold">{nice(stddev)}</div>
                </div>
                <div>
                    <div className="text-xs text-gray-500">Composition</div>
                    <div className="text-sm text-gray-700">
                        + {formatPct(positive_ratio)} · − {formatPct(negative_ratio)} · ○ {formatPct(neutral_ratio)}
                    </div>
                </div>
            </div>
        </div>
    );
}
