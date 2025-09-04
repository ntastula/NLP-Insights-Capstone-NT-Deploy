import React from "react";

const nice = (x, d = 3) => (typeof x === "number" ? x.toFixed(d) : "—");

export default function SentimentWordList({ title, rows, sign = "pos" }) {
    const items = (rows || []).map((t) => ({
        word: t.word,
        count: t.count,
        contribution: t.contribution,
        sentiment: t.sentiment_score,
    }));

    return (
        <>
            <h4 className="font-semibold mb-3">{title}</h4>
            <ul className="space-y-2">
                {items.length === 0 && <li className="text-sm text-gray-500">No items.</li>}
                {items.map((it, idx) => (
                    <li key={`${it.word}-${idx}`} className="flex items-center justify-between gap-3">
                        <div className="truncate">
                            <span className="font-mono">{it.word}</span>{" "}
                            <span className="text-xs text-gray-500">×{it.count}</span>
                        </div>
                        <div className="text-right">
                            <div className={`text-xs ${sign === "pos" ? "text-green-700" : "text-red-700"}`}>
                                {sign === "pos" ? "+" : ""}
                                {nice(it.contribution)} contrib
                            </div>
                            <div className="text-[11px] text-gray-500">score {nice(it.sentiment)}</div>
                        </div>
                    </li>
                ))}
            </ul>
        </>
    );
}

export function OOVList({ pairs }) {
    return (
        <>
            <h4 className="font-semibold mb-3">Top Out-of-Vocabulary</h4>
            <ul className="grid grid-cols-2 gap-x-4 gap-y-2">
                {(pairs || []).length === 0 && <li className="text-sm text-gray-500">No OOV tokens.</li>}
                {(pairs || []).map(([w, c], idx) => (
                    <li key={`${w}-${idx}`} className="flex justify-between">
                        <span className="font-mono truncate">{w}</span>
                        <span className="text-xs text-gray-500">×{c}</span>
                    </li>
                ))}
            </ul>
        </>
    );
}
