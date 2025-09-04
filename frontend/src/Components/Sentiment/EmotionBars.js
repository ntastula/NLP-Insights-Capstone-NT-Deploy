import React from "react";

export default function EmotionBars({ emotions }) {
    const ordered = ["joy", "sadness", "anger", "fear", "disgust"];
    return (
        <div>
            {ordered.map((k) => {
                const v = Math.max(0, Math.min(1, emotions?.[k] || 0));
                return (
                    <div key={k} style={{ marginBottom: 12 }}>
                        {/* Label + % */}
                        <div
                            className="flex justify-between text-xs text-gray-600 mb-1"
                            style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}
                        >
                            <span style={{ fontWeight: 600 }}>
                                {k[0].toUpperCase() + k.slice(1)}
                            </span>
                            <span>{(v * 100).toFixed(1)}%</span>
                        </div>

                        {/* Track */}
                        <div
                            className="h-2 bg-gray-200 rounded-full overflow-hidden"
                            style={{
                                height: 8,                          // fallback height
                                backgroundColor: "#e5e7eb",         // fallback track color (Tailwind gray-200)
                                borderRadius: 9999,
                                overflow: "hidden",
                            }}
                        >
                            {/* Fill */}
                            <div
                                className="h-full rounded-full"
                                style={{
                                    height: "100%",
                                    width: `${v * 100}%`,
                                    borderRadius: 9999,
                                    background: "linear-gradient(90deg, #8b5cf6, #3b82f6)",
                                    transition: "width 300ms ease",
                                }}
                            />
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
