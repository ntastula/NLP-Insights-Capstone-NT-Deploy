import Papa from "papaparse";

export const exportAnalysisToCSV = ({
  comparisonResults,
  stats,
  posGroups,
  chartData
}) => {
  const rows = [];

  // --- Summary Stats ---
  rows.push(["=== Summary Stats ==="]);
  rows.push(["Uploaded Total", stats.uploadedTotal]);
  rows.push(["Corpus Total", stats.corpusTotal]);
  rows.push(["Total Significant Keywords", stats.total_significant]);
  rows.push([]); // empty line

  // --- Top Keywords (POS groups) ---
  rows.push(["=== Top Keywords by POS ==="]);
  Object.entries(posGroups).forEach(([pos, words]) => {
    rows.push([pos]);
    words.forEach((w) => {
      rows.push([w.word, w.keyness ?? w.log_likelihood, pos]);
    });
    rows.push([]); // spacing
  });

  // --- Word-level results (from backend) ---
  rows.push(["=== All Significant Words ==="]);
  comparisonResults.forEach((w) => {
    rows.push([
      w.word,
      w.keyness ?? w.log_likelihood,
      w.pos ?? "N/A",
      w.freq_uploaded ?? 0,
      w.freq_corpus ?? 0,
    ]);
  });
  rows.push([]);

  // --- Chart data ---
  if (chartData) {
    rows.push(["=== Chart Data ==="]);
    chartData.forEach((d) => {
      rows.push([d.label, d.value]);
    });
  }

  // Convert to CSV
  const csv = Papa.unparse(rows);

  // Trigger download
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", "analysis_results.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
