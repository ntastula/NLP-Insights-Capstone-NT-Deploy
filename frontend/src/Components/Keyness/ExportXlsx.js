import * as XLSX from "xlsx";
import { formatNumber } from "../../Utils";

export function exportKeynessToXlsx(
  results = [],
  method = "nltk",
  stats = {},
  posGroups = {},
  chartData = []
) {
  if (!Array.isArray(results) || results.length === 0) return;

  const workbook = XLSX.utils.book_new();
  const methodUpper = method.toUpperCase();
  const isSklearn = methodUpper === "SKLEARN";
  const isGensim = methodUpper === "GENSIM";
  const isSpacy = methodUpper === "SPACY";
  const isNltk = methodUpper === "NLTK";

  // --- Table Tab ---
  const tableData = results.map((r) => {
    const base = {
      Word: r.word ?? "-",
      "Your Text Freq": r.uploaded_count ?? r.uploaded_freq ?? 0,
      "Sample Freq": r.sample_count ?? r.sample_freq ?? 0,
    };

    // Chi² and p-value for SKLEARN and SPACY
    if (isSklearn || isSpacy) {
      base["Chi²"] = formatNumber(r.chi2);
      base["p-value"] = formatNumber(r.p_value, 2);
    }

    // Effect Size, Log-Likelihood, Keyness for NLTK and SPACY
    if (isNltk || isSpacy) {
      base["Effect Size"] = formatNumber(r.effect_size);
      base["Log-Likelihood"] = formatNumber(r.log_likelihood);
      base["Keyness"] = r.keyness ?? "-"; // keep as string
    }

    // TF-IDF Score for GENSIM
    if (isGensim) {
      base["TF-IDF Score"] = formatNumber(r.tfidf_score);
    }

    return base;
  });

  const tableWS = XLSX.utils.json_to_sheet(tableData);
  XLSX.utils.book_append_sheet(workbook, tableWS, "Table");

  // --- Summary Tab ---
  const summaryData = [
    ["Uploaded Total", stats.uploadedTotal ?? 0],
    ["Corpus Total", stats.corpusTotal ?? 0],
    ["Total Significant Keywords", stats.totalSignificant ?? 0],
  ];
  const summaryWS = XLSX.utils.aoa_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(workbook, summaryWS, "Summary");

  // --- Top Keywords Tab ---
  const keywordRows = [["Word", "Keyness", "POS"]];
  Object.entries(posGroups || {}).forEach(([pos, words]) => {
    (words || []).forEach((w) => {
      keywordRows.push([w.word ?? "-", w.keyness ?? w.log_likelihood ?? "-", pos]);
    });
  });
  const keywordsWS = XLSX.utils.aoa_to_sheet(keywordRows);
  XLSX.utils.book_append_sheet(workbook, keywordsWS, "Top Keywords");

  // --- Word Data Tab ---
  const wordDataRows = [
  [
    "Word",
    "Your Text Count",
    "Corpus Count",
    "Chi²",
    "p-value",
    "Log-Likelihood",
    "Effect Size",
    "Keyness",
    "POS"
  ]
];

results.forEach((w) => {
  wordDataRows.push([
    w.word ?? "-",
    w.uploaded_count ?? w.count_a ?? 0,
    w.sample_count ?? w.count_b ?? 0,
    formatNumber(w.chi2),
    formatNumber(w.p_value, 2),
    formatNumber(w.log_likelihood),
    formatNumber(w.effect_size),
    w.keyness ?? "-",
    w.pos ?? "N/A"
  ]);
});

const wordDataWS = XLSX.utils.aoa_to_sheet(wordDataRows);
XLSX.utils.book_append_sheet(workbook, wordDataWS, "Word Data");

  // --- Charts Tab ---
  if (chartData && chartData.length > 0) {
    const chartRows = [["Label", "Value"]];
    chartData.forEach((d) => {
      chartRows.push([d.label ?? "-", d.value ?? 0]);
    });
    const chartsWS = XLSX.utils.aoa_to_sheet(chartRows);
    XLSX.utils.book_append_sheet(workbook, chartsWS, "Charts");
  }

  // Save file
  XLSX.writeFile(workbook, "analysis_results.xlsx");
}
