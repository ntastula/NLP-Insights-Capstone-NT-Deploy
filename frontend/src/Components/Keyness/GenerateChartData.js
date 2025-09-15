export function generateChartData(results, method = "nltk") {
  if (!results || results.length === 0) return [];

  const topResults = results.slice(0, 20);

  let chartData = [];

  // Decide which value to use based on method
  topResults.forEach((r) => {
    let value = 0;
    if (method.toLowerCase() === "nltk" || method.toLowerCase() === "spacy") {
      value = r.log_likelihood ?? 0;
    } else if (method.toLowerCase() === "sklearn") {
      value = r.chi2 ?? 0;
    } else if (method.toLowerCase() === "gensim") {
      value = r.tfidf_score ?? 0;
    }

    chartData.push({
      label: r.word,
      value,
    });
  });

  return chartData;
}
