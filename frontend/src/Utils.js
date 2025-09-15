export const formatNumber = (num, decimals = 3) => {
  if (num === undefined || num === null) return "-";

  const n = Number(num);
  if (isNaN(n)) return "-";

  if (n === 0) return "0";

  return Math.abs(n) < 1e-6 ? n.toExponential(2) : n.toFixed(decimals);
};
