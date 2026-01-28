export function computeStats(rows, col) {
  const vals = rows
    .map(r => Number(r[col]))
    .filter(v => Number.isFinite(v));

  if (vals.length < 5) return { mean: 0, sd: 0 };

  const mean = vals.reduce((a,b)=>a+b,0) / vals.length;
  const varr = vals.reduce((a,b)=>a + (b-mean)*(b-mean), 0) / (vals.length - 1);
  const sd = Math.sqrt(varr);
  return { mean, sd };
}

export function heatColor(value, mean, sd) {
  const x = Number(value);
  if (!Number.isFinite(x) || sd === 0) return undefined;

  let z = (x - mean) / sd;
  z = Math.max(-2, Math.min(2, z));

  const t = (z + 2) / 4;      // 0..1
  const hue = 120 * t;        // red->green
  return `hsl(${hue} 70% 85%)`;
}
