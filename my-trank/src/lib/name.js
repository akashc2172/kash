export function displayName(raw) {
  if (!raw) return "";
  const i = raw.indexOf(",");
  if (i >= 0) {
    const last = raw.slice(0, i).trim();
    const first = raw.slice(i + 1).trim();
    return `${first} ${last}`.replace(/\s+/g, " ").trim();
  }
  return raw;
}

export function heightFromInches(inches) {
  const n = Number(inches);
  if (!Number.isFinite(n)) return "";
  const ft = Math.floor(n / 12);
  const inch = Math.round(n % 12);
  return `${ft}-${inch}`;
}