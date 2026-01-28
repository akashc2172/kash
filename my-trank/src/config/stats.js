export const PERMANENT_COLS = ["exp", "key", "team", "height"] as const;

// default visible (besides permanent)
export const DEFAULT_STATS = [
  "porpag","bpm","ts","efg","oreb_rate","dreb_rate",
  "ast","to","blk","stl","ftr",
  "total RAPM",
  "2pa","2p%","3pa","3p%",
  "g"
];

export const DEFAULT_GAMES_MIN = 10;

// do NOT heatmap these
export const NO_HEATMAP = new Set([
  "2pa","3pa","2pm","3pm","two_a","two_m","three_a","three_m",
  "g","height","key","team","exp"
]);

export const TEXT_FIELDS = new Set(["key","team","conf","pos"]);
