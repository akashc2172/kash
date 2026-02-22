const STORAGE_KEY = "torvikDatasets";
const REMOTE_MANIFEST_URL = "https://checkthesheets.com/torvik-datasets/manifest.json";
const INJECTED_HEADER_ATTR = "data-torvik-overlay-header";
const INJECTED_CELL_ATTR = "data-torvik-overlay-cell";
const INJECTED_ROW_ATTR = "data-torvik-overlay-row";
const overlayState = {
  dataset: null,
  datasetRows: null,
  observer: null,
  reapplying: false,
  sortColIndex: 0,
  sortTrackerInstalled: false
};

init().catch((err) => {
  showBanner(`Torvik Overlay error: ${err.message}`, true);
});

async function init() {
  const season = getSeasonFromUrl(window.location.href);
  if (!season) {
    return;
  }

  const dataset = await resolveDatasetForSeason(season);

  if (!dataset) {
    showBanner(`No dataset configured for season ${season}. Open extension popup once or use checkthesheets manifest.`, false);
    return;
  }

  const rows = await fetchDatasetRows(dataset.url);
  if (!rows.length) {
    showBanner(`Dataset for ${season} returned no rows.`, true);
    return;
  }

  const table = await waitForTargetTable(60000);
  if (!table) {
    showBanner(`Could not find player stats table on this page. ${collectTableDiagnostics()}`, true);
    return;
  }

  injectColumns(table, rows, dataset);
  installReapplyObserver();
  installSortTracker();
}

async function resolveDatasetForSeason(season) {
  const datasets = await loadDatasets();
  const local = datasets.find((d) => Number(d.season) === season);
  if (local) {
    return local;
  }

  const remote = await loadDatasetFromRemoteManifest(season);
  if (remote) {
    return remote;
  }

  return null;
}

function getSeasonFromUrl(url) {
  const parsed = new URL(url);
  const year = parsed.searchParams.get("year");
  const n = Number(year);
  return Number.isFinite(n) ? n : null;
}

async function loadDatasets() {
  const data = await chrome.storage.sync.get(STORAGE_KEY);
  return Array.isArray(data[STORAGE_KEY]) ? data[STORAGE_KEY] : [];
}

async function loadDatasetFromRemoteManifest(season) {
  try {
    const text = await fetchTextViaBackground(REMOTE_MANIFEST_URL);
    if (!text) {
      return null;
    }

    const payload = JSON.parse(text);
    if (!payload || !Array.isArray(payload.datasets)) {
      return null;
    }

    const match = payload.datasets.find((d) => Number(d.season) === Number(season));
    if (!match || !match.url) {
      return null;
    }

    const absoluteUrl = new URL(String(match.url), REMOTE_MANIFEST_URL).toString();
    const primary = Array.isArray(match.match?.primary) ? match.match.primary : [];
    const fallback = Array.isArray(match.match?.fallback) ? match.match.fallback : [];
    const keyCandidates = [...primary, ...fallback].map((v) => String(v).trim()).filter(Boolean);
    const matchKey = keyCandidates[0] || "player";
    const teamKey = keyCandidates[1] || "team";
    const defaultColumns = Array.isArray(match.inject_columns)
      ? match.inject_columns.map((v) => String(v).trim()).filter(Boolean)
      : [];

    return {
      id: `remote-${season}`,
      season: Number(season),
      name: match.name || `CTS Intl Overlay ${season}`,
      url: absoluteUrl,
      matchKey,
      teamKey,
      columns: defaultColumns
    };
  } catch (_err) {
    return null;
  }
}

async function fetchDatasetRows(url) {
  const text = await fetchTextViaBackground(url);
  const trimmed = text.trim();
  if (!trimmed) {
    return [];
  }

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed);
    return normalizeJsonRows(parsed);
  }

  return parseCsv(trimmed);
}

async function fetchTextViaBackground(url) {
  const response = await chrome.runtime.sendMessage({
    type: "fetchText",
    url: String(url)
  });

  if (!response || !response.ok) {
    const status = response?.status || 0;
    const err = response?.error ? `: ${response.error}` : "";
    throw new Error(`Fetch failed (${status}) for ${url}${err}`);
  }

  return String(response.text || "");
}

function normalizeJsonRows(payload) {
  if (Array.isArray(payload)) {
    return payload.filter((r) => r && typeof r === "object");
  }
  if (payload && Array.isArray(payload.rows)) {
    return payload.rows.filter((r) => r && typeof r === "object");
  }
  return [];
}

function parseCsv(csvText) {
  const lines = csvText.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 2) {
    return [];
  }

  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    const row = {};
    headers.forEach((header, i) => {
      row[header] = values[i] ?? "";
    });
    return row;
  });
}

function splitCsvLine(line) {
  const output = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      output.push(current.trim());
      current = "";
      continue;
    }

    current += char;
  }

  output.push(current.trim());
  return output;
}

function findTargetTable() {
  const inTble = Array.from(document.querySelectorAll("#tble table")).filter(isCandidateTable);
  const strongMatch = inTble.find(isLikelyStatsTable);
  if (strongMatch) {
    return strongMatch;
  }
  if (inTble.length) {
    return pickLargestTable(inTble);
  }

  const allTables = Array.from(document.querySelectorAll("table")).filter(isCandidateTable);
  const fallbackStrong = allTables.find(isLikelyStatsTable);
  if (fallbackStrong) {
    return fallbackStrong;
  }
  if (allTables.length) {
    return pickLargestTable(allTables);
  }

  return null;
}

async function waitForTargetTable(timeoutMs) {
  const found = findTargetTable();
  if (found) {
    return found;
  }

  return new Promise((resolve) => {
    let settled = false;

    const finish = (value) => {
      if (settled) {
        return;
      }
      settled = true;
      observer.disconnect();
      clearInterval(pollTimer);
      clearTimeout(timeoutTimer);
      resolve(value);
    };

    const check = () => {
      const table = findTargetTable();
      if (table) {
        finish(table);
      }
    };

    const observeNode = document.querySelector("#tble") || document.body;
    const observer = new MutationObserver(check);
    observer.observe(observeNode, { childList: true, subtree: true });

    const pollTimer = setInterval(check, 400);
    const timeoutTimer = setTimeout(() => finish(null), timeoutMs);
  });
}

function isLikelyStatsTable(table) {
  const rows = Array.from(table.querySelectorAll("tr"));
  if (rows.length < 2) {
    return false;
  }

  const sampleRow = rows.find((tr) => tr.querySelectorAll("td").length >= 4);
  if (!sampleRow) {
    return false;
  }

  const text = (table.textContent || "").toLowerCase();
  if (text.includes("show 100 more")) {
    return false;
  }

  return true;
}

function isCandidateTable(table) {
  if (!table || table.id === "filtertable") {
    return false;
  }
  if (table.closest("#limits")) {
    return false;
  }
  return true;
}

function pickLargestTable(tables) {
  let best = null;
  let bestRows = -1;
  for (const table of tables) {
    const rows = table.querySelectorAll("tr").length;
    if (rows > bestRows) {
      bestRows = rows;
      best = table;
    }
  }
  return best;
}

function collectTableDiagnostics() {
  const inTble = document.querySelectorAll("#tble table").length;
  const allTables = Array.from(document.querySelectorAll("table"));
  const candidates = allTables.filter(isCandidateTable);
  const rowSizes = candidates
    .map((t) => t.querySelectorAll("tr").length)
    .sort((a, b) => b - a)
    .slice(0, 5)
    .join(",");
  return `diag: #tble tables=${inTble}, candidates=${candidates.length}, topRows=[${rowSizes}]`;
}

function injectColumns(table, datasetRows, dataset) {
  clearOldInjectedColumns(table);

  const key = (dataset.matchKey || "player").trim();
  const teamKey = (dataset.teamKey || "team").trim();

  const headerRow =
    table.querySelector("thead tr:last-child") ||
    table.querySelector("tr");
  if (!headerRow) {
    showBanner("Could not find table header row.", true);
    return;
  }

  const { playerIndex, teamIndex } = getPlayerTeamColumnIndices(table, headerRow);
  const bodyRows = Array.from(table.querySelectorAll("tbody tr"));
  const dataRows = bodyRows.length
    ? bodyRows
    : Array.from(table.querySelectorAll("tr")).filter((tr) => tr.querySelectorAll("td").length > 0);
  const existingFull = new Set();
  const existingPlayer = new Set();
  for (const tr of dataRows) {
    const cells = tr.querySelectorAll("td");
    const playerName = cells[playerIndex]?.textContent?.trim() || "";
    const teamName = cells[teamIndex]?.textContent?.trim() || "";
    const playerNorm = normalizeName(playerName);
    const teamNorm = normalizeName(teamName);
    if (!playerNorm) {
      continue;
    }
    existingPlayer.add(playerNorm);
    existingFull.add(`${playerNorm}||${teamNorm}`);
  }

  const normalizedRows = datasetRows
    .map((row) => normalizeObjectKeys(row))
    .filter((row) => {
      const playerVal = row[key.toLowerCase()];
      const teamVal = row[teamKey.toLowerCase()];
      const playerNorm = normalizeName(String(playerVal || ""));
      const teamNorm = normalizeName(String(teamVal || ""));
      if (!playerNorm) {
        return false;
      }
      if (existingFull.has(`${playerNorm}||${teamNorm}`) || existingPlayer.has(playerNorm)) {
        return false;
      }
      return true;
    });

  const appendedCount = appendIntegratedRows(table, headerRow, normalizedRows, {
    key: key.toLowerCase(),
    teamKey: teamKey.toLowerCase(),
    playerIndex,
    teamIndex
  });

  showBanner(`Loaded ${dataset.name} (${dataset.season}). Added ${appendedCount} integrated rows.`, false);

  overlayState.dataset = dataset;
  overlayState.datasetRows = datasetRows;
}

function appendIntegratedRows(table, headerRow, rowsToAdd, params) {
  const { key, teamKey, playerIndex, teamIndex } = params;
  if (!rowsToAdd.length) {
    return 0;
  }

  const tbody = getPrimaryDataTbody(table);
  const columnCount = headerRow.querySelectorAll("th,td").length;
  const headers = Array.from(headerRow.querySelectorAll("th,td")).map((h) => normalizeName(h.textContent || ""));
  const templateRow = findTemplateRow(tbody, columnCount);
  const templateCells = templateRow ? Array.from(templateRow.querySelectorAll("td")) : [];

  let maxRank = 0;
  const allRows = Array.from(table.querySelectorAll("tbody tr"));
  for (const tr of allRows) {
    const first = tr.querySelector("td");
    const n = Number((first?.textContent || "").trim());
    if (Number.isFinite(n) && n > maxRank) {
      maxRank = n;
    }
  }

  for (const row of rowsToAdd) {
    maxRank += 1;
    const tr = document.createElement("tr");
    tr.setAttribute(INJECTED_ROW_ATTR, "1");
    if (templateRow) {
      tr.className = templateRow.className;
    }

    for (let i = 0; i < columnCount; i += 1) {
      const td = templateCells[i]
        ? templateCells[i].cloneNode(false)
        : document.createElement("td");
      if (td.hasAttribute("id")) {
        td.removeAttribute("id");
      }
      td.textContent = resolveIntegratedCellValue(headers[i], row, {
        rank: maxRank,
        key,
        teamKey,
        playerIndex,
        teamIndex,
        colIndex: i
      });
      tr.appendChild(td);
    }

    insertRowByCurrentSort(table, tbody, tr);
  }

  resortCombinedRows(table, tbody);

  return rowsToAdd.length;
}

function findTemplateRow(tbody, columnCount) {
  const rows = Array.from(tbody.querySelectorAll("tr"));
  return rows.find((tr) => {
    if (tr.hasAttribute(INJECTED_ROW_ATTR)) {
      return false;
    }
    const text = normalizeName(tr.textContent || "");
    if (text.includes("show 100 more") || text.includes("show chart")) {
      return false;
    }
    return tr.querySelectorAll("td").length >= columnCount;
  }) || null;
}

function resolveIntegratedCellValue(headerLabel, row, ctx) {
  const { rank, key, teamKey, playerIndex, teamIndex, colIndex } = ctx;
  const player = formatCell(row[key] || "");
  const team = formatCell(row[teamKey] || row.team || "INTL");
  const conf = compactConf(row);
  const label = normalizeName(headerLabel || "");

  if (colIndex === playerIndex) {
    return player;
  }
  if (colIndex === teamIndex) {
    return team;
  }
  if (label === "player") {
    return player;
  }
  if (label === "team" || label === "tm") {
    return team;
  }
  if (label === "rk") {
    return String(rank);
  }
  if (label === "pick") {
    return "-";
  }
  if (label === "conf") {
    return conf;
  }
  if (label.includes("role")) {
    return formatCell(row.role || "INTL");
  }
  if (label === "g") {
    return formatNumeric(row.g);
  }
  if (label.includes("min")) {
    return formatNumeric(row.min_pct ?? row.min ?? row.mpg);
  }
  if (label.includes("prpg") || label.includes("porpag")) {
    return formatNumeric(row.porpag ?? row.ppg);
  }
  if (label === "bpm" || label.includes("box")) {
    return formatNumeric(row.bpm);
  }
  if (label === "obpm") {
    return formatNumeric(row.obpm);
  }
  if (label === "dbpm") {
    return formatNumeric(row.dbpm);
  }
  if (label.includes("ortg") || label.includes("o rating")) {
    return formatNumeric(row.ortg);
  }
  if (label.includes("drtg") || label.includes("d rtg")) {
    return formatNumeric(row.drtg ?? row.adrtg);
  }
  if (label.includes("usg") || label.includes("usage")) {
    return formatNumeric(row.usg, { pct: true });
  }
  if (label.includes("efg")) {
    return formatNumeric(row.efg, { pct: true });
  }
  if (label === "ts" || label.includes("ts")) {
    return formatNumeric(row.ts, { pct: true });
  }
  if (label === "or") {
    return formatNumeric(row.oreb_pct, { pct: true });
  }
  if (label === "dr") {
    return formatNumeric(row.dreb_pct, { pct: true });
  }
  if (label === "ast") {
    return formatNumeric(row.ast_pct, { pct: true });
  }
  if (label === "to") {
    return formatNumeric(row.tov_pct, { pct: true });
  }
  if (label === "a to") {
    return formatNumeric(row.ast_to);
  }
  if (label === "blk") {
    return formatNumeric(row.blk_pct, { pct: true });
  }
  if (label === "stl") {
    return formatNumeric(row.stl_pct, { pct: true });
  }
  if (label === "ftr") {
    return formatNumeric(row.ftr, { pct: true });
  }
  if (label.includes("3p 100")) {
    return formatNumeric(row.threepa_100);
  }
  if (label === "pts") {
    return formatNumeric(row.ppg);
  }
  if (label === "reb") {
    return formatNumeric(row.reb ?? row.rpg);
  }
  if (label === "3p") {
    return formatNumeric(row.threep_pct ?? row.three_pct, { pct: true });
  }

  return "";
}

function compactConf(row) {
  const rawConf = formatCell(row.conf || "");
  if (rawConf && rawConf.length <= 6) {
    return rawConf;
  }
  const league = formatCell(row.league || "");
  if (!league) {
    return "INTL";
  }
  const cleaned = league
    .replace(/\(intl\)/i, "")
    .replace(/french/gi, "FRA")
    .replace(/euroleague/gi, "EUROL")
    .replace(/eurocup/gi, "ECUP")
    .replace(/pro b/gi, "PROB")
    .replace(/jeep/gi, "JEEP")
    .replace(/turk\.?\s*bsl/gi, "BSL")
    .replace(/[^a-zA-Z0-9 ]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toUpperCase();
  if (!cleaned) {
    return "INTL";
  }
  return cleaned.length > 8 ? cleaned.slice(0, 8) : cleaned;
}

function formatNumeric(value, options = {}) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return String(value);
  }

  let out = n;
  if (options.pct && Math.abs(out) <= 1) {
    out *= 100;
  }

  if (Math.abs(out - Math.round(out)) < 1e-9) {
    return String(Math.round(out));
  }
  return String(Math.round(out * 10) / 10);
}

function getPlayerTeamColumnIndices(table, headerRow) {
  const defaultGuess = { playerIndex: 4, teamIndex: 5 };

  const headerCells = Array.from(headerRow.querySelectorAll("th,td"));
  if (!headerCells.length) {
    return defaultGuess;
  }

  let playerIndex = -1;
  let teamIndex = -1;

  headerCells.forEach((cell, idx) => {
    const label = normalizeName(cell.textContent || "");
    if (playerIndex === -1 && (label === "player" || label.includes("player"))) {
      playerIndex = idx;
    }
    if (teamIndex === -1 && (label === "team" || label === "tm" || label.includes("team"))) {
      teamIndex = idx;
    }
  });

  if (playerIndex !== -1 && teamIndex !== -1) {
    return { playerIndex, teamIndex };
  }

  // Strong fallback: find player/team link columns in real rows.
  const sampleRows = Array.from(table.querySelectorAll("tbody tr")).slice(0, 10);
  for (const row of sampleRows) {
    const cells = Array.from(row.querySelectorAll("td"));
    let p = -1;
    let t = -1;
    cells.forEach((cell, idx) => {
      const a = cell.querySelector("a[href]");
      const href = a?.getAttribute("href") || "";
      if (p === -1 && href.includes("player")) {
        p = idx;
      }
      if (t === -1 && href.includes("team")) {
        t = idx;
      }
    });
    if (p !== -1 && t !== -1) {
      return { playerIndex: p, teamIndex: t };
    }
  }

  // Fallback from first body row if headers are non-standard.
  const sampleRow = table.querySelector("tbody tr");
  if (sampleRow) {
    const cells = Array.from(sampleRow.querySelectorAll("td")).map((td) =>
      normalizeName(td.textContent || "")
    );
    for (let i = 0; i < cells.length; i += 1) {
      const val = cells[i];
      if (playerIndex === -1 && /\b[a-z]{2,}\b/.test(val) && val.split(" ").length >= 2) {
        playerIndex = i;
      }
      if (teamIndex === -1 && val.length > 0 && val.length <= 25) {
        teamIndex = i + 1;
      }
      if (playerIndex !== -1 && teamIndex !== -1) {
        break;
      }
    }
  }

  if (playerIndex === -1 || teamIndex === -1) {
    return defaultGuess;
  }
  return { playerIndex, teamIndex };
}

function findControlRow(tbody) {
  const rows = Array.from(tbody.querySelectorAll("tr"));
  return rows.find((tr) => {
    const text = normalizeName(tr.textContent || "");
    return text.includes("show 100 more") || text.includes("show chart");
  }) || null;
}

function findControlRowGlobal(table) {
  const rows = Array.from(table.querySelectorAll("tr"));
  return rows.find((tr) => {
    const text = normalizeName(tr.textContent || "");
    return text.includes("show 100 more") || text.includes("show chart");
  }) || null;
}

function insertRowByCurrentSort(table, tbody, newRow) {
  const sortCol = Number.isFinite(overlayState.sortColIndex) ? overlayState.sortColIndex : 0;
  const controlRow = findControlRow(tbody);
  const rows = Array.from(tbody.querySelectorAll("tr")).filter((tr) => {
    if (tr === newRow) {
      return false;
    }
    if (tr === controlRow) {
      return false;
    }
    if (tr.querySelectorAll("td").length === 0) {
      return false;
    }
    return true;
  });

  const nativeRows = rows.filter((tr) => !tr.hasAttribute(INJECTED_ROW_ATTR));
  const direction = inferDirectionFromRows(nativeRows.length ? nativeRows : rows, sortCol);
  const newVal = rowCellValue(newRow, sortCol);

  let inserted = false;
  for (const row of rows) {
    const existingVal = rowCellValue(row, sortCol);
    const cmp = compareMixed(newVal, existingVal);
    const goesBefore = direction === "asc" ? cmp <= 0 : cmp >= 0;
    if (goesBefore) {
      tbody.insertBefore(newRow, row);
      inserted = true;
      break;
    }
  }

  if (!inserted) {
    if (controlRow && controlRow.parentElement === tbody) {
      tbody.insertBefore(newRow, controlRow);
    } else {
      tbody.appendChild(newRow);
    }
  }
}

function resortCombinedRows(table, tbody) {
  const sortCol = Number.isFinite(overlayState.sortColIndex) ? overlayState.sortColIndex : 0;
  const controlRow = findControlRow(tbody);
  const rows = Array.from(tbody.querySelectorAll("tr")).filter((tr) => {
    if (tr === controlRow) {
      return false;
    }
    return tr.querySelectorAll("td").length > 0;
  });

  if (rows.length < 2) {
    return;
  }

  const nativeRows = rows.filter((tr) => !tr.hasAttribute(INJECTED_ROW_ATTR));
  const direction = inferDirectionFromRows(nativeRows.length ? nativeRows : rows, sortCol);

  rows.sort((a, b) => {
    const av = rowCellValue(a, sortCol);
    const bv = rowCellValue(b, sortCol);
    const cmp = compareMixed(av, bv);
    return direction === "asc" ? cmp : -cmp;
  });

  for (const row of rows) {
    if (controlRow && controlRow.parentElement === tbody) {
      tbody.insertBefore(row, controlRow);
    } else {
      tbody.appendChild(row);
    }
  }
}

function rowCellValue(row, colIndex) {
  const cells = row.querySelectorAll("td");
  return (cells[colIndex]?.textContent || "").trim();
}

function inferDirectionFromRows(rows, colIndex) {
  if (rows.length < 2) {
    return "asc";
  }

  for (let i = 1; i < rows.length; i += 1) {
    const prev = rowCellValue(rows[i - 1], colIndex);
    const curr = rowCellValue(rows[i], colIndex);
    if (!String(prev || "").trim() || !String(curr || "").trim()) {
      continue;
    }
    const cmp = compareMixed(prev, curr);
    if (cmp < 0) {
      return "asc";
    }
    if (cmp > 0) {
      return "desc";
    }
  }
  return "asc";
}

function compareMixed(a, b) {
  const na = parseMaybeNumber(a);
  const nb = parseMaybeNumber(b);
  if (na !== null && nb !== null) {
    if (na === nb) {
      return 0;
    }
    return na < nb ? -1 : 1;
  }
  const sa = normalizeName(String(a || ""));
  const sb = normalizeName(String(b || ""));
  if (sa === sb) {
    return 0;
  }
  return sa < sb ? -1 : 1;
}

function parseMaybeNumber(value) {
  const s = String(value || "").replace(/,/g, "").replace(/%/g, "").trim();
  if (!s) {
    return null;
  }
  const m = s.match(/-?\d+(\.\d+)?/);
  if (!m) {
    return null;
  }
  const n = Number(m[0]);
  return Number.isFinite(n) ? n : null;
}

function installSortTracker() {
  if (overlayState.sortTrackerInstalled) {
    return;
  }
  overlayState.sortTrackerInstalled = true;
  document.addEventListener("click", (event) => {
    const th = event.target instanceof Element ? event.target.closest("th") : null;
    if (!th) {
      return;
    }
    const table = findTargetTable();
    if (!table || !table.contains(th)) {
      return;
    }
    const headerRow = table.querySelector("thead tr:last-child") || table.querySelector("tr");
    if (!headerRow) {
      return;
    }
    const headers = Array.from(headerRow.querySelectorAll("th,td"));
    const idx = headers.indexOf(th);
    if (idx >= 0) {
      overlayState.sortColIndex = idx;
    }
  }, true);
}

function getPrimaryDataTbody(table) {
  const tbodies = Array.from(table.tBodies || []);
  if (!tbodies.length) {
    return table.appendChild(document.createElement("tbody"));
  }

  let best = tbodies[0];
  let bestScore = -1;
  for (const tbody of tbodies) {
    const rows = Array.from(tbody.querySelectorAll("tr"));
    let score = 0;
    for (const tr of rows) {
      if (tr.hasAttribute(INJECTED_ROW_ATTR)) {
        continue;
      }
      if (findControlRow(tbody) === tr) {
        continue;
      }
      if (tr.querySelectorAll("td").length > 0) {
        score += 1;
      }
    }
    if (score > bestScore) {
      bestScore = score;
      best = tbody;
    }
  }
  return best;
}

function installReapplyObserver() {
  if (overlayState.observer) {
    return;
  }
  const target = document.querySelector("#tble") || document.body;
  if (!target) {
    return;
  }

  let timer = null;
  overlayState.observer = new MutationObserver(() => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      if (overlayState.reapplying || !overlayState.dataset || !overlayState.datasetRows) {
        return;
      }
      const table = findTargetTable();
      if (!table) {
        return;
      }
      const hasRows = table.querySelector(`[${INJECTED_ROW_ATTR}]`);
      if (hasRows) {
        return;
      }
      overlayState.reapplying = true;
      try {
        injectColumns(table, overlayState.datasetRows, overlayState.dataset);
      } finally {
        overlayState.reapplying = false;
      }
    }, 120);
  });

  overlayState.observer.observe(target, { childList: true, subtree: true });
}

function clearOldInjectedColumns(table) {
  table.querySelectorAll(`[${INJECTED_HEADER_ATTR}]`).forEach((n) => n.remove());
  table.querySelectorAll(`[${INJECTED_CELL_ATTR}]`).forEach((n) => n.remove());
  table.querySelectorAll(`[${INJECTED_ROW_ATTR}]`).forEach((n) => n.remove());
}

function normalizeObjectKeys(obj) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    out[String(k).toLowerCase().trim()] = v;
  }
  return out;
}

function normalizeName(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9 ]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function formatCell(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function showBanner(message, isError) {
  let banner = document.getElementById("torvik-overlay-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "torvik-overlay-banner";
    const target = document.body.firstElementChild || document.body;
    target.before(banner);
  }

  banner.classList.toggle("error", Boolean(isError));
  banner.textContent = message;
}
