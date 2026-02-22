const STORAGE_KEY = "torvikDatasets";
const REMOTE_MANIFEST_URL = "https://checkthesheets.com/torvik-datasets/manifest.json";

const form = document.getElementById("dataset-form");
const list = document.getElementById("dataset-list");
const remoteList = document.getElementById("remote-dataset-list");
const quickStatus = document.getElementById("quick-status");
const loadRemoteBtn = document.getElementById("load-remote-btn");
const installAllBtn = document.getElementById("install-all-btn");

let remoteDatasets = [];

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const dataset = {
    id: crypto.randomUUID(),
    name: document.getElementById("name").value.trim(),
    season: Number(document.getElementById("season").value),
    url: document.getElementById("url").value.trim(),
    matchKey: document.getElementById("matchKey").value.trim(),
    teamKey: document.getElementById("teamKey").value.trim(),
    columns: parseColumns(document.getElementById("columns").value)
  };

  if (!dataset.name || !dataset.season || !dataset.url || !dataset.matchKey || !dataset.columns.length) {
    setStatus("Fill all fields first.", true);
    return;
  }

  await upsertDataset(dataset);
  resetManualForm();
  setStatus(`Saved season ${dataset.season}.`, false);
});

loadRemoteBtn.addEventListener("click", async () => {
  await loadRemoteDatasets();
});

installAllBtn.addEventListener("click", async () => {
  if (!remoteDatasets.length) {
    await loadRemoteDatasets();
  }
  if (!remoteDatasets.length) {
    setStatus("No remote datasets found.", true);
    return;
  }

  try {
    installAllBtn.disabled = true;
    const converted = remoteDatasets.map((item) => convertRemoteToLocal(item));
    const count = await upsertDatasetsBatch(converted);
    setStatus(`Installed ${count} seasons.`, false);
  } catch (err) {
    setStatus(`Install failed: ${err.message}`, true);
  } finally {
    installAllBtn.disabled = false;
  }
});

async function loadRemoteDatasets() {
  try {
    setStatus("Loading remote seasons...", false);
    const response = await fetch(REMOTE_MANIFEST_URL, { credentials: "omit" });
    if (!response.ok) {
      throw new Error(`Manifest fetch failed (${response.status})`);
    }

    const payload = await response.json();
    const datasets = Array.isArray(payload.datasets) ? payload.datasets : [];

    remoteDatasets = datasets
      .filter((d) => Number.isFinite(Number(d.season)) && d.url)
      .sort((a, b) => Number(b.season) - Number(a.season));

    renderRemoteList();
    setStatus(`Loaded ${remoteDatasets.length} remote seasons.`, false);
  } catch (err) {
    setStatus(`Could not load remote seasons: ${err.message}`, true);
  }
}

function renderRemoteList() {
  remoteList.innerHTML = "";
  if (!remoteDatasets.length) {
    return;
  }

  remoteDatasets.forEach((item) => {
    const card = document.createElement("article");
    card.className = "remote-card";

    const top = document.createElement("div");
    top.className = "remote-top";
    top.innerHTML = `<span>${escapeHtml(item.name || `Season ${item.season}`)}</span><span>${item.season}</span>`;

    const meta = document.createElement("div");
    meta.className = "remote-meta";
    const cols = Array.isArray(item.inject_columns) ? item.inject_columns.join(", ") : "";
    meta.textContent = cols ? `Default columns: ${cols}` : "Default columns: auto";

    const actions = document.createElement("div");
    actions.className = "remote-actions";

    const useBtn = document.createElement("button");
    useBtn.type = "button";
    useBtn.textContent = `Use ${item.season}`;
    useBtn.addEventListener("click", async () => {
      await upsertDataset(convertRemoteToLocal(item));
      setStatus(`Installed season ${item.season}.`, false);
    });

    const fillBtn = document.createElement("button");
    fillBtn.type = "button";
    fillBtn.className = "secondary";
    fillBtn.textContent = "Fill Form";
    fillBtn.addEventListener("click", () => fillManualForm(convertRemoteToLocal(item)));

    actions.appendChild(useBtn);
    actions.appendChild(fillBtn);

    card.appendChild(top);
    card.appendChild(meta);
    card.appendChild(actions);
    remoteList.appendChild(card);
  });
}

function convertRemoteToLocal(remoteItem) {
  const match = remoteItem.match || {};
  const primary = Array.isArray(match.primary) ? match.primary : [];
  const fallback = Array.isArray(match.fallback) ? match.fallback : [];
  const keys = [...primary, ...fallback].map((v) => String(v).trim()).filter(Boolean);
  const matchKey = keys[0] || "player";
  const teamKey = keys[1] || "team";
  const columns = Array.isArray(remoteItem.inject_columns)
    ? remoteItem.inject_columns.map((v) => String(v).trim()).filter(Boolean)
    : defaultColumns();

  return {
    id: `remote-${remoteItem.season}`,
    name: remoteItem.name || `CTS Intl Overlay ${remoteItem.season}`,
    season: Number(remoteItem.season),
    url: new URL(String(remoteItem.url), REMOTE_MANIFEST_URL).toString(),
    matchKey,
    teamKey,
    columns
  };
}

function fillManualForm(dataset) {
  document.getElementById("name").value = dataset.name;
  document.getElementById("season").value = String(dataset.season);
  document.getElementById("url").value = dataset.url;
  document.getElementById("matchKey").value = dataset.matchKey;
  document.getElementById("teamKey").value = dataset.teamKey || "team";
  document.getElementById("columns").value = dataset.columns.join(",");
  setStatus(`Filled form for ${dataset.season}.`, false);
}

async function loadDatasets() {
  const data = await chrome.storage.sync.get(STORAGE_KEY);
  return Array.isArray(data[STORAGE_KEY]) ? data[STORAGE_KEY] : [];
}

async function upsertDataset(dataset) {
  const datasets = await loadDatasets();
  const idx = datasets.findIndex((d) => Number(d.season) === Number(dataset.season));

  if (idx >= 0) {
    datasets[idx] = { ...datasets[idx], ...dataset, id: datasets[idx].id };
  } else {
    datasets.push(dataset);
  }

  await chrome.storage.sync.set({ [STORAGE_KEY]: datasets });
  await renderList();
}

async function upsertDatasetsBatch(incoming) {
  const datasets = await loadDatasets();
  const bySeason = new Map();

  for (const d of datasets) {
    bySeason.set(Number(d.season), d);
  }

  for (const item of incoming) {
    const season = Number(item.season);
    const existing = bySeason.get(season);
    if (existing) {
      bySeason.set(season, { ...existing, ...item, id: existing.id });
    } else {
      bySeason.set(season, item);
    }
  }

  const merged = Array.from(bySeason.values());
  await chrome.storage.sync.set({ [STORAGE_KEY]: merged });
  await renderList();
  return incoming.length;
}

async function removeDataset(id) {
  const datasets = await loadDatasets();
  const next = datasets.filter((d) => d.id !== id);
  await chrome.storage.sync.set({ [STORAGE_KEY]: next });
  await renderList();
}

async function renderList() {
  const datasets = await loadDatasets();
  list.innerHTML = "";

  if (!datasets.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No datasets yet.";
    list.appendChild(empty);
    return;
  }

  datasets
    .sort((a, b) => Number(b.season) - Number(a.season))
    .forEach((d) => {
      const card = document.createElement("article");
      card.className = "dataset-card";

      const top = document.createElement("div");
      top.className = "dataset-top";
      top.innerHTML = `<span>${escapeHtml(d.name)}</span><span>${d.season}</span>`;

      const meta = document.createElement("div");
      const teamPart = d.teamKey ? ` + ${d.teamKey}` : "";
      meta.textContent = `Key: ${d.matchKey}${teamPart} | Inject: ${d.columns.join(", ")}`;

      const url = document.createElement("div");
      url.className = "dataset-url";
      url.textContent = d.url;

      const actions = document.createElement("div");
      actions.className = "dataset-actions";
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.textContent = "Delete";
      removeBtn.addEventListener("click", () => removeDataset(d.id));
      actions.appendChild(removeBtn);

      card.appendChild(top);
      card.appendChild(meta);
      card.appendChild(url);
      card.appendChild(actions);
      list.appendChild(card);
    });
}

function parseColumns(text) {
  const columns = (text || "")
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);
  return columns.length ? columns : defaultColumns();
}

function defaultColumns() {
  return [
    "conf",
    "g",
    "mpg",
    "ppg",
    "usg",
    "ast_pct",
    "tov_pct",
    "ast_to",
    "oreb_pct",
    "dreb_pct",
    "ftr",
    "blk_pct",
    "stl_pct",
    "efg",
    "ts",
    "porpag",
    "bpm",
    "per"
  ];
}

function resetManualForm() {
  form.reset();
  document.getElementById("matchKey").value = "player";
  document.getElementById("teamKey").value = "team";
  document.getElementById("columns").value = defaultColumns().join(",");
}

function setStatus(message, isError) {
  quickStatus.textContent = message;
  quickStatus.classList.toggle("error", Boolean(isError));
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

(async function boot() {
  document.getElementById("columns").value = defaultColumns().join(",");
  await renderList();
  await loadRemoteDatasets();
})();
