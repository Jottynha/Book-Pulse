// URL /exec do App da Web criado com anotacoes_apps_script.gs. Vazia, a
// anotação fica só neste navegador e precisa ser exportada em CSV.
const SHEETS_URL = "https://script.google.com/macros/s/AKfycbzm8Mzalx-8xEHibiyMpxAu-t4yM4S_sWbx7oD_mkeBg59S1JQ4njTpHWRDxTpT4DOhwQ/exec";
const CSV_PATH = "processed/07_sentimento/amostra_anotacao.csv";
const STORE_PREFIX = "bookpulse-anotacao-v1:";
const GOAL = 100;
const SWIPE_THRESHOLD = 110;
const PREFETCH = 2;
const LABEL_NAMES = { positive: "positiva", negative: "negativa", neutral: "neutra", mixed: "mista", unclear: "incerta" };
const ENGLISH_HINTS = new Set(["the", "and", "is", "it", "to", "of", "this", "was", "i", "a", "in", "that", "book", "but", "for", "with", "not", "you", "my", "read"]);

const $ = (selector) => document.querySelector(selector);
const state = { reviews: [], index: 0, annotator: "", labels: {}, translations: {}, worker: null, variant: "big", workerStatus: "", pending: [], syncing: false, syncError: "" };

const storage = {
  get(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* armazenamento indisponível */ }
  },
};

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { row.push(field); field = ""; }
    else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [header, ...body] = rows.filter((item) => item.some((value) => value !== ""));
  const idIndex = header.indexOf("review_id");
  const textIndex = header.indexOf("text_for_inference") >= 0 ? header.indexOf("text_for_inference") : header.indexOf("review_text");
  if (idIndex < 0 || textIndex < 0) throw new Error("CSV precisa das colunas review_id e text_for_inference.");
  return body.map((item) => ({ id: item[idIndex], text: item[textIndex] || "" })).filter((item) => item.id);
}

const csvField = (value) => /[",\n\r]/.test(String(value)) ? `"${String(value).replaceAll('"', '""')}"` : String(value);

function looksEnglish(text) {
  const words = text.toLowerCase().match(/[a-z']+/g) || [];
  if (words.length < 4) return words.length > 0;
  return words.filter((word) => ENGLISH_HINTS.has(word)).length / words.length >= 0.08;
}

function storeKey() { return `${STORE_PREFIX}${state.annotator}`; }

function save() {
  storage.set(storeKey(), { labels: state.labels, index: state.index });
  const cache = storage.get(`${STORE_PREFIX}translations:${state.variant}`, {});
  for (const [id, entry] of Object.entries(state.translations)) if (entry.done && entry.variant === state.variant) cache[id] = entry.text;
  storage.set(`${STORE_PREFIX}translations:${state.variant}`, cache);
}

function renderSync() {
  const element = $("#sync-status");
  if (!SHEETS_URL) { element.textContent = "· salvo só neste navegador"; return; }
  const waiting = state.pending.length;
  element.textContent = waiting
    ? `· ${waiting} aguardando envio${state.syncError ? " (sem conexão com a planilha, tentando de novo)" : ""}`
    : "· tudo salvo na planilha";
  element.classList.toggle("sync-warning", Boolean(waiting && state.syncError));
}

async function sync() {
  if (!SHEETS_URL || state.syncing || !state.pending.length) { renderSync(); return; }
  state.syncing = true;
  const batch = state.pending.slice(0, 100);
  try {
    // text/plain evita o preflight CORS, que o Apps Script não responde.
    const response = await fetch(SHEETS_URL, { method: "POST", body: JSON.stringify({ items: batch }) });
    const result = await response.json();
    if (!result.ok) throw new Error(result.error || "falha na planilha");
    const sent = new Set(batch.map((item) => `${item.review_id}|${item.annotated_at}`));
    state.pending = state.pending.filter((item) => !sent.has(`${item.review_id}|${item.annotated_at}`));
    state.syncError = "";
  } catch (error) {
    state.syncError = String(error.message || error);
  } finally {
    state.syncing = false;
    storage.set(`${STORE_PREFIX}pending`, state.pending);
    renderSync();
  }
  if (state.pending.length && !state.syncError) sync();
}

async function pullRemote() {
  if (!SHEETS_URL) return;
  try {
    const response = await fetch(`${SHEETS_URL}?annotator=${encodeURIComponent(state.annotator)}`);
    const { rows = [] } = await response.json();
    for (const row of rows) {
      const local = state.labels[row.review_id];
      if (!local || row.annotated_at > local.at) {
        state.labels[row.review_id] = { label: row.label, uncertain: String(row.uncertain) === "1", at: row.annotated_at };
      }
    }
  } catch {
    state.syncError = "não foi possível ler a planilha";
  }
}

function loadTranslationCache() {
  state.translations = {};
  for (const [id, text] of Object.entries(storage.get(`${STORE_PREFIX}translations:${state.variant}`, {}))) {
    state.translations[id] = { text, done: true, variant: state.variant };
  }
}

function startWorker() {
  if (state.worker) return;
  try {
    state.worker = new Worker("traducao-worker.js", { type: "module" });
  } catch {
    state.workerStatus = "tradução indisponível neste navegador";
    return;
  }
  state.worker.onmessage = ({ data }) => {
    if (data.type === "status") { state.workerStatus = data.message; renderTranslation(); return; }
    const entry = state.translations[data.id] || (state.translations[data.id] = {});
    if (data.type === "partial") Object.assign(entry, { text: data.text, done: false });
    if (data.type === "done") { Object.assign(entry, { text: data.text, done: true, variant: data.variant }); state.workerStatus = ""; save(); }
    if (data.type === "error") Object.assign(entry, { error: data.message, done: true });
    if (state.reviews[state.index]?.id === data.id) renderTranslation();
  };
  state.worker.onerror = () => { state.workerStatus = "falha ao iniciar o tradutor"; renderTranslation(); };
}

function requestTranslations() {
  if (!$("#show-translation").checked) return;
  startWorker();
  if (!state.worker) return;
  for (let offset = PREFETCH; offset >= 0; offset--) {
    const review = state.reviews[state.index + offset];
    if (!review || !looksEnglish(review.text)) continue;
    const entry = state.translations[review.id];
    if (entry && (entry.done || entry.pending)) { if (offset === 0 && !entry.done) state.worker.postMessage({ type: "translate", id: review.id, text: review.text, variant: state.variant, priority: true }); continue; }
    state.translations[review.id] = { pending: true, text: "" };
    state.worker.postMessage({ type: "translate", id: review.id, text: review.text, variant: state.variant, priority: offset === 0 });
  }
}

function renderTranslation() {
  const review = state.reviews[state.index];
  const block = $("#translation-block");
  block.hidden = !$("#show-translation").checked || !review;
  if (block.hidden) return;
  const entry = state.translations[review.id];
  if (!looksEnglish(review.text)) {
    $("#translation").textContent = "";
    $("#translation-status").textContent = "· resenha não parece estar em inglês; tradução desativada";
  } else if (entry?.error) {
    $("#translation").textContent = "";
    $("#translation-status").textContent = `· erro: ${entry.error}`;
  } else {
    $("#translation").textContent = entry?.text || "";
    $("#translation-status").textContent = entry?.done ? "· OPUS-MT, local" : `· traduzindo… ${state.workerStatus}`;
  }
}

function renderCard() {
  const total = state.reviews.length;
  const done = Object.keys(state.labels).length;
  $("#done-count").textContent = done;
  $("#total").textContent = total;
  $("#position").textContent = Math.min(state.index + 1, total);
  $("#progress-bar").style.width = `${(done / total) * 100}%`;
  $("#goal-mark").style.left = `${Math.min(100, (GOAL / total) * 100)}%`;
  const card = $("#card");
  card.className = "review-card";
  card.style.transform = "";
  setStamp(0);
  const review = state.reviews[state.index];
  $("#finished").hidden = Boolean(review);
  card.querySelector(".card-scroll").hidden = !review;
  card.querySelector(".card-meta").hidden = !review;
  if (!review) return;
  const current = state.labels[review.id];
  $("#card-id").textContent = `review ${review.id.slice(0, 8)}`;
  $("#card-current-label").textContent = current ? `já anotada: ${LABEL_NAMES[current.label]}` : "";
  $("#original").textContent = review.text;
  $("#doubt").checked = Boolean(current?.uncertain);
  document.querySelectorAll(".choice").forEach((button) => button.classList.toggle("selected", button.dataset.label === current?.label));
  card.querySelector(".card-scroll").scrollTop = 0;
  renderTranslation();
  requestTranslations();
}

function annotate(label, direction) {
  const review = state.reviews[state.index];
  if (!review) return;
  const entry = { label, uncertain: $("#doubt").checked, at: new Date().toISOString() };
  state.labels[review.id] = entry;
  state.pending.push({ review_id: review.id, label, uncertain: entry.uncertain, annotator: state.annotator, annotated_at: entry.at });
  storage.set(`${STORE_PREFIX}pending`, state.pending);
  sync();
  $("#doubt").checked = false;
  const card = $("#card");
  card.classList.add("settling", direction === "right" ? "leaving-right" : direction === "left" ? "leaving-left" : "leaving-up");
  state.index = Math.min(state.index + 1, state.reviews.length);
  save();
  setTimeout(renderCard, 260);
}

function undo() {
  if (state.index === 0) return;
  state.index -= 1;
  save();
  renderCard();
}

function setStamp(dx) {
  const strength = Math.min(1, Math.abs(dx) / SWIPE_THRESHOLD);
  $(".stamp-positive").style.opacity = dx > 0 ? strength : 0;
  $(".stamp-negative").style.opacity = dx < 0 ? strength : 0;
}

function enableSwipe() {
  const card = $("#card");
  let start = null;
  card.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    start = { x: event.clientX, y: event.clientY, id: event.pointerId, horizontal: null };
  });
  card.addEventListener("pointermove", (event) => {
    if (!start || event.pointerId !== start.id) return;
    const dx = event.clientX - start.x;
    const dy = event.clientY - start.y;
    if (start.horizontal === null && Math.hypot(dx, dy) > 8) {
      // Gesto vertical continua rolando o texto; só o horizontal vira swipe.
      start.horizontal = Math.abs(dx) > Math.abs(dy);
      if (start.horizontal) { card.setPointerCapture(event.pointerId); card.classList.add("dragging"); }
    }
    if (!start.horizontal) return;
    card.style.transform = `translateX(${dx}px) rotate(${dx / 22}deg)`;
    setStamp(dx);
  });
  const end = (event) => {
    if (!start || event.pointerId !== start.id) return;
    const dx = event.clientX - start.x;
    const wasHorizontal = start.horizontal;
    start = null;
    card.classList.remove("dragging");
    if (!wasHorizontal) return;
    if (dx > SWIPE_THRESHOLD) annotate("positive", "right");
    else if (dx < -SWIPE_THRESHOLD) annotate("negative", "left");
    else { card.classList.add("settling"); card.style.transform = ""; setStamp(0); setTimeout(() => card.classList.remove("settling"), 320); }
  };
  card.addEventListener("pointerup", end);
  card.addEventListener("pointercancel", end);
}

function exportCsv() {
  const rows = [["review_id", "label", "uncertain", "annotator", "annotated_at"]];
  for (const review of state.reviews) {
    const entry = state.labels[review.id];
    if (entry) rows.push([review.id, entry.label, entry.uncertain ? 1 : 0, state.annotator, entry.at]);
  }
  const blob = new Blob([rows.map((row) => row.map(csvField).join(",")).join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
  const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `anotacoes_${state.annotator}.csv` });
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

async function begin() {
  const name = $("#annotator").value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "_");
  if (!name) { $("#annotator").focus(); $("#load-status").textContent = "Informe seu nome para separar os arquivos dos anotadores."; return; }
  if (!state.reviews.length) { $("#load-status").textContent = "Nenhuma resenha carregada."; return; }
  state.annotator = name;
  storage.set(`${STORE_PREFIX}last`, name);
  const saved = storage.get(storeKey(), { labels: {}, index: 0 });
  state.labels = { ...(saved.labels || {}) };
  const hasLocalProgress = Object.keys(state.labels).length > 0;
  $("#start-button").disabled = true;
  $("#load-status").textContent = SHEETS_URL ? "Buscando suas anotações na planilha…" : "";
  await pullRemote();
  $("#start-button").disabled = false;
  // Em outro aparelho, o índice local não existe: retoma na primeira pendente.
  const firstOpen = state.reviews.findIndex((item) => !state.labels[item.id]);
  state.index = hasLocalProgress && saved.index != null
    ? Math.min(saved.index, state.reviews.length)
    : firstOpen < 0 ? state.reviews.length : firstOpen;
  $("#annotator-name").textContent = name;
  $("#start").hidden = true;
  $("#workspace").hidden = false;
  renderCard();
  sync();
  $("#card").focus();
}

function bindControls() {
  $("#start-button").addEventListener("click", begin);
  $("#annotator").addEventListener("keydown", (event) => { if (event.key === "Enter") begin(); });
  $("#csv-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try { state.reviews = parseCsv(await file.text()); $("#load-status").textContent = `${state.reviews.length} resenhas carregadas de ${file.name}.`; }
    catch (error) { $("#load-status").textContent = error.message; }
  });
  document.querySelectorAll(".choice").forEach((button) => button.addEventListener("click", () => {
    const label = button.dataset.label;
    annotate(label, label === "positive" ? "right" : label === "negative" ? "left" : "up");
  }));
  $("#undo").addEventListener("click", undo);
  $("#export").addEventListener("click", exportCsv);
  $("#switch").addEventListener("click", () => { save(); $("#workspace").hidden = true; $("#start").hidden = false; $("#annotator").value = ""; $("#annotator").focus(); });
  $("#show-translation").addEventListener("change", () => { storage.set(`${STORE_PREFIX}show-translation`, $("#show-translation").checked); renderTranslation(); requestTranslations(); });
  $("#model-choice").addEventListener("change", (event) => {
    state.variant = event.target.value;
    storage.set(`${STORE_PREFIX}variant`, state.variant);
    state.worker?.postMessage({ type: "clear" });
    loadTranslationCache();
    renderTranslation();
    requestTranslations();
  });
  document.addEventListener("keydown", (event) => {
    if ($("#workspace").hidden || event.target.closest?.("input[type=text], select")) return;
    if (event.target.closest?.("button") && (event.key === "Enter" || event.key === " ")) return;
    const keys = { ArrowRight: ["positive", "right"], ArrowLeft: ["negative", "left"], ArrowUp: ["neutral", "up"], ArrowDown: ["mixed", "up"], u: ["unclear", "up"], U: ["unclear", "up"] };
    if (keys[event.key]) { event.preventDefault(); annotate(...keys[event.key]); }
    else if (event.key === "z" || event.key === "Z" || event.key === "Backspace") { event.preventDefault(); undo(); }
    else if (event.key === "d" || event.key === "D") $("#doubt").checked = !$("#doubt").checked;
    else if (event.key === "t" || event.key === "T") { $("#show-translation").checked = !$("#show-translation").checked; $("#show-translation").dispatchEvent(new Event("change")); }
  });
}

async function init() {
  bindControls();
  enableSwipe();
  state.variant = storage.get(`${STORE_PREFIX}variant`, "big");
  $("#model-choice").value = state.variant;
  $("#show-translation").checked = storage.get(`${STORE_PREFIX}show-translation`, true);
  $("#annotator").value = storage.get(`${STORE_PREFIX}last`, "");
  state.pending = storage.get(`${STORE_PREFIX}pending`, []);
  window.addEventListener("online", sync);
  setInterval(sync, 30000);
  loadTranslationCache();
  $("#start-button").disabled = true;
  try {
    const response = await fetch(CSV_PATH);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.reviews = parseCsv(await response.text());
    $("#load-status").textContent = `${state.reviews.length} resenhas carregadas. Meta mínima: ${GOAL}, na ordem em que aparecem.`;
  } catch {
    $("#load-status").textContent = "Não foi possível carregar a amostra automaticamente. Use “Carregar outro CSV” e escolha amostra_anotacao.csv.";
  }
  $("#start-button").disabled = false;
}

init();
