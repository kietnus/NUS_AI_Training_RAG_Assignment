"use strict";

// ---------------------------------------------------------------- config & state
const DEFAULT_TENANTS = ["company-a", "company-b"];
const ADD_TENANT = "__add__";

const state = {
  tenant: DEFAULT_TENANTS[0],
  documents: [],
  selected: new Set(),
  dialogDocId: null,
};

const $ = (id) => document.getElementById(id);

function storageGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

// Small DOM builder: el("div", {class: "x", onclick: fn}, "text", childNode)
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function toast(message, kind = "bad") {
  const t = $("toast");
  t.textContent = message;
  t.className = `toast ${kind === "ok" ? "ok" : ""}`;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, kind === "ok" ? 3500 : 7000);
}

// ---------------------------------------------------------------- API
async function api(path, { method = "GET", json, form } = {}) {
  const headers = { "X-Tenant-Id": state.tenant };
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  if (form !== undefined) body = form;
  let res;
  try {
    res = await fetch(path, { method, headers, body });
  } catch (err) {
    throw new Error(`Cannot reach the backend (${err.message}).`);
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const detail = data && typeof data === "object" ? data.detail : data;
    throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}: ${JSON.stringify(detail)}`);
  }
  return data;
}

// ---------------------------------------------------------------- tenant
function renderTenants() {
  const custom = storageGet("nusaskify.customTenants", []);
  const tenants = [...new Set([...DEFAULT_TENANTS, ...custom, state.tenant])];
  const select = $("tenant-select");
  select.replaceChildren(
    ...tenants.map((t) => el("option", { value: t, selected: t === state.tenant }, t)),
    el("option", { value: ADD_TENANT }, "+ Add tenant…"),
  );
}

function setTenant(tenant) {
  state.tenant = tenant;
  state.selected.clear();
  storageSet("nusaskify.tenant", tenant);
  renderTenants();
  clearResults();
  loadDocuments();
}

$("tenant-select").addEventListener("change", (e) => {
  if (e.target.value !== ADD_TENANT) { setTenant(e.target.value); return; }
  const name = (prompt("New tenant id (letters, digits, - _ .):") || "").trim();
  if (!name) { renderTenants(); return; }
  const custom = storageGet("nusaskify.customTenants", []);
  if (!custom.includes(name)) storageSet("nusaskify.customTenants", [...custom, name]);
  setTenant(name);
});

// ---------------------------------------------------------------- health
async function loadHealth() {
  const node = $("health");
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    const problems = [];
    if (h.database !== "ok") problems.push("database");
    if (h.ollama !== "ok") problems.push("Ollama");
    if (h.embedding_model_available === false) problems.push(`model ${h.embedding_model} not pulled`);
    if (h.chat_model_available === false) problems.push(`model ${h.chat_model} not pulled`);
    node.className = `health small ${problems.length ? "bad" : "ok"}`;
    node.textContent = problems.length ? `Problem: ${problems.join(", ")}` : `${h.embedding_model} · ${h.chat_model}`;
    node.title = JSON.stringify(h, null, 2);
  } catch {
    node.className = "health small bad";
    node.textContent = "backend unreachable";
  }
}

// ---------------------------------------------------------------- documents
const STATUS_LABEL = {
  current: "Current",
  superseded: "Superseded",
  expired: "Expired",
  not_yet_valid: "Not yet valid",
};

async function loadDocuments() {
  const list = $("doc-list");
  try {
    state.documents = await api("/api/documents");
  } catch (err) {
    state.documents = [];
    list.replaceChildren(el("p", { class: "status bad" }, err.message));
    updateScopeInfo();
    return;
  }
  const ids = new Set(state.documents.map((d) => d.id));
  for (const id of [...state.selected]) if (!ids.has(id)) state.selected.delete(id);
  renderDocuments();
}

function renderDocuments() {
  const list = $("doc-list");
  $("doc-count").textContent = state.documents.length ? `(${state.documents.length})` : "";
  if (!state.documents.length) {
    list.replaceChildren(el("p", { class: "muted" }, `No documents for tenant “${state.tenant}” yet. Upload one above.`));
    updateScopeInfo();
    return;
  }
  list.replaceChildren(...state.documents.map((d) => {
    const checkbox = el("input", {
      type: "checkbox",
      checked: state.selected.has(d.id),
      "aria-label": `Select ${d.title} v${d.version}`,
      onchange: (e) => {
        if (e.target.checked) state.selected.add(d.id); else state.selected.delete(d.id);
        updateScopeInfo();
      },
    });
    return el("div", { class: `doc ${d.effective_status === "current" ? "" : "inactive"}` },
      checkbox,
      el("div", {},
        el("div", { class: "doc-title" }, d.title, " ",
          el("span", { class: "muted small" }, `v${d.version}`), " ",
          el("span", { class: `badge ${d.effective_status}` }, STATUS_LABEL[d.effective_status] || d.effective_status)),
        el("div", { class: "doc-meta" },
          el("span", {}, `${d.valid_from} → ${d.valid_to}`),
          el("span", {}, `${d.file_type.toUpperCase()} · ${d.file_name}`),
          el("span", {}, `${d.semantic_chunks} semantic / ${d.structure_chunks} structure chunks`)),
      ),
      el("div", { class: "doc-actions" },
        el("button", { type: "button", class: "btn ghost small", onclick: () => openChunks(d.id) }, "Chunks"),
        el("button", { type: "button", class: "btn ghost small danger", onclick: () => deleteDocument(d) }, "Delete")),
    );
  }));
  updateScopeInfo();
}

function updateScopeInfo() {
  const n = state.selected.size;
  const picked = state.documents.filter((d) => state.selected.has(d.id));
  const inactive = picked.filter((d) => d.effective_status !== "current").length;
  let text = n ? `Scope: ${n} document${n > 1 ? "s" : ""} selected.` : "Select at least one document in the list.";
  if (inactive) text += ` ${inactive} of them ${inactive > 1 ? "are" : "is"} not current and will be filtered out by retrieval.`;
  $("scope-info").textContent = text;
}

$("select-current").addEventListener("click", () => {
  state.selected = new Set(state.documents.filter((d) => d.effective_status === "current").map((d) => d.id));
  renderDocuments();
});
$("select-none").addEventListener("click", () => { state.selected.clear(); renderDocuments(); });
$("refresh-docs").addEventListener("click", loadDocuments);

async function deleteDocument(doc) {
  if (!confirm(`Delete “${doc.title}” v${doc.version} and all its chunks?`)) return;
  try {
    const r = await api(`/api/documents/${doc.id}`, { method: "DELETE" });
    toast(`Deleted ${r.title} v${r.version} (${r.chunks_deleted} chunks).`, "ok");
    state.selected.delete(doc.id);
    loadDocuments();
  } catch (err) {
    toast(err.message);
  }
}

// ---------------------------------------------------------------- upload
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const status = $("upload-status");
  const fd = new FormData(form);
  const file = fd.get("file");
  // Client-side checks mirror the server; the server is still the source of truth.
  const missing = ["title", "version", "valid_from", "valid_to"].filter((k) => !String(fd.get(k) || "").trim());
  if (!file || !file.name) missing.unshift("file");
  if (missing.length) {
    status.className = "status bad";
    status.textContent = `Missing required: ${missing.join(", ")}.`;
    return;
  }
  const btn = $("upload-btn");
  btn.disabled = true;
  status.className = "status";
  status.textContent = "Parsing, chunking and embedding… (can take a little while)";
  try {
    const r = await api("/api/documents", { method: "POST", form: fd });
    let msg = `Indexed “${r.document.title}” v${r.document.version}: ${r.chunks_created.semantic} semantic + ` +
      `${r.chunks_created.structure} structure chunks in ${(r.ingestion_ms / 1000).toFixed(1)} s.`;
    if (r.expired_documents.length) msg += ` Marked expired: v${r.expired_documents.map((x) => x.version).join(", v")}.`;
    if (r.stored_as_expired_because_newer_version) {
      msg += ` Stored as expired: v${r.stored_as_expired_because_newer_version.version} is newer.`;
    }
    status.className = "status ok";
    status.textContent = msg;
    form.reset();
    if (r.document.effective_status === "current") state.selected.add(r.document.id);
    loadDocuments();
  } catch (err) {
    status.className = "status bad";
    status.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------- ask / search
function clearResults() {
  $("answer-card").hidden = true;
  $("chunks-card").hidden = true;
}

function requestBody() {
  const question = $("question").value.trim();
  if (!question) { toast("Type a question first."); return null; }
  if (!state.selected.size) { toast("Select at least one document (search scope)."); return null; }
  return {
    question,
    document_ids: [...state.selected],
    top_k: Math.min(20, Math.max(1, parseInt($("top-k").value, 10) || 5)),
    chunking_method: $("chunking-method").value,
  };
}

async function run(endpoint) {
  const body = requestBody();
  if (!body) return;
  const buttons = [$("ask-btn"), $("search-btn")];
  buttons.forEach((b) => { b.disabled = true; });
  const label = $(endpoint === "ask" ? "ask-btn" : "search-btn");
  const original = label.textContent;
  label.textContent = endpoint === "ask" ? "Thinking…" : "Searching…";
  try {
    const r = await api(`/api/${endpoint}`, { method: "POST", json: body });
    if (endpoint === "ask") {
      renderAnswer(r);
      renderChunks(r.chunks, new Set(r.citations.map((c) => c.marker)), body, r);
    } else {
      $("answer-card").hidden = true;
      renderChunks(r.results, new Set(), body, null);
    }
  } catch (err) {
    toast(err.message);
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
    label.textContent = original;
  }
}

$("ask-form").addEventListener("submit", (e) => { e.preventDefault(); run("ask"); });
$("search-btn").addEventListener("click", () => run("search"));
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run("ask"); }
});

function citeLink(n) {
  return el("a", {
    class: "cite", href: `#chunk-${n}`, title: `Source [${n}]`,
    onclick: (e) => {
      e.preventDefault();
      const target = $(`chunk-${n}`);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.remove("flash"); void target.offsetWidth; target.classList.add("flash");
    },
  }, n);
}

// Render answer text, turning [n] markers into clickable links (no innerHTML).
function answerNodes(text) {
  const nodes = [];
  let last = 0;
  for (const m of text.matchAll(/\[(\d+)\]/g)) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    nodes.push(citeLink(Number(m[1])));
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

function renderAnswer(r) {
  $("answer-card").hidden = false;
  const badge = $("answer-status");
  badge.className = `badge ${r.status}`;
  badge.textContent = r.status === "answered" ? "Answered" : "Not enough information";
  $("answer-text").replaceChildren(...answerNodes(r.answer));
  const reasonParts = [];
  if (r.reason) reasonParts.push(r.reason);
  if (r.top_score !== null && r.top_score !== undefined) reasonParts.push(`Best score ${r.top_score.toFixed(3)} (MIN_SCORE ${r.min_score}).`);
  if (r.dropped_citations && r.dropped_citations.length) reasonParts.push(`Removed invalid citations: ${r.dropped_citations.map((n) => `[${n}]`).join(" ")}.`);
  $("answer-reason").textContent = reasonParts.join(" ");

  $("citations").replaceChildren(...r.citations.map((c) => el("div", { class: "citation" },
    citeLink(c.marker), " ",
    el("strong", {}, c.document_title), ` v${c.version} · chunk #${c.chunk_index} · score ${c.score.toFixed(3)}`,
    c.quote ? el("div", {}, el("q", {}, c.quote), " ",
      c.quote_verified === true ? el("span", { class: "verified", title: "Quote found in the cited chunk" }, "✓ quote found in chunk")
        : el("span", { class: "unverified", title: "Quote not found verbatim in the cited chunk" }, "⚠ quote not found in chunk"))
      : null,
  )));
}

function renderChunks(chunks, cited, body, askResult) {
  $("chunks-card").hidden = false;
  $("chunks-count").textContent = `(${chunks.length})`;
  $("chunks-meta").textContent = `${body.chunking_method} chunking · top-${body.top_k}`;
  const list = $("chunk-results");
  if (!chunks.length) {
    list.replaceChildren(el("li", { class: "muted" },
      "No chunks in scope. The selected documents may be expired, not yet valid, or belong to another tenant."));
    return;
  }
  const lowScore = askResult ? askResult.min_score : null;
  list.replaceChildren(...chunks.map((c) => el("li", {
    id: `chunk-${c.rank}`, class: `chunk ${cited.has(c.rank) ? "cited" : ""}`,
  },
    el("div", { class: "chunk-head" },
      el("span", { class: "rank" }, `[${c.rank}]`),
      el("strong", {}, c.document_title),
      el("span", { class: "muted" }, `v${c.version}`),
      el("span", { class: "muted" }, `chunk #${c.chunk_index}`),
      el("span", { class: "score", title: "Cosine similarity" },
        el("span", { class: "bar" }, el("span", { style: `width:${Math.max(0, Math.min(1, c.score)) * 100}%` })),
        c.score.toFixed(3),
        lowScore !== null && c.score < lowScore ? el("span", { class: "badge not_yet_valid" }, "below MIN_SCORE") : null),
      cited.has(c.rank) ? el("span", { class: "badge answered" }, "cited") : null,
    ),
    el("p", { class: "chunk-content" }, c.content),
  )));
}

// ---------------------------------------------------------------- chunk viewer (debug)
async function openChunks(docId) {
  state.dialogDocId = docId;
  $("chunks-dialog").showModal();
  await renderDialog();
}

async function renderDialog() {
  const body = $("dialog-body");
  const method = $("dialog-method").value;
  body.replaceChildren(el("p", { class: "muted" }, "Loading…"));
  try {
    const r = await api(`/api/documents/${state.dialogDocId}/chunks?method=${method}`);
    const d = r.document;
    $("dialog-title").textContent = `${d.title} v${d.version}`;
    $("dialog-subtitle").textContent = `${d.valid_from} → ${d.valid_to} · ${STATUS_LABEL[d.effective_status]} · ${d.file_name}`;
    const s = d.ingestion_stats || {};
    let stats = "";
    if (method === "semantic" && s.semantic) {
      const x = s.semantic;
      stats = `${x.units} pieces → ${x.chunks} chunks · threshold ${x.threshold} (${x.threshold_source}) · ` +
        `buffer ${x.buffer_size} · breaks: ${x.breaks_low_similarity} low-similarity, ${x.breaks_max_length} max-length · ` +
        `adjacent similarity mean ${x.similarity_mean} (min ${x.similarity_min}, max ${x.similarity_max})`;
    } else if (s.structure) {
      stats = `${s.structure.chunks} chunks · max ${s.structure.max_chars} chars per chunk`;
    }
    body.replaceChildren(
      el("p", { class: "stats" }, stats),
      ...r.chunks.map((c) => el("div", { class: "chunk" },
        el("div", { class: "chunk-head" },
          el("span", { class: "rank" }, `#${c.chunk_index}`),
          el("span", { class: "muted" }, `${c.char_count} chars · ${c.embedding_dim}-dim vector`)),
        el("p", { class: "chunk-content" }, c.content),
        el("div", { class: "meta-line" }, JSON.stringify(c.metadata)),
      )),
    );
  } catch (err) {
    body.replaceChildren(el("p", { class: "status bad" }, err.message));
  }
}

$("dialog-method").addEventListener("change", renderDialog);
$("dialog-close").addEventListener("click", () => $("chunks-dialog").close());

// ---------------------------------------------------------------- boot
state.tenant = storageGet("nusaskify.tenant", DEFAULT_TENANTS[0]);
renderTenants();
loadHealth();
loadDocuments();
