// config-search SPA entry: hash router, tab nav, theme toggle, shared helpers.
// Vanilla ES modules; data arrives via window.SCENARIOS / window.CATALOG / window.MODE (classic <script>s).
import * as reports from "./reports.js";
import * as detail from "./detail.js";
import * as models from "./models.js";
import * as workloads from "./workloads.js";
import * as gpus from "./gpus.js";
import * as submit from "./submit.js";

/* ---------------- shared helpers (function declarations: safe across the import cycle) ---------------- */
export function esc(v) {
  return String(v == null ? "" : v).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
export const scenarios = () => (Array.isArray(window.SCENARIOS) ? window.SCENARIOS : []);
export const catalog = () => window.CATALOG || { models: [], hardware: [], workloads: [], fabrics: [], storage: [], registry: [], versions: {} };
export const mode = () => (window.MODE === "local" ? "local" : "pages");

export const isNum = v => typeof v === "number" && isFinite(v);
// number formatters; absent/non-finite renders as an em dash, never invented
export function fmt(v, d = 0) {
  return isNum(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—";
}
export const f0 = v => fmt(v, 0);
export const f1 = v => fmt(v, 1);
export function dash(v) { return v == null || v === "" || (typeof v === "number" && !isFinite(v)) ? "—" : String(v); }

// <td> with number styling; null -> muted dash
export function numCell(v, d, label, unit) {
  const ok = isNum(v);
  return `<td class="num${ok ? "" : " dash"}" data-label="${esc(label)}">${ok ? fmt(v, d) : "—"}${ok && unit ? `<span class="u">${esc(unit)}</span>` : ""}</td>`;
}
export function strCell(v, label, cls) {
  const ok = !(v == null || v === "");
  return `<td${cls || ok ? ` class="${ok ? (cls || "") : "dash"}"` : ""} data-label="${esc(label)}">${ok ? esc(v) : "—"}</td>`;
}
export const lc = v => String(v == null ? "" : v).toLowerCase();
export const hrefFor = id => encodeURIComponent(id) + "/index.html";
export const scenarioHref = id => "#/scenario/" + encodeURIComponent(id);
export function reportsHref(params) {
  const q = Object.keys(params || {}).filter(k => params[k] != null && params[k] !== "")
    .map(k => encodeURIComponent(k) + "=" + encodeURIComponent(params[k])).join("&");
  return "#/reports" + (q ? "?" + q : "");
}
// Reports whose facet value matches (case-insensitive: catalog says "h100", scenarios say "H100").
export function reportsFor(key, name) {
  return scenarios().filter(s => lc(s[key]) === lc(name));
}

// ---- generic stable sort over column defs {id,get,num,nosort} ----
export function sortRows(rows, cols, colId, dir) {
  const c = cols.find(x => x.id === colId);
  if (!c || c.nosort) return rows.slice();
  const m = dir === "asc" ? 1 : -1;
  return rows.map((r, i) => [r, i]).sort((a, b) => {
    const x = c.get(a[0]), y = c.get(b[0]);
    const xn = x == null || (typeof x === "number" && !isFinite(x)), yn = y == null || (typeof y === "number" && !isFinite(y));
    if (xn || yn) return xn && yn ? a[1] - b[1] : (xn ? 1 : -1); // nulls always last
    const d = c.num ? (x - y) : String(x).localeCompare(String(y));
    return d ? d * m : a[1] - b[1];
  }).map(p => p[0]);
}
// <thead> html + bind sorting for catalog-style tables
export function sortableHead(thead, cols, state, rerender) {
  thead.innerHTML = "<tr>" + cols.map(c => {
    const active = c.id === state.col;
    const aria = c.nosort ? "" : ` aria-sort="${active ? (state.dir === "asc" ? "ascending" : "descending") : "none"}"`;
    const arrow = c.nosort ? "" : `<span class="arrow">${active ? (state.dir === "asc" ? "▲" : "▼") : "↕"}</span>`;
    return `<th class="${c.num ? "num " : ""}${c.nosort ? "nosort" : "th"}" data-c="${esc(c.id)}"${aria}${c.nosort ? "" : ' tabindex="0" role="columnheader button"'}>${arrow}${esc(c.label)}</th>`;
  }).join("") + "</tr>";
  thead.querySelectorAll("th.th").forEach(th => {
    const act = () => {
      const id = th.dataset.c, c = cols.find(x => x.id === id);
      if (state.col === id) state.dir = state.dir === "asc" ? "desc" : "asc";
      else { state.col = id; state.dir = c.num ? "desc" : "asc"; }
      rerender();
    };
    th.onclick = act;
    th.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } };
  });
}

// ---- facet engine ----
// sel: {key:Set}; passes ignores `exceptKey` so chip counts show "what you'd get if you picked this".
export function makeSel(keys) { const o = {}; keys.forEach(k => (o[k] = new Set())); return o; }
export function passes(item, sel, keys, getv, exceptKey) {
  return keys.every(k => k === exceptKey || sel[k].size === 0 || sel[k].has(getv(item, k)));
}
export function facetCount(items, sel, keys, getv, key, v) {
  return items.filter(it => passes(it, sel, keys, getv, key) && getv(it, key) === v).length;
}
// groups: [{key,label,values:[{v,label,n,on,color}]}]; onToggle(key,v)
export function facetsHTML(groups) {
  return groups.map(g => `<div class="fgroup" role="group" aria-label="${esc(g.label)}"><span class="flabel">${esc(g.label)}</span>` +
    g.values.map(x => {
      const cls = "fchip" + (x.on ? " on" : "") + (x.n === 0 && !x.on ? " empty" : "");
      return `<span class="${cls}" role="button" tabindex="0" aria-pressed="${!!x.on}" data-k="${esc(g.key)}" data-v="${esc(x.v)}">${x.color ? `<span class="dot" style="background:${x.color}"></span>` : ""}${esc(x.label != null ? x.label : x.v)}<span class="n">${x.n}</span></span>`;
    }).join("") + `</div>`).join("");
}
export function bindFacets(el, onToggle) {
  el.querySelectorAll(".fchip").forEach(c => {
    const act = () => { if (!c.classList.contains("empty")) onToggle(c.dataset.k, c.dataset.v); };
    c.onclick = act;
    c.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } };
  });
}

// ---- plot helpers (shared by reports + detail; no d3) ----
// "nice" axis: zero-based, step in {1,2,2.5,5}x10^k, never NaN, never zero-span.
export function niceScale(max, n) {
  n = n || 5;
  if (!(max > 0) || !isFinite(max)) max = 1;
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const count = Math.max(1, Math.ceil(max / step - 1e-9));
  const ticks = []; for (let i = 0; i <= count; i++) ticks.push(Math.round(i * step * 1e9) / 1e9);
  return { max: ticks[ticks.length - 1], step, ticks };
}
// data -> pixel. Degenerate (zero / non-finite) span maps to the range midpoint.
export function linScale(d0, d1, r0, r1) {
  const span = d1 - d0;
  return v => (!isFinite(span) || span === 0 || !isFinite(v)) ? (r0 + r1) / 2 : r0 + (v - d0) / span * (r1 - r0);
}
// marker radius from tok/s/GPU: sqrt-area scaling, always finite, 5..10px
export function sizeOf(v, max) {
  if (!isFinite(v) || !(max > 0) || v <= 0) return 5;
  return 5 + 5 * Math.sqrt(Math.min(1, v / max));
}
export function markerSVG(kind, r, color, mock) {
  const c = color, st = mock
    ? `fill="${c}" fill-opacity=".16" stroke="${c}" stroke-width="1.8" stroke-dasharray="3 2"`
    : `fill="${c}" stroke="var(--panel)" stroke-width="1.5"`;
  const k = r * 1.18;
  if (kind === "circle") return `<circle r="${r}" ${st}/>`;
  if (kind === "diamond") return `<path d="M0 ${-k * 1.2}L${k * 1.2} 0L0 ${k * 1.2}L${-k * 1.2} 0Z" ${st}/>`;
  if (kind === "square") return `<rect x="${-r}" y="${-r}" width="${2 * r}" height="${2 * r}" rx="1.5" ${st}/>`;
  return `<path d="M0 ${-k * 1.15}L${k * 1.1} ${k * .85}L${-k * 1.1} ${k * .85}Z" ${st}/>`;
}
export const PALETTE = ["var(--plot-s1)", "var(--plot-s2)", "var(--plot-s3)", "var(--plot-s4)"];

/* ---------------- theme ---------------- */
function initTheme() {
  const root = document.documentElement, btn = document.getElementById("theme");
  const modes = ["system", "light", "dark"];
  let i = 0;
  try { const sv = localStorage.getItem("cs-theme"); if (modes.includes(sv)) i = modes.indexOf(sv); } catch (e) { /* ignore */ }
  const apply = () => {
    const m = modes[i];
    if (m === "system") delete root.dataset.theme; else root.dataset.theme = m;
    btn.textContent = "theme: " + m;
    try { localStorage.setItem("cs-theme", m); } catch (e) { /* ignore */ }
  };
  btn.onclick = () => { i = (i + 1) % 3; apply(); };
  apply();
}

/* ---------------- router ---------------- */
export function parseHash(h) {
  h = (h || "").replace(/^#/, "");
  const qi = h.indexOf("?");
  const path = (qi < 0 ? h : h.slice(0, qi)).replace(/^\/+|\/+$/g, "");
  const params = {};
  if (qi >= 0) new URLSearchParams(h.slice(qi + 1)).forEach((v, k) => { params[k] = v; });
  const parts = path.split("/").filter(Boolean).map(p => { try { return decodeURIComponent(p); } catch (e) { return p; } });
  return { name: parts[0] || "reports", arg: parts[1], params };
}
const VIEWS = { reports, models, workloads, gpus, submit, scenario: detail };
let cleanup = null;

function route() {
  const r = parseHash(location.hash);
  const view = VIEWS[r.name] || reports;
  const tab = VIEWS[r.name] ? (r.name === "scenario" ? "reports" : r.name) : "reports";
  document.querySelectorAll("#tabs a").forEach(a => {
    if (a.dataset.tab === tab) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  const el = document.getElementById("view");
  if (typeof cleanup === "function") { try { cleanup(); } catch (e) { /* ignore */ } }
  cleanup = null;
  el.innerHTML = "";
  try {
    cleanup = view.mount(el, r) || null;
  } catch (err) {
    el.innerHTML = `<div class="card"><h2>Something went wrong</h2><p class="note">${esc(err && err.message)}</p></div>`;
  }
  document.title = "config-search · " + (r.name === "scenario" ? (r.arg || "scenario") : tab);
  if (!route.first) window.scrollTo(0, 0);
  route.first = false;
}
route.first = true;

function renderVersions() {
  const v = (catalog().versions) || {};
  const ks = Object.keys(v);
  document.getElementById("vers").innerHTML = ks.length
    ? `<span>Catalog sources:</span>` + ks.map(k => `<span>${esc(k)} <b>${esc(v[k])}</b></span>`).join("")
    : `<span>Catalog versions unavailable.</span>`;
}

// Update the URL (filters) without re-routing.
export function replaceHash(h) {
  try { history.replaceState(null, "", h); } catch (e) { /* file:// etc. */ }
}

initTheme();
renderVersions();
window.addEventListener("hashchange", route);
route();
