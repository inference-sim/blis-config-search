// Workloads tab: BLIS workload presets from window.CATALOG.workloads.
import { esc, catalog, fmt, numCell, lc, reportsFor, reportsHref, sortRows, sortableHead } from "./app.js";

const state = { col: "name", dir: "asc", q: "" };
const rng = (a, b) => (a == null && b == null ? null : (a == null ? "—" : fmt(a)) + " – " + (b == null ? "—" : fmt(b)));
const COLS = [
  { id: "name", label: "Preset", get: w => w.name },
  { id: "isl_mean", label: "ISL mean", get: w => w.isl_mean, num: true },
  { id: "isl_stdev", label: "ISL σ", get: w => w.isl_stdev, num: true },
  { id: "isl_min", label: "ISL min", get: w => w.isl_min, num: true },
  { id: "isl_max", label: "ISL max", get: w => w.isl_max, num: true },
  { id: "osl_mean", label: "OSL mean", get: w => w.osl_mean, num: true },
  { id: "osl_stdev", label: "OSL σ", get: w => w.osl_stdev, num: true },
  { id: "osl_min", label: "OSL min", get: w => w.osl_min, num: true },
  { id: "osl_max", label: "OSL max", get: w => w.osl_max, num: true },
  { id: "prefix", label: "Prefix tokens", get: w => w.prefix_tokens, num: true },
  { id: "reports", label: "Reports", get: w => reportsFor("workload", w.name).length, num: true },
];

export function mount(root, route) {
  const W = (catalog().workloads || []);
  if (route.params && route.params.q) state.q = route.params.q;
  root.innerHTML = `
    <h2 class="vt">Workloads</h2>
    <p class="dek">Workload presets from inference-sim: input (ISL) and output (OSL) sequence-length distributions in tokens. "—" means the preset does not define the value.</p>
    <div class="toolbar"><span id="st" class="mono"></span><span class="tright"><input class="search" id="q" type="search" placeholder="Search presets" aria-label="Search workloads" value="${esc(state.q)}"></span></div>
    <div class="tablewrap"><table class="cat"><thead id="h"></thead><tbody id="b"></tbody></table></div>`;
  const $ = id => root.querySelector("#" + id);
  function render() {
    const q = lc(state.q).trim();
    const rows = W.filter(w => !q || lc(w.name).includes(q));
    sortableHead($("h"), COLS, state, render);
    const sorted = sortRows(rows, COLS, state.col, state.dir);
    $("b").innerHTML = sorted.length ? sorted.map(w => {
      const n = reportsFor("workload", w.name).length;
      return `<tr class="row"><td class="first"><span class="mname">${esc(w.name)}</span></td>
        ${numCell(w.isl_mean, 0, "ISL mean")}${numCell(w.isl_stdev, 0, "ISL σ")}${numCell(w.isl_min, 0, "ISL min")}${numCell(w.isl_max, 0, "ISL max")}
        ${numCell(w.osl_mean, 0, "OSL mean")}${numCell(w.osl_stdev, 0, "OSL σ")}${numCell(w.osl_min, 0, "OSL min")}${numCell(w.osl_max, 0, "OSL max")}
        ${numCell(w.prefix_tokens, 0, "Prefix tokens")}
        <td class="num${n ? "" : " dash"}" data-label="Reports">${n ? `<a class="xl" href="${reportsHref({ workload: w.name })}">${fmt(n)} →</a>` : "—"}</td></tr>`;
    }).join("") : `<tr class="emptyrow"><td colspan="${COLS.length}">${W.length ? "No presets match." : "Catalog unavailable — built without inference-sim."}</td></tr>`;
    $("st").textContent = `${rows.length} of ${W.length} presets`;
  }
  $("q").oninput = e => { state.q = e.target.value; render(); };
  render();
}
void rng;
