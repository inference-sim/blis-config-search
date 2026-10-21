// Models tab: every model BLIS can run, from window.CATALOG.models.
import { esc, catalog, fmt, isNum, numCell, strCell, lc, reportsFor, reportsHref, sortRows, sortableHead,
  makeSel, passes, facetCount, facetsHTML, bindFacets } from "./app.js";

const state = { col: "name", dir: "asc", q: "" };
const FK = ["arch", "model_type", "quant"];
const FL = { arch: "Architecture", model_type: "Family", quant: "Quantization" };

const COLS = [
  { id: "name", label: "Model", get: m => m.name },
  { id: "arch", label: "Arch", get: m => m.arch },
  { id: "total", label: "Params (B)", get: m => m.params && m.params.total_b, num: true },
  { id: "active", label: "Active (B)", get: m => m.params && m.params.active_b, num: true },
  { id: "layers", label: "Layers", get: m => m.layers, num: true },
  { id: "hidden", label: "Hidden", get: m => m.hidden, num: true },
  { id: "hq", label: "Heads Q", get: m => m.heads_q, num: true },
  { id: "hkv", label: "Heads KV", get: m => m.heads_kv, num: true },
  { id: "experts", label: "Experts", get: m => m.experts, num: true },
  { id: "topk", label: "Top-k", get: m => m.top_k, num: true },
  { id: "context", label: "Context", get: m => m.context, num: true },
  { id: "quant", label: "Quant", get: m => m.quant },
  { id: "attn", label: "Attention", get: m => m.attn_kind },
  { id: "reports", label: "Reports", get: m => reportsFor("model", m.name).length, num: true },
];

export function mount(root, route) {
  const M = (catalog().models || []);
  const gv = (m, k) => m[k];
  const sel = makeSel(FK);
  const p = route.params || {};
  FK.forEach(k => { if (p[k]) sel[k].add(p[k]); });
  if (p.q) state.q = p.q;

  root.innerHTML = `
    <h2 class="vt">Models</h2>
    <p class="dek">Every model the BLIS catalog can simulate. Values come straight from the catalog; "—" means the catalog has no value. Click a model's report count to see the scenarios that used it.</p>
    <section class="facets f3" id="f"></section>
    <div class="toolbar"><span id="st" class="mono"></span><span class="tright"><input class="search" id="q" type="search" placeholder="Search models" aria-label="Search models" value="${esc(state.q)}"><button class="lnk" id="rs" type="button" hidden>clear filters</button></span></div>
    <div class="tablewrap"><table class="cat"><thead id="h"></thead><tbody id="b"></tbody></table></div>`;
  const $ = id => root.querySelector("#" + id);
  if (!M.length) { $("b").innerHTML = `<tr class="emptyrow"><td>Catalog unavailable — built without blis-catalog.</td></tr>`; }

  function render() {
    const q = lc(state.q).trim();
    const rows = M.filter(m => passes(m, sel, FK, gv, null) && (!q || lc(m.name).includes(q) || lc(m.source_repo).includes(q)));
    $("f").innerHTML = facetsHTML(FK.map(k => ({
      key: k, label: FL[k],
      values: [...new Set(M.map(m => m[k]).filter(v => v != null))].sort().map(v => ({ v, n: facetCount(M, sel, FK, gv, k, v), on: sel[k].has(v) })),
    })));
    bindFacets($("f"), (k, v) => { sel[k].has(v) ? sel[k].delete(v) : sel[k].add(v); render(); });
    sortableHead($("h"), COLS, state, render);
    const sorted = sortRows(rows, COLS, state.col, state.dir);
    $("b").innerHTML = sorted.length ? sorted.map(m => {
      const n = reportsFor("model", m.name).length;
      return `<tr class="row">
        <td class="first"><span class="mname">${esc(m.name)}</span></td>
        ${strCell(m.arch, "Arch")}
        ${numCell(m.params && m.params.total_b, 1, "Params (B)")}${numCell(m.params && m.params.active_b, 1, "Active (B)")}
        ${numCell(m.layers, 0, "Layers")}${numCell(m.hidden, 0, "Hidden")}${numCell(m.heads_q, 0, "Heads Q")}${numCell(m.heads_kv, 0, "Heads KV")}
        ${numCell(m.experts, 0, "Experts")}${numCell(m.top_k, 0, "Top-k")}${numCell(m.context, 0, "Context")}
        ${strCell(m.quant, "Quant")}${strCell(m.attn_kind, "Attention")}
        <td class="num${n ? "" : " dash"}" data-label="Reports">${n ? `<a class="xl" href="${reportsHref({ model: m.name })}">${fmt(n)} →</a>` : "—"}</td></tr>`;
    }).join("") : `<tr class="emptyrow"><td colspan="${COLS.length}">No models match.</td></tr>`;
    $("st").textContent = `${rows.length} of ${M.length} models`;
    const any = FK.some(k => sel[k].size) || !!q;
    $("rs").hidden = !any;
  }
  $("q").oninput = e => { state.q = e.target.value; render(); };
  $("rs").onclick = () => { FK.forEach(k => sel[k].clear()); state.q = ""; $("q").value = ""; render(); };
  render();
}
