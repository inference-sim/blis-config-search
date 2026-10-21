// GPUs tab: hardware BLIS knows, from window.CATALOG.hardware (spec-sheet values only; absent = "—").
import { esc, catalog, fmt, numCell, strCell, lc, reportsFor, reportsHref, sortRows, sortableHead } from "./app.js";

const state = { col: "name", dir: "asc", q: "" };
const COLS = [
  { id: "name", label: "GPU", get: h => h.name },
  { id: "mem", label: "Memory (GiB)", get: h => h.memory_gib, num: true },
  { id: "bw", label: "Mem BW (TB/s)", get: h => h.bw_tbs, num: true },
  { id: "bf16", label: "BF16 (TFLOPs)", get: h => h.tflops_bf16, num: true },
  { id: "fp8", label: "FP8 (TFLOPs)", get: h => h.tflops_fp8, num: true },
  { id: "fp4", label: "NVFP4 (TFLOPs)", get: h => h.tflops_nvfp4, num: true },
  { id: "sm", label: "SMs", get: h => h.sm_count, num: true },
  { id: "nv", label: "Intra-node BW (GB/s)", get: h => h.intra_node_bw_gbps, num: true },
  { id: "gpn", label: "GPUs/node", get: h => h.gpus_per_node, num: true },
  { id: "gpr", label: "GPUs/rack", get: h => h.gpus_per_rack, num: true },
  { id: "prov", label: "Provenance", get: h => h.provenance },
  { id: "reports", label: "Reports", get: h => reportsFor("hardware", h.name).length, num: true },
];

export function mount(root, route) {
  const H = (catalog().hardware || []);
  if (route.params && route.params.q) state.q = route.params.q;
  root.innerHTML = `
    <h2 class="vt">GPUs</h2>
    <p class="dek">Hardware the simulator models, from the inference-sim hardware config. "—" means the config carries no value for that field.</p>
    <div class="toolbar"><span id="st" class="mono"></span><span class="tright"><input class="search" id="q" type="search" placeholder="Search GPUs" aria-label="Search GPUs" value="${esc(state.q)}"></span></div>
    <div class="tablewrap"><table class="cat"><thead id="h"></thead><tbody id="b"></tbody></table></div>`;
  const $ = id => root.querySelector("#" + id);
  function render() {
    const q = lc(state.q).trim();
    const rows = H.filter(h => !q || lc(h.name).includes(q));
    sortableHead($("h"), COLS, state, render);
    const sorted = sortRows(rows, COLS, state.col, state.dir);
    $("b").innerHTML = sorted.length ? sorted.map(h => {
      const n = reportsFor("hardware", h.name).length;
      return `<tr class="row"><td class="first"><span class="mname">${esc(h.name)}</span></td>
        ${numCell(h.memory_gib, 1, "Memory (GiB)")}${numCell(h.bw_tbs, 2, "Mem BW (TB/s)")}
        ${numCell(h.tflops_bf16, 0, "BF16 (TFLOPs)")}${numCell(h.tflops_fp8, 0, "FP8 (TFLOPs)")}${numCell(h.tflops_nvfp4, 0, "NVFP4 (TFLOPs)")}
        ${numCell(h.sm_count, 0, "SMs")}${numCell(h.intra_node_bw_gbps, 0, "Intra-node BW (GB/s)")}
        ${numCell(h.gpus_per_node, 0, "GPUs/node")}${numCell(h.gpus_per_rack, 0, "GPUs/rack")}
        ${strCell(h.provenance, "Provenance")}
        <td class="num${n ? "" : " dash"}" data-label="Reports">${n ? `<a class="xl" href="${reportsHref({ hardware: h.name })}">${fmt(n)} →</a>` : "—"}</td></tr>`;
    }).join("") : `<tr class="emptyrow"><td colspan="${COLS.length}">${H.length ? "No GPUs match." : "Catalog unavailable — built without inference-sim."}</td></tr>`;
    $("st").textContent = `${rows.length} of ${H.length} GPUs`;
  }
  $("q").oninput = e => { state.q = e.target.value; render(); };
  render();
}
