// Scenario Detail: executive briefing + technical Pareto explorer over site/<run_id>/records.json.
import {
  esc, scenarios, f0, f1, fmt, isNum, hrefFor, reportsHref, makeSel, passes, facetCount, facetsHTML, bindFacets,
  niceScale, linScale, markerSVG, PALETTE,
} from "./app.js";

const DIMS = [
  { key: "series", label: "Config / topology" },
  { key: "conc", label: "Concurrency", fmt: v => "c=" + v },
  { key: "ok", label: "Servable", fmt: v => (v ? "servable" : "not servable") },
];
const KEYS = DIMS.map(d => d.key);
const getv = (r, k) => r[k];

// "50 tok/s/user (ITL p99 ≤ 20 ms)" -> 50 ; absent -> null
function floorValue(s) {
  const m = /([0-9]+(?:\.[0-9]+)?)\s*tok/i.exec(String((s && s.floor) || ""));
  return m ? Number(m[1]) : null;
}

async function loadRecords(runId) {
  try {
    const res = await fetch(encodeURIComponent(runId) + "/records.json", { cache: "no-cache" });
    if (!res.ok) return { rows: [], err: "records.json not found (HTTP " + res.status + ")" };
    const j = await res.json();
    if (!Array.isArray(j)) return { rows: [], err: "records.json is not a list" };
    return { rows: j.filter(r => r && typeof r === "object"), err: null };
  } catch (e) {
    return { rows: [], err: "could not load records.json (" + (e && e.message ? e.message : "error") + ")" };
  }
}

export function mount(root, route) {
  const id = route.arg;
  const s = scenarios().find(x => x.run_id === id);
  if (!s) {
    root.innerHTML = `<p class="crumb"><a href="#/reports">← All reports</a></p><div class="card"><h2>Scenario not found</h2><p class="note">No scenario with run id <span class="mono">${esc(id || "")}</span> in this build.</p></div>`;
    return null;
  }
  const w = s.winner || {};
  const floor = floorValue(s);
  const fin = (s.finalists || []).slice().sort((a, b) => b.tps - a.tps);
  const metric = (v, d, unit, label) => `<div class="metric"><div class="mv">${isNum(v) ? fmt(v, d) : "—"}${unit ? `<small>${esc(unit)}</small>` : ""}</div><div class="ml">${esc(label)}</div></div>`;
  const filt = k => reportsHref({ [k]: s[k] });

  root.innerHTML = `
  <p class="crumb"><a href="#/reports">← All reports</a></p>
  <div class="dhead">
    <div>
      <h2>${esc(s.model)} on ${esc(s.hardware)}${s.synthetic ? ` <span class="mockflag">mock</span>` : ""}</h2>
      <div class="meta"><span>${esc(s.gpu_budget)} GPU budget</span><span>workload <a class="lnk" href="#/workloads">${esc(s.workload)}</a> (${esc(s.isl_osl)})</span><span class="mono">${esc(s.run_id)}</span><span>generated ${esc(s.generated || "—")}</span></div>
    </div>
    <a class="btn primary" href="${hrefFor(s.run_id)}" target="_blank" rel="noopener">Open full report ↗</a>
  </div>

  <section class="brief" aria-label="Briefing">
    <div class="card">
      <p class="eyebrow">Recommended config</p>
      <p class="recocfg">${esc(w.label || "—")}</p>
      <p class="why">${esc(s.reason || "")}</p>
      <dl class="kv">
        <dt>topology</dt><dd>${isNum(w.ni) ? esc(w.ni) + "× " : ""}TP${esc(w.tp != null ? w.tp : "—")}${w.ep ? " + EP" : ""}${w.quant ? " · " + esc(w.quant) : ""}</dd>
        <dt>GPUs used</dt><dd>${esc(w.gpus != null ? w.gpus : "—")} of ${esc(s.gpu_budget)}</dd>
        <dt>operating load</dt><dd>c=${esc(w.c_use != null ? w.c_use : "—")}${isNum(w.c_star) ? ` (C* ${esc(w.c_star)}, headroom ${esc(s.headroom)})` : ""}</dd>
        <dt>servable runs</dt><dd>${esc(s.n_servable)} / ${esc(s.n_runs)}</dd>
      </dl>
    </div>
    <div class="card">
      <p class="eyebrow">Headline metrics at the usable operating point</p>
      <div class="metrics">
        ${metric(w.tps, 0, "tok/s", "Cluster throughput")}
        ${metric(w.tps_per_gpu, 1, "tok/s", "Per GPU")}
        ${metric(w.per_user, 1, "tok/s", "Per user")}
        ${metric(w.itl_p99, 1, "ms", "ITL p99")}
      </div>
      <div class="floorbar">Interactivity floor held: <b>${esc(s.floor || "—")}</b></div>
      <p class="note">Browse more: <a class="lnk" href="${filt("model")}">all ${esc(s.model)} reports</a> · <a class="lnk" href="${filt("hardware")}">all ${esc(s.hardware)} reports</a> · <a class="lnk" href="${filt("workload")}">all ${esc(s.workload)} reports</a></p>
    </div>
  </section>

  <section class="section" aria-label="Finalists">
    <h3>Finalists the recommendation beat</h3>
    <div class="tablewrap"><table class="cat"><thead><tr>
      <th class="nosort">Config</th><th class="num nosort">GPUs</th><th class="num nosort">tok/s</th><th class="num nosort">tok/s/GPU</th><th class="num nosort">tok/s/user</th><th class="num nosort">ITL p99</th><th class="num nosort">Load (c)</th></tr></thead>
      <tbody>${fin.length ? fin.map(f => {
        const isW = f.label === w.label;
        return `<tr class="row ${isW ? "real" : ""}"><td class="first cfgcell" style="${isW ? "" : "color:inherit"}">${esc(f.label)}${isW ? ` <span class="real">recommended</span>` : ""}</td>
          <td class="num" data-label="GPUs">${esc(f.gpus)}</td><td class="num${isW ? " best" : ""}" data-label="tok/s"><span>${f0(f.tps)}</span></td><td class="num" data-label="tok/s/GPU">${f1(f.tps_per_gpu)}</td>
          <td class="num" data-label="tok/s/user">${f1(f.per_user)}</td><td class="num" data-label="ITL p99">${f1(f.itl_p99)}<span class="u">ms</span></td><td class="num" data-label="Load (c)">${esc(f.c_use)}</td></tr>`;
      }).join("") : `<tr class="emptyrow"><td colspan="7">No finalists recorded.</td></tr>`}</tbody></table></div>
  </section>

  <section class="section" aria-label="Run explorer">
    <h3>Run explorer · every simulated run</h3>
    <p class="dek" style="margin-bottom:8px">Filter by config, concurrency or servability; the operating curve and the run table update together. Filled markers are servable runs; hollow markers fail the floor or a limit.</p>
    <div id="xbody"><p class="note">Loading run data…</p></div>
  </section>`;

  let alive = true, onResize = null, rt = null;
  const xbody = root.querySelector("#xbody");

  loadRecords(s.run_id).then(({ rows, err }) => {
    if (!alive) return;
    if (err || !rows.length) {
      xbody.innerHTML = `<div class="card"><p class="note" style="margin:0">${esc(err || "No run records were emitted for this scenario.")} The briefing above is unaffected; the <a class="lnk" href="${hrefFor(s.run_id)}" target="_blank" rel="noopener">full report ↗</a> carries the complete data when available.</p></div>`;
      return;
    }
    explorer(xbody, s, rows, floor, fn => { onResize = fn; });
    window.addEventListener("resize", resizeHandler);
  });
  function resizeHandler() { clearTimeout(rt); rt = setTimeout(() => { if (onResize) onResize(); }, 80); }

  return () => { alive = false; window.removeEventListener("resize", resizeHandler); clearTimeout(rt); };
}

function explorer(host, s, R, floor, setResize) {
  const sel = makeSel(KEYS);
  const series = [...new Set(R.map(r => r.series))].sort((a, b) => String(a).localeCompare(String(b)));
  const colorOf = v => PALETTE[series.indexOf(v)] || "var(--plot-other)";
  const shapeOf = r => ["circle", "diamond", "square", "triangle"][Math.max(0, series.indexOf(r.series)) % 4];
  const sortState = { col: "cl", dir: "desc" };

  host.innerHTML = `
    <section class="facets f3" id="xf" aria-label="Run filters"></section>
    <div class="toolbar"><span id="xstatus" class="mono"></span><button class="lnk" id="xreset" type="button" hidden>clear filters</button></div>
    <section class="plotcard" aria-label="Operating curve">
      <h2>Operating curve</h2>
      <p class="psub">Each marker is one simulated run. Right = more cluster throughput, up = faster per user${floor != null ? `; the dashed line is the ${esc(fmt(floor, 0))} tok/s/user floor` : ""}.</p>
      <div class="plotbox" id="xbox"><svg class="plot" id="xplot" role="img" aria-label="Scatter of runs: cluster tokens per second on X, tokens per second per user on Y"></svg><div class="tip" id="xtip" role="status"></div></div>
      <div class="plegend" id="xleg"></div>
    </section>
    <div class="section"><h3>Runs in view</h3>
      <div class="tablewrap dtbl"><table class="cat"><thead id="xhead"></thead><tbody id="xbodyrows"></tbody></table></div>
      <p class="note" id="xnote"></p></div>`;
  const $ = i => host.querySelector("#" + i);

  const COLS = [
    { id: "series", label: "Config", get: r => r.series },
    { id: "conc", label: "Conc", get: r => r.conc, num: true },
    { id: "gpus", label: "GPUs", get: r => r.gpus, num: true },
    { id: "cl", label: "Cluster tok/s", get: r => r.cl, num: true },
    { id: "pu", label: "Per-user tok/s", get: r => r.pu, num: true },
    { id: "itl", label: "ITL mean (ms)", get: r => r.itl, num: true },
    { id: "ttft", label: "TTFT (ms)", get: r => r.ttft, num: true },
    { id: "ok", label: "Servable", get: r => (r.ok ? 1 : 0), num: true },
    { id: "viol", label: "Limits hit", get: r => (Array.isArray(r.viol) ? r.viol.join(", ") : ""), nosort: true },
  ];
  const cmp = (a, b, col, dir) => {
    const c = COLS.find(x => x.id === col); if (!c || c.nosort) return 0;
    const x = c.get(a), y = c.get(b), xn = x == null || (typeof x === "number" && !isFinite(x)), yn = y == null || (typeof y === "number" && !isFinite(y));
    if (xn || yn) return xn && yn ? 0 : (xn ? 1 : -1);
    const d = c.num ? x - y : String(x).localeCompare(String(y));
    return dir === "asc" ? d : -d;
  };

  function rowsNow() { return R.filter(r => passes(r, sel, KEYS, getv, null)); }

  function renderFacets() {
    const groups = DIMS.map(d => {
      let vals = [...new Set(R.map(r => r[d.key]))];
      vals.sort((a, b) => (typeof a === "number" ? a - b : typeof a === "boolean" ? Number(b) - Number(a) : String(a).localeCompare(b)));
      return { key: d.key, label: d.label, values: vals.map(v => ({
        v, label: d.fmt ? d.fmt(v) : v, n: facetCount(R, sel, KEYS, getv, d.key, v), on: sel[d.key].has(v),
        color: d.key === "series" ? colorOf(v) : null })) };
    });
    const el = $("xf"); el.innerHTML = facetsHTML(groups);
    bindFacets(el, (k, raw) => {
      const v = k === "conc" ? Number(raw) : k === "ok" ? raw === "true" : raw;
      sel[k].has(v) ? sel[k].delete(v) : sel[k].add(v); render();
    });
  }

  function renderPlot(rows) {
    const box = $("xbox"), svg = $("xplot");
    const W = Math.max(280, Math.floor(box.clientWidth || 720));
    const H = W < 520 ? Math.round(W * 0.95) : Math.max(340, Math.min(460, Math.round(W * 0.5)));
    const m = { l: W < 520 ? 48 : 60, r: W < 520 ? 16 : 28, t: 14, b: 50 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const pts = rows.filter(r => isNum(r.cl) && isNum(r.pu));
    const mx = Math.max(0, ...pts.map(r => r.cl)), my = Math.max(0, ...pts.map(r => r.pu), floor || 0);
    const xs = niceScale(mx * 1.06), ys = niceScale(my * 1.1);
    const sx = linScale(0, xs.max, m.l, m.l + iw), sy = linScale(0, ys.max, m.t + ih, m.t);
    const tx = v => v >= 1000 ? f0(v) : (xs.step < 1 ? v.toFixed(1) : String(v)), ty = v => ys.step < 1 ? v.toFixed(1) : f0(v);
    const xt = W < 520 ? xs.ticks.filter((_, i) => i % 2 === 0 || xs.ticks.length <= 5) : xs.ticks;
    let g = "";
    ys.ticks.forEach(v => { const y = sy(v); g += `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}"/><text x="${m.l - 8}" y="${y + 3.5}" text-anchor="end">${ty(v)}</text>`; });
    xt.forEach(v => { const x = sx(v); g += `<line class="grid" x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ih}"/><text x="${x}" y="${m.t + ih + 17}" text-anchor="middle">${tx(v)}</text>`; });
    g += `<line class="axis" x1="${m.l}" x2="${m.l + iw}" y1="${m.t + ih}" y2="${m.t + ih}"/><line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/>`;
    g += `<text class="axtitle" x="${m.l + iw / 2}" y="${H - 8}" text-anchor="middle">Cluster throughput (tok/s) →</text>`;
    g += `<text class="axtitle" transform="translate(13 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">Per-user speed (tok/s/user) →</text>`;
    if (floor != null && isNum(floor)) {
      const y = sy(floor);
      g += `<line class="floorline" x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}"/><text class="floorlbl" x="${m.l + iw - 4}" y="${y - 5}" text-anchor="end">floor ${fmt(floor, 0)}</text>`;
    }
    if (!pts.length) g += `<text class="emptymsg" x="${m.l + iw / 2}" y="${m.t + ih / 2}" text-anchor="middle">No runs match these filters.</text>`;
    // draw not-ok first (hollow), then ok on top
    const draw = r => {
      const col = colorOf(r.series), mk = markerSVG(shapeOf(r), 4.6, col, !r.ok);
      const tt = `${r.series} · c=${r.conc} · ${f0(r.cl)} tok/s cluster, ${f1(r.pu)} tok/s/user${r.ok ? "" : " · not servable"}`;
      return `<g class="pt" data-id="${esc(r.id)}" transform="translate(${sx(r.cl).toFixed(1)} ${sy(r.pu).toFixed(1)})"><circle r="10" fill="transparent"/><g class="mk">${mk}</g><title>${esc(tt)}</title></g>`;
    };
    g += pts.filter(r => !r.ok).map(draw).join("") + pts.filter(r => r.ok).map(draw).join("");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.innerHTML = g;
    $("xleg").innerHTML = `<div class="grp"><span class="eyebrow">Config</span>` + series.map(v => {
      const n = rows.filter(r => r.series === v).length;
      return `<span class="it${n ? "" : " off"}"><svg viewBox="-7 -7 14 14">${markerSVG(["circle", "diamond", "square", "triangle"][series.indexOf(v) % 4], 5, colorOf(v), false)}</svg>${esc(v)} <i>${n}</i></span>`;
    }).join("") + `</div><div class="grp"><span class="eyebrow">Marker</span><span class="it"><i>filled = servable · hollow dashed = not servable</i></span></div>`;
  }

  function renderTable(rows) {
    const th = $("xhead");
    th.innerHTML = "<tr>" + COLS.map(c => {
      const active = c.id === sortState.col;
      return `<th class="${c.num ? "num " : ""}${c.nosort ? "nosort" : "th"}" data-c="${c.id}"${c.nosort ? "" : ` tabindex="0" aria-sort="${active ? (sortState.dir === "asc" ? "ascending" : "descending") : "none"}"`}>${c.nosort ? "" : `<span class="arrow">${active ? (sortState.dir === "asc" ? "▲" : "▼") : "↕"}</span>`}${esc(c.label)}</th>`;
    }).join("") + "</tr>";
    th.querySelectorAll("th.th").forEach(h => {
      const act = () => {
        const c = h.dataset.c;
        if (sortState.col === c) sortState.dir = sortState.dir === "asc" ? "desc" : "asc"; else { sortState.col = c; sortState.dir = COLS.find(x => x.id === c).num ? "desc" : "asc"; }
        renderTable(rowsNow());
      };
      h.onclick = act; h.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } };
    });
    const sorted = rows.map((r, i) => [r, i]).sort((a, b) => cmp(a[0], b[0], sortState.col, sortState.dir) || a[1] - b[1]).map(p => p[0]);
    const MAX = 300, shown = sorted.slice(0, MAX);
    $("xbodyrows").innerHTML = shown.length ? shown.map(r => `<tr class="row ${r.ok ? "real" : "mockrow"}">
      <td class="first cfgcell" style="color:inherit">${esc(r.series)}</td>
      <td class="num" data-label="Conc">${esc(r.conc)}</td><td class="num" data-label="GPUs">${esc(r.gpus != null ? r.gpus : "—")}</td>
      <td class="num" data-label="Cluster tok/s">${f1(r.cl)}</td><td class="num" data-label="Per-user tok/s">${f1(r.pu)}</td>
      <td class="num" data-label="ITL mean (ms)">${f1(r.itl)}</td><td class="num" data-label="TTFT (ms)">${f0(r.ttft)}</td>
      <td data-label="Servable">${r.ok ? "yes" : "no"}</td>
      <td data-label="Limits hit">${Array.isArray(r.viol) && r.viol.length ? esc(r.viol.join(", ")) : "—"}</td></tr>`).join("")
      : `<tr class="emptyrow"><td colspan="${COLS.length}">No runs match these filters.</td></tr>`;
    $("xnote").textContent = sorted.length > MAX ? `Showing first ${MAX} of ${sorted.length} runs in this sort order.` : "";
  }

  function render() {
    const rows = rowsNow();
    renderFacets(); renderPlot(rows); renderTable(rows);
    const ok = rows.filter(r => r.ok).length;
    $("xstatus").textContent = `${rows.length} of ${R.length} runs · ${ok} servable`;
    const any = KEYS.some(k => sel[k].size);
    $("xreset").hidden = !any;
    $("xreset").onclick = () => { KEYS.forEach(k => sel[k].clear()); render(); };
  }
  setResize(() => render());
  render();
}
