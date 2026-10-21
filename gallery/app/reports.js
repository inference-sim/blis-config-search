// Reports tab: sortable table (default) + Pareto plot over the same crossfiltered rows.
// Ported from the single-page gallery. A row/point click opens #/scenario/<run_id>;
// a separate "report ↗" link goes to the standalone <run_id>/index.html.
import {
  esc, scenarios, f0, f1, fmt, hrefFor, scenarioHref, lc, sortRows as sortGeneric,
  makeSel, passes, facetCount, facetsHTML, bindFacets, niceScale, linScale, sizeOf, markerSVG,
  PALETTE, replaceHash, reportsHref,
} from "./app.js";

const FACETS = [
  { key: "model", label: "Model" },
  { key: "hardware", label: "Hardware" },
  { key: "workload", label: "Workload" },
  { key: "gpu_budget", label: "GPU budget", fmt: v => v + "×" },
];
const KEYS = FACETS.map(f => f.key);
const getv = (s, k) => s[k];
const COLS = [
  { id: "model", label: "Model", get: s => s.model },
  { id: "hardware", label: "Hardware", get: s => s.hardware },
  { id: "gpu_budget", label: "GPUs", get: s => s.gpu_budget, num: true },
  { id: "workload", label: "Workload", get: s => s.workload },
  { id: "config", label: "Recommended config", get: s => s.winner.label },
  { id: "tps", label: "tok/s", get: s => s.winner.tps, num: true, best: "max" },
  { id: "tps_per_gpu", label: "tok/s/GPU", get: s => s.winner.tps_per_gpu, num: true, best: "max" },
  { id: "per_user", label: "tok/s/user", get: s => s.winner.per_user, num: true, best: "max" },
  { id: "itl_p99", label: "ITL p99", get: s => s.winner.itl_p99, num: true, best: "min" },
  { id: "servable", label: "Servable", get: s => (s.n_runs ? s.n_servable / s.n_runs : 0), num: true },
  { id: "load", label: "Load (c)", get: s => s.winner.c_use, num: true },
  { id: "report", label: "Report", get: s => s.run_id, nosort: true },
];
const LBL = id => esc(COLS.find(c => c.id === id).label);

// state survives tab switches within a session
const st = { sortCol: "tps", sortDir: "desc", expanded: new Set(), showFin: false, view: null };

function bestIds(rows) { // colId -> run_id of best REAL row; synthetic never wins; needs >1 real row
  const out = {}, real = rows.filter(r => !r.synthetic);
  if (real.length < 2) return out;
  COLS.filter(c => c.best).forEach(c => {
    let b = real[0];
    real.forEach(r => { const v = c.get(r), bv = c.get(b); if (c.best === "max" ? v > bv : v < bv) b = r; });
    out[c.id] = b.run_id;
  });
  return out;
}

function encodings(S) {
  const MODELS = [...new Set(S.map(s => s.model))].sort(), HWS = [...new Set(S.map(s => s.hardware))].sort();
  const SHAPES = ["circle", "diamond", "square", "triangle"];
  return {
    MODELS, HWS,
    colorOf: m => PALETTE[MODELS.indexOf(m)] || "var(--plot-other)",
    shapeOf: h => SHAPES[Math.max(0, HWS.indexOf(h)) % SHAPES.length],
  };
}

function plotData(rows, withFinalists) {
  const winners = rows.map(s => ({ s, x: s.winner.tps, y: s.winner.per_user }));
  const finals = [];
  if (withFinalists) rows.forEach(s => (s.finalists || []).forEach(f => {
    if (f.label !== s.winner.label) finals.push({ s, f, x: f.tps, y: f.per_user });
  }));
  const all = winners.concat(finals);
  const mx = Math.max(0, ...all.map(p => p.x).filter(Number.isFinite));
  const my = Math.max(0, ...all.map(p => p.y).filter(Number.isFinite));
  return { winners, finals, xs: niceScale(mx * 1.06), ys: niceScale(my * 1.1) };
}
const maxPerGpu = rows => Math.max(0, ...rows.map(s => s.winner.tps_per_gpu).filter(Number.isFinite));

/* values in the hash may differ in case from the data (catalog "h100" vs scenario "H100"); resolve to real values */
function selFromParams(S, params) {
  const sel = makeSel(KEYS);
  KEYS.forEach(k => {
    const raw = params[k]; if (!raw) return;
    const vals = [...new Set(S.map(s => s[k]))];
    String(raw).split(",").forEach(p => {
      const hit = vals.find(v => lc(v) === lc(p));
      if (hit !== undefined) sel[k].add(hit);
    });
  });
  return sel;
}

export function mount(root, route) {
  const S = scenarios();
  const sel = selFromParams(S, route.params || {});
  const ENC = encodings(S), colorOf = ENC.colorOf, shapeOf = ENC.shapeOf;
  if (st.view === null) {
    st.view = "table";
    try { const sv = localStorage.getItem("cs-view"); if (sv === "plot" || sv === "table") st.view = sv; } catch (e) { /* ignore */ }
  }
  let hoverId = null, lastPtr = "mouse", geom = null, rowsNow = [];

  root.innerHTML = `
  <div class="rview" id="rview" data-view="${st.view}">
    <p class="dek">Every scenario, one row: the Pareto-optimal serving config BLIS recommends for a model, hardware, GPU budget and workload. Switch between a sortable table and a Pareto plot; click a row or point to open the scenario briefing and its full run explorer.</p>
    <section class="summary" id="summary" aria-live="polite"></section>
    <section class="facets" id="facets" aria-label="Filters"></section>
    <div class="toolbar">
      <span id="status" class="mono"></span>
      <label class="tright tableonly" style="gap:6px"><select class="sortsel" id="sortsel" aria-label="Sort scenarios by"></select></label>
      <span class="tright">
        <label class="toggle plotonly"><input type="checkbox" id="fin"> show all finalists</label>
        <span class="seg" id="viewseg" role="group" aria-label="View"><button type="button" data-v="table">Table</button><button type="button" data-v="plot">Plot</button></span>
        <button class="lnk tableonly" id="expandall" type="button">expand all finalists</button>
        <button class="lnk" id="reset" type="button" hidden>clear filters</button>
      </span>
    </div>
    <div class="tablewrap tableonly"><table id="tbl" class="wide"><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
    <section class="plotcard" id="plotview" aria-label="Pareto plot view">
      <h2>Throughput vs interactivity</h2>
      <p class="psub">Each point is one scenario's recommended config at its usable operating point. Right = more cluster throughput, up = faster per user. Models are not ranked against each other.</p>
      <div class="mockstrip" id="mockstrip"></div>
      <div class="plotbox" id="plotbox">
        <svg class="plot" id="plot" role="img" aria-label="Scatter plot of each scenario's winning config: total cluster tokens per second on X, tokens per second per user on Y"></svg>
        <div class="tip" id="tip" role="status"></div>
      </div>
      <div class="plegend" id="plegend"></div>
    </section>
    <div class="legend tableonly">
      <span><span class="sw"></span>best value in column (real BLIS rows only)</span>
      <span><span class="sw m"></span><span class="mockflag">mock</span> illustrative scenario, not BLIS output</span>
      <span>tok/s columns are at the usable operating point (headroom-adjusted)</span>
    </div>
  </div>`;
  const $ = id => root.querySelector("#" + id);
  const rview = $("rview");
  const mockTag = s => s.synthetic ? ` <span class="mockflag">mock</span>` : "";

  function syncHash() {
    const p = {};
    KEYS.forEach(k => { if (sel[k].size) p[k] = [...sel[k]].join(","); });
    replaceHash(reportsHref(p));
  }
  const filterRows = () => S.filter(s => passes(s, sel, KEYS, getv, null));

  function renderSummary(rows) {
    const u = k => new Set(rows.map(s => s[k])).size;
    const runs = rows.reduce((a, s) => a + (s.n_servable || 0), 0), mock = rows.filter(s => s.synthetic).length;
    const m = (n, l) => `<span class="sm"><span class="sv">${n}</span>${l}</span>`;
    $("summary").innerHTML =
      m(rows.length, rows.length === 1 ? "scenario" : "scenarios") + m(runs, runs === 1 ? "servable run" : "servable runs") +
      m(u("model"), u("model") === 1 ? "model" : "models") + m(u("hardware"), u("hardware") === 1 ? "hardware type" : "hardware types") +
      `<span>in view${rows.length < S.length ? ` (of ${S.length})` : ""}</span>` +
      (mock ? `<span><span class="mockflag">mock</span> ${mock} illustrative</span>` : "") +
      `<span style="flex-basis:100%;font-size:12px">Each row carries its own recommended config. Throughput is not comparable across models or workloads.</span>`;
  }

  function renderFacets() {
    const el = $("facets");
    const groups = FACETS.map(f => {
      const vals = [...new Set(S.map(s => s[f.key]))].sort((a, b) => typeof a === "number" ? a - b : String(a).localeCompare(b));
      return {
        key: f.key, label: f.label,
        values: vals.map(v => ({ v, label: f.fmt ? f.fmt(v) : v, n: facetCount(S, sel, KEYS, getv, f.key, v), on: sel[f.key].has(v),
          color: f.key === "model" ? colorOf(v) : null })),
      };
    });
    el.innerHTML = facetsHTML(groups);
    bindFacets(el, (k, raw) => {
      const v = k === "gpu_budget" ? Number(raw) : raw;
      sel[k].has(v) ? sel[k].delete(v) : sel[k].add(v);
      render();
    });
  }

  function renderHead() {
    $("thead").innerHTML = "<tr>" + COLS.map(c => {
      const active = c.id === st.sortCol;
      const aria = c.nosort ? "" : ` aria-sort="${active ? (st.sortDir === "asc" ? "ascending" : "descending") : "none"}"`;
      const arrow = c.nosort ? "" : `<span class="arrow">${active ? (st.sortDir === "asc" ? "▲" : "▼") : "↕"}</span>`;
      return `<th class="${c.num ? "num " : ""}${c.nosort ? "nosort" : "th"}" data-c="${c.id}"${aria}${c.nosort ? "" : ' tabindex="0" role="columnheader button"'}>${arrow}${esc(c.label)}</th>`;
    }).join("") + "</tr>";
    const ss = $("sortsel");
    ss.innerHTML = COLS.filter(c => !c.nosort).map(c => `<option value="${c.id}">Sort: ${esc(c.label)}</option>`).join("");
    ss.value = st.sortCol;
    ss.onchange = () => {
      if (ss.value === st.sortCol) return;
      st.sortCol = ss.value; const c = COLS.find(x => x.id === st.sortCol);
      st.sortDir = (c.num && c.best !== "min") ? "desc" : "asc"; render();
    };
    $("thead").querySelectorAll("th.th").forEach(th => {
      const act = () => {
        const id = th.dataset.c;
        if (st.sortCol === id) st.sortDir = st.sortDir === "asc" ? "desc" : "asc";
        else { st.sortCol = id; const c = COLS.find(x => x.id === id); st.sortDir = (c.num && c.best !== "min") ? "desc" : "asc"; }
        render();
      };
      th.onclick = act;
      th.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } };
    });
  }

  function renderBody(rows) {
    const tb = $("tbody");
    if (!rows.length) {
      tb.innerHTML = `<tr class="emptyrow"><td colspan="${COLS.length}">No scenarios match this filter. <button class="lnk" id="reset2" type="button">Clear filters</button></td></tr>`;
      const r2 = $("reset2"); if (r2) r2.onclick = clearAll; return;
    }
    const sorted = sortGeneric(rows, COLS, st.sortCol, st.sortDir), best = bestIds(sorted);
    const bc = (id, s) => !s.synthetic && best[id] === s.run_id ? " best" : "";
    tb.innerHTML = sorted.map(s => {
      const w = s.winner, open = st.expanded.has(s.run_id), href = scenarioHref(s.run_id);
      const pct = s.n_runs ? Math.round(100 * s.n_servable / s.n_runs) : 0;
      let h = `<tr class="row click ${s.synthetic ? "mockrow" : "real"}" tabindex="0" data-id="${esc(s.run_id)}" aria-label="Open scenario ${esc(s.model)} on ${esc(s.hardware)}">
        <td class="first"><button class="exp" type="button" data-id="${esc(s.run_id)}" aria-expanded="${open}" aria-label="Toggle finalists">${open ? "−" : "+"}</button><a class="mname" href="${href}">${esc(s.model)}</a>${mockTag(s)}</td>
        <td data-label="${LBL("hardware")}">${esc(s.hardware)}</td>
        <td class="num" data-label="${LBL("gpu_budget")}">${esc(s.gpu_budget)}</td>
        <td data-label="${LBL("workload")}">${esc(s.workload)} <span class="chip">${esc(s.isl_osl)}</span></td>
        <td class="cfgcell" data-label="${LBL("config")}">${esc(w.label)}</td>
        <td class="num${bc("tps", s)}" data-label="${LBL("tps")}"><span>${f0(w.tps)}</span></td>
        <td class="num${bc("tps_per_gpu", s)}" data-label="${LBL("tps_per_gpu")}"><span>${f1(w.tps_per_gpu)}</span></td>
        <td class="num${bc("per_user", s)}" data-label="${LBL("per_user")}"><span>${f1(w.per_user)}</span></td>
        <td class="num${bc("itl_p99", s)}" data-label="${LBL("itl_p99")}"><span>${f1(w.itl_p99)}</span><span class="u">ms</span></td>
        <td class="num" data-label="${LBL("servable")}"><span class="srv"><i><b style="width:${pct}%"></b></i>${esc(s.n_servable)}/${esc(s.n_runs)}</span></td>
        <td class="num" data-label="${LBL("load")}">${esc(w.c_use)}</td>
        <td class="rep"><a href="${hrefFor(s.run_id)}" target="_blank" rel="noopener">full report</a></td>
      </tr>`;
      if (open) {
        h += (s.finalists || []).slice().sort((a, b) => b.tps - a.tps).map(f => {
          const isW = f.label === w.label;
          return `<tr class="sub"><td class="first">${isW ? '<span class="pick">recommended</span>' : "finalist"}</td><td></td><td class="num" data-label="${LBL("gpu_budget")}">${esc(f.gpus)}</td><td></td>
            <td class="mono cfgcell" style="color:inherit" data-label="${LBL("config")}">${esc(f.label)}</td>
            <td class="num" data-label="${LBL("tps")}">${f0(f.tps)}</td><td class="num" data-label="${LBL("tps_per_gpu")}">${f1(f.tps_per_gpu)}</td><td class="num" data-label="${LBL("per_user")}">${f1(f.per_user)}</td>
            <td class="num" data-label="${LBL("itl_p99")}">${f1(f.itl_p99)}<span class="u">ms</span></td><td></td><td class="num" data-label="${LBL("load")}">${esc(f.c_use)}</td><td></td></tr>`;
        }).join("");
      }
      return h;
    }).join("");
    tb.querySelectorAll("tr.row").forEach(tr => {
      const go = () => { location.hash = scenarioHref(tr.dataset.id); };
      tr.onclick = e => { if (e.target.closest(".exp") || e.target.closest("a")) return; go(); };
      tr.onkeydown = e => { if (e.target === tr && e.key === "Enter") go(); };
    });
    tb.querySelectorAll(".exp").forEach(b => b.onclick = e => {
      e.stopPropagation(); const id = b.dataset.id;
      st.expanded.has(id) ? st.expanded.delete(id) : st.expanded.add(id); render();
    });
  }

  /* ---------------- plot ---------------- */
  function renderPlot(rows) {
    const box = $("plotbox"), svg = $("plot");
    const W = Math.max(280, Math.floor(box.clientWidth || 720));
    const H = W < 520 ? Math.round(W * 0.95) : Math.max(380, Math.min(500, Math.round(W * 0.56)));
    const m = { l: W < 520 ? 48 : 60, r: W < 520 ? 26 : 40, t: 16, b: 50 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const D = plotData(rows, st.showFin), pgMax = maxPerGpu(rows);
    const sx = linScale(0, D.xs.max, m.l, m.l + iw), sy = linScale(0, D.ys.max, m.t + ih, m.t);
    const tickX = v => v >= 1000 ? f0(v) : (D.xs.step < 1 ? v.toFixed(1) : String(v));
    const tickY = v => D.ys.step < 1 ? v.toFixed(1) : f0(v);
    const xt = W < 520 ? D.xs.ticks.filter((_, i) => i % 2 === 0 || D.xs.ticks.length <= 5) : D.xs.ticks;
    let g = "";
    D.ys.ticks.forEach(v => { const y = sy(v);
      g += `<line class="grid" x1="${m.l}" x2="${m.l + iw}" y1="${y}" y2="${y}"/><text x="${m.l - 8}" y="${y + 3.5}" text-anchor="end">${tickY(v)}</text>`; });
    xt.forEach(v => { const x = sx(v);
      g += `<line class="grid" x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ih}"/><text x="${x}" y="${m.t + ih + 17}" text-anchor="middle">${tickX(v)}</text>`; });
    g += `<line class="axis" x1="${m.l}" x2="${m.l + iw}" y1="${m.t + ih}" y2="${m.t + ih}"/><line class="axis" x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ih}"/>`;
    g += `<text class="axtitle" x="${m.l + iw / 2}" y="${H - 8}" text-anchor="middle">Total cluster throughput (tok/s) →</text>`;
    g += `<text class="axtitle" transform="translate(13 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">Per-user speed (tok/s/user) →</text>`;
    if (iw > 380) g += `<text class="better" x="${m.l + iw - 6}" y="${m.t + 14}" text-anchor="end">BETTER ↗</text>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
    if (!rows.length) {
      g += `<text class="emptymsg" x="${m.l + iw / 2}" y="${m.t + ih / 2}" text-anchor="middle">No scenarios match these filters.</text>`;
      svg.innerHTML = g; return { sx, sy, W, H };
    }
    if (st.showFin) rows.forEach(s => {
      const pts = (s.finalists || []).slice().sort((a, b) => a.tps - b.tps).map(f => [sx(f.tps), sy(f.per_user)]);
      if (pts.length > 1) g += `<polyline points="${pts.map(p => p.map(n => n.toFixed(1)).join(",")).join(" ")}" fill="none" stroke="${colorOf(s.model)}" stroke-opacity=".4" stroke-width="1.4" stroke-dasharray="${s.synthetic ? "4 3" : "none"}"/>`;
    });
    D.finals.forEach(p => {
      g += `<g transform="translate(${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)})"><title>${esc(p.s.model + " · " + p.f.label + " — " + f0(p.f.tps) + " tok/s, " + f1(p.f.per_user) + " tok/s/user")}</title>`
        + markerSVG(shapeOf(p.s.hardware), 3.2, colorOf(p.s.model), !!p.s.synthetic) + `</g>`;
    });
    D.winners.forEach(p => {
      const s = p.s, cx = sx(p.x), cy = sy(p.y), col = colorOf(s.model), kind = shapeOf(s.hardware);
      const flip = cx > m.l + iw - 54, r = sizeOf(s.winner.tps_per_gpu, pgMax);
      g += `<a class="pt" href="${scenarioHref(s.run_id)}" data-id="${esc(s.run_id)}" tabindex="0"
          aria-label="${esc(s.model + " on " + s.hardware + ", " + s.gpu_budget + " GPUs, " + s.workload + (s.synthetic ? " (mock)" : "") + ": " + s.winner.label + ", " + f0(s.winner.tps) + " tok/s, " + f1(s.winner.per_user) + " tok/s per user. Open scenario.")}">
        <g transform="translate(${cx.toFixed(1)} ${cy.toFixed(1)})">
          <circle r="${(r + 8).toFixed(1)}" fill="transparent"/>
          ${s.synthetic ? "" : `<circle r="${(r + 4).toFixed(1)}" fill="${col}" fill-opacity=".18"/>`}
          <g class="mk">${markerSVG(kind, r, col, !!s.synthetic)}</g>
          <text class="lbl" x="${flip ? -(r + 7) : r + 7}" y="3.5" text-anchor="${flip ? "end" : "start"}">${esc(s.gpu_budget)}×</text>
        </g></a>`;
    });
    svg.innerHTML = g;
    svg.querySelectorAll(".pt").forEach(a => {
      const id = a.dataset.id;
      a.addEventListener("mouseenter", () => setHover(id));
      a.addEventListener("mouseleave", clearHover);
      a.addEventListener("focus", () => setHover(id));
      a.addEventListener("blur", clearHover);
      a.addEventListener("pointerdown", e => { lastPtr = e.pointerType; });
      a.addEventListener("click", e => { if (lastPtr === "touch" && hoverId !== id) { e.preventDefault(); setHover(id); } });
    });
    return { sx, sy, W, H };
  }
  function setHover(id) {
    hoverId = id;
    const p = $("plot"); p.classList.add("has-hl");
    p.querySelectorAll(".pt").forEach(e => e.classList.toggle("hl", e.dataset.id === id));
    showTip(id);
  }
  function clearHover() {
    hoverId = null;
    $("plot").classList.remove("has-hl");
    $("plot").querySelectorAll(".pt.hl").forEach(e => e.classList.remove("hl"));
    $("tip").classList.remove("on");
  }
  function showTip(id) {
    const s = rowsNow.find(r => r.run_id === id), tip = $("tip");
    if (!s || !geom) { tip.classList.remove("on"); return; }
    const w = s.winner;
    tip.innerHTML = `
      <div class="t1"><span>${esc(s.model)} · ${esc(s.hardware)}</span>${s.synthetic ? `<span class="mockflag">mock</span>` : `<span class="real">real run</span>`}</div>
      <div class="t2">${esc(w.label)}</div>
      <div class="t3">${esc(s.gpu_budget)} GPU budget · ${esc(s.workload)} · ${esc(s.isl_osl)}</div>
      <dl><dt>cluster</dt><dd>${f1(w.tps)} tok/s</dd><dt>per GPU</dt><dd>${f1(w.tps_per_gpu)} tok/s</dd>
      <dt>per user</dt><dd>${f1(w.per_user)} tok/s</dd><dt>ITL p99</dt><dd>${f1(w.itl_p99)} ms</dd></dl>
      <div class="go">Click to open scenario →</div>`;
    const cx = geom.sx(w.tps), cy = geom.sy(w.per_user);
    tip.style.left = "0px"; tip.style.top = "0px";
    const tw = tip.offsetWidth || 0, th = tip.offsetHeight || 0;
    let left = cx + 18; if (left + tw > geom.W - 4) left = cx - tw - 18;
    left = Math.max(4, Math.min(left, geom.W - tw - 4));
    const top = Math.max(4, Math.min(cy - th / 2, geom.H - th - 4));
    tip.style.left = left + "px"; tip.style.top = top + "px";
    tip.classList.add("on");
  }
  const glyphSVG = (kind, mock) => `<svg viewBox="-7 -7 14 14">${markerSVG(kind, 5, "var(--ink)", mock).replace('stroke="var(--panel)"', 'stroke="none"')}</svg>`;
  function renderLegend(rows) {
    const cnt = (k, v) => rows.filter(s => s[k] === v).length;
    let h = `<div class="grp"><span class="eyebrow">Model</span>`;
    ENC.MODELS.forEach(m => { const n = cnt("model", m);
      h += `<span class="it${n ? "" : " off"}"><svg viewBox="-7 -7 14 14"><circle r="5" fill="${colorOf(m)}"/></svg>${esc(m)} <i>${n}</i></span>`; });
    h += `</div><div class="grp"><span class="eyebrow">Hardware</span>`;
    ENC.HWS.forEach(x => { const n = cnt("hardware", x);
      h += `<span class="it${n ? "" : " off"}">${glyphSVG(shapeOf(x), false)}${esc(x)} <i>${n}</i></span>`; });
    h += `</div><div class="grp"><span class="eyebrow">Source</span>
      <span class="it">${glyphSVG("circle", false)}real BLIS run</span>
      <span class="it">${glyphSVG("circle", true)}mock (dashed, hollow)</span>
      <span class="it"><i>label = GPU budget · size = tok/s/GPU</i></span></div>`;
    $("plegend").innerHTML = h;
  }
  function renderPlotView(rows) {
    rowsNow = rows;
    const mocks = rows.filter(s => s.synthetic).length, strip = $("mockstrip");
    if (mocks) {
      strip.classList.add("on");
      strip.innerHTML = mocks === rows.length
        ? `<b>MOCK DATA.</b> Every point in view is a synthetic placeholder, not a BLIS result.`
        : `<b>${mocks} of ${rows.length}</b> points in view are MOCK (dashed, hollow markers). Only solid markers are real BLIS runs.`;
    } else strip.classList.remove("on");
    geom = renderPlot(rows); renderLegend(rows); hoverId = null;
  }

  function setView(v) {
    st.view = v === "plot" ? "plot" : "table";
    rview.dataset.view = st.view;
    try { localStorage.setItem("cs-view", st.view); } catch (e) { /* ignore */ }
    render();
  }
  function clearAll() { KEYS.forEach(k => sel[k].clear()); render(); }

  function render() {
    const rows = filterRows();
    renderSummary(rows); renderFacets();
    root.querySelectorAll("#viewseg button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === st.view)));
    if (st.view === "plot") renderPlotView(rows); else { renderHead(); renderBody(rows); }
    const any = KEYS.some(k => sel[k].size);
    $("reset").hidden = !any; $("reset").onclick = clearAll;
    const nMock = rows.filter(r => r.synthetic).length;
    $("status").textContent = `${rows.length} of ${S.length} scenarios` + (nMock ? ` · ${nMock} mock` : "");
    const allOpen = rows.length && rows.every(r => st.expanded.has(r.run_id));
    $("expandall").textContent = allOpen ? "collapse all finalists" : "expand all finalists";
    $("expandall").onclick = () => { rows.forEach(r => allOpen ? st.expanded.delete(r.run_id) : st.expanded.add(r.run_id)); render(); };
    syncHash();
  }

  root.querySelectorAll("#viewseg button").forEach(b => b.onclick = () => { if (b.dataset.v !== st.view) setView(b.dataset.v); });
  $("fin").checked = st.showFin;
  $("fin").onchange = e => { st.showFin = e.target.checked; render(); };
  let rt; const onResize = () => { clearTimeout(rt); rt = setTimeout(() => { if (st.view === "plot") render(); }, 80); };
  window.addEventListener("resize", onResize);
  render();
  return () => { window.removeEventListener("resize", onResize); clearTimeout(rt); };
}
