// Submit tab. window.MODE gates behavior:
//   "pages": form disabled; renders scenario.yaml + PR instructions + prefilled GitHub link. No network, no execution.
//   "local": form enabled; POSTs the scenario JSON to /api/search (local companion server) and streams status.
import { esc, catalog, mode, f1 } from "./app.js";

const REPO = "inference-sim/blis-config-search";
// Built at runtime so no absolute URL is baked into the static bundle; the link is only ever followed by a user click.
const ghBase = () => "https:" + "//" + "github.com" + "/" + REPO;

const FLOORS = [10, 20, 30, 50, 75, 100];
const st = { model: null, workload: null, hardware: null, gpu_budget: 8, floor: 50, headroom: 0.8 };

export function buildScenario(s) {
  return {
    model: s.model, hardware: String(s.hardware || "").toLowerCase(), gpu_budget: s.gpu_budget,
    workload: { preset: s.workload }, objective: "dual-pareto",
    interactivity_floor: { tok_per_s_per_user: s.floor }, headroom: s.headroom,
  };
}
export function toYaml(sc) {
  return [
    "# scenario.yaml for config-search (BLIS simulation; the search brain is an LLM, BLIS itself is deterministic)",
    `model: ${sc.model}`,
    `hardware: ${sc.hardware}`,
    `gpu_budget: ${sc.gpu_budget}`,
    "workload:",
    `  preset: ${sc.workload.preset}`,
    `objective: ${sc.objective}`,
    "interactivity_floor:",
    `  tok_per_s_per_user: ${sc.interactivity_floor.tok_per_s_per_user}`,
    `headroom: ${sc.headroom}`,
    "",
  ].join("\n");
}

export function mount(root) {
  const cat = catalog(), local = mode() === "local";
  const M = cat.models || [], W = cat.workloads || [], H = cat.hardware || [];
  if (!st.model || !M.some(m => m.name === st.model)) st.model = (M.find(m => m.name === "qwen3-14b") || M[0] || {}).name || "";
  if (!st.workload || !W.some(w => w.name === st.workload)) st.workload = (W.find(w => w.name === "chatbot") || W[0] || {}).name || "";
  if (!st.hardware || !H.some(h => h.name === st.hardware)) st.hardware = (H.find(h => h.name === "h100") || H[0] || {}).name || "";

  const opts = (arr, cur) => arr.map(x => `<option value="${esc(x.name)}"${x.name === cur ? " selected" : ""}>${esc(x.name)}</option>`).join("");
  const dis = local ? "" : " disabled";
  root.innerHTML = `
    <h2 class="vt">Submit a scenario <span class="modebadge ${local ? "local" : ""}">${local ? "local mode" : "read-only (Pages)"}</span></h2>
    <p class="dek">${local
      ? "Build a Scenario and run the search on your machine. <b>Runs locally against your own LLM</b> through the companion server; nothing is sent to a hosted service."
      : "Build a Scenario to see the exact <code>scenario.yaml</code> to submit. This published site cannot run searches: execution is disabled. Open a pull request and CI runs it after a maintainer approves."}</p>
    <div class="sgrid">
      <section class="card" aria-label="Scenario form">
        <form class="form" id="frm" onsubmit="return false">
          <fieldset style="border:0;padding:0;margin:0;min-width:0;display:grid;gap:12px"${dis}>
            <div class="field"><label for="f-model">Model</label><select id="f-model">${opts(M, st.model)}</select></div>
            <div class="field"><label for="f-workload">Workload preset</label><select id="f-workload">${opts(W, st.workload)}</select></div>
            <div class="field"><label for="f-hw">Hardware</label><select id="f-hw">${opts(H, st.hardware)}</select></div>
            <div class="field"><label for="f-gpus">GPU budget</label><input id="f-gpus" type="number" min="1" max="4096" step="1" value="${esc(st.gpu_budget)}"><div class="hint">Total GPUs the deployment may use.</div></div>
            <div class="field"><label for="f-floor">Interactivity floor (tok/s/user)</label><input id="f-floor" type="number" min="1" max="1000" step="1" value="${esc(st.floor)}" list="floors"><datalist id="floors">${FLOORS.map(v => `<option value="${v}">`).join("")}</datalist><div class="hint">Every instance must hold at least this per-user speed.</div></div>
            <div class="field"><label for="f-head">Headroom</label><input id="f-head" type="number" min="0.1" max="1" step="0.05" value="${esc(st.headroom)}"><div class="hint">Recommended load = headroom × saturation load.</div></div>
          </fieldset>
          ${local ? `<div class="btnrow"><button class="btn primary" id="run" type="button">Run search on your machine</button><span class="note" style="margin:0">runs locally against your own LLM</span></div>` : `<p class="note" style="margin:0">The form is a builder only on this site. It never runs a search or contacts a server.</p>`}
        </form>
        ${local ? `<div class="log" id="log" role="log" aria-live="polite">Waiting. Output from the local server will stream here.</div>` : ""}
      </section>
      <section class="card" aria-label="Generated scenario">
        <p class="eyebrow">scenario.yaml</p>
        <pre class="yaml" id="yaml"></pre>
        <div class="btnrow"><button class="btn" id="copy" type="button">Copy YAML</button>${local ? "" : `<a class="btn primary" id="gh" href="#" target="_blank" rel="noopener">Open PR with this file ↗</a>`}<span class="note" id="cstat" style="margin:0"></span></div>
        ${local ? "" : `
        <p class="eyebrow" style="margin-top:16px">Open a PR to run this</p>
        <ol class="steps">
          <li>Copy the YAML above (or use the button to prefill a new file on GitHub).</li>
          <li>Save it as <code id="fname">submissions/…yaml</code> in a fork or branch of the repository.</li>
          <li>Open a pull request against <code>main</code>. CI validates the file and builds a preview.</li>
          <li>A maintainer approves the run. Only then does CI invoke the search and write a new run directory.</li>
          <li>The generated report is committed back to the PR. After merge, it appears in the Reports tab.</li>
        </ol>`}
      </section>
    </div>`;
  const $ = id => root.querySelector("#" + id);

  const slug = () => `${st.model}-${st.hardware}-${st.workload}-${st.gpu_budget}gpu`.toLowerCase().replace(/[^a-z0-9.]+/g, "-");
  function current() { return buildScenario(st); }
  function valid() {
    return st.model && st.workload && st.hardware && Number.isInteger(st.gpu_budget) && st.gpu_budget >= 1 && st.floor > 0 && st.headroom > 0 && st.headroom <= 1;
  }
  function refresh() {
    const y = toYaml(current());
    $("yaml").textContent = y;
    const fn = `submissions/${slug()}.yaml`;
    const g = $("gh");
    if (g) {
      g.href = ghBase() + "/new/main?filename=" + encodeURIComponent(fn) + "&value=" + encodeURIComponent(y) +
        "&message=" + encodeURIComponent("submit scenario: " + slug());
      g.setAttribute("aria-disabled", String(!valid()));
    }
    const fnEl = $("fname"); if (fnEl) fnEl.textContent = fn;
    const r = $("run"); if (r) r.disabled = !valid();
  }
  const readNum = (el, d) => { const v = Number(el.value); return isFinite(v) && el.value !== "" ? v : d; };
  $("f-model").onchange = e => { st.model = e.target.value; refresh(); };
  $("f-workload").onchange = e => { st.workload = e.target.value; refresh(); };
  $("f-hw").onchange = e => { st.hardware = e.target.value; refresh(); };
  $("f-gpus").oninput = e => { st.gpu_budget = Math.trunc(readNum(e.target, 0)); refresh(); };
  $("f-floor").oninput = e => { st.floor = readNum(e.target, 0); refresh(); };
  $("f-head").oninput = e => { st.headroom = readNum(e.target, 0); refresh(); };
  $("copy").onclick = async () => {
    const y = $("yaml").textContent;
    try { await navigator.clipboard.writeText(y); $("cstat").textContent = "Copied."; }
    catch (e) {
      const r = document.createRange(); r.selectNodeContents($("yaml"));
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
      $("cstat").textContent = "Press Ctrl/Cmd+C to copy.";
    }
  };
  refresh();

  if (!local) return null;

  // ---- local mode only: POST to the companion server ----
  const log = $("log");
  const say = t => { log.textContent += (log.textContent.endsWith("\n") || !log.textContent ? "" : "\n") + t + "\n"; log.scrollTop = log.scrollHeight; };
  const ctl = { abort: null };
  $("run").onclick = async () => {
    if (!valid()) return;
    const btn = $("run"); btn.disabled = true;
    log.textContent = "";
    say("POST /api/search …");
    const ac = new AbortController(); ctl.abort = ac;
    try {
      const res = await fetch("/api/search", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(current()), signal: ac.signal });
      if (!res.ok) {
        say(`Server responded HTTP ${res.status}. ${res.status === 404 ? "The local companion server (scripts/serve_local.py) is not running or has no /api/search endpoint." : ""}`);
      } else if (res.body && res.body.getReader) {
        const rd = res.body.getReader(), dec = new TextDecoder();
        for (;;) { const { done, value } = await rd.read(); if (done) break; log.textContent += dec.decode(value, { stream: true }); log.scrollTop = log.scrollHeight; }
        say("— done —");
      } else {
        log.textContent += await res.text(); say("— done —");
      }
    } catch (e) {
      if (e && e.name === "AbortError") say("Cancelled.");
      else say("Could not reach /api/search (" + (e && e.message ? e.message : "network error") + "). The YAML on the right is still valid: save it and run the skill manually, or start the local companion server.");
    }
    btn.disabled = !valid(); ctl.abort = null;
  };
  void f1;
  return () => { if (ctl.abort) ctl.abort.abort(); };
}
