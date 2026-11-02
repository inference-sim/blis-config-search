// explorer_smoketest.js -- drives the rendered report's inline explorer IIFE against a record set
// that includes a NULL-bearing row (pu/cl/itl = null, as build_records emits for rejected/MTP/
// missing-metric runs), with minimal d3 + crossfilter shims, to prove the explorer renders without
// throwing (review F4). Exits non-zero (with a message) on any error. Invoked by test_report.py.
//
// Usage: node explorer_smoketest.js <path-to-index.html>
const fs = require("fs");

const htmlPath = process.argv[2];
if (!htmlPath) { console.error("usage: node explorer_smoketest.js <index.html>"); process.exit(2); }
const html = fs.readFileSync(htmlPath, "utf8");

// --- extract the inline explorer script (the last <script>...</script> block) ---
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).filter(s => s.trim());
const body = blocks[blocks.length - 1];
if (!body) { console.error("no inline explorer script found"); process.exit(2); }

// --- minimal crossfilter shim (only the surface the explorer uses) ---
function crossfilter(records) {
  const filters = [];
  const apply = () => records.filter(r => filters.every(f => f(r)));
  return {
    dimension(acc) {
      let pred = () => true;
      const dim = {
        _acc: acc,
        filterFunction(fn) { pred = v => fn(v); const i = filters.indexOf(dim._f); if (i >= 0) filters.splice(i, 1); dim._f = r => pred(acc(r)); filters.push(dim._f); return dim; },
        filterAll() { const i = filters.indexOf(dim._f); if (i >= 0) filters.splice(i, 1); dim._f = null; return dim; },
        top() { return apply().slice().sort((a, b) => 0); },
        group() {
          return { all() {
            const counts = new Map();
            for (const r of apply()) { const k = acc(r); counts.set(k, (counts.get(k) || 0) + 1); }
            return [...counts.entries()].map(([key, value]) => ({ key, value }));
          } };
        },
      };
      return dim;
    },
    groupAll() { return { value: () => apply().length }; },
  };
}

// --- d3 shim: a chainable selection that, crucially, INVOKES accessor functions over its bound
//     data (so a per-datum callback like r => r.pu.toFixed(1) actually runs on the null row; a hollow
//     no-op shim would let the test pass vacuously and give false confidence). ---
function sel(data) {
  const bound = Array.isArray(data) ? data : (data === undefined ? [] : [data]);
  const invoke = (fn) => { if (typeof fn === "function") bound.forEach((d, i) => fn(d, i)); };
  const s = {
    _data: bound,
    append: () => sel(bound), style: () => s, html: (v) => { invoke(v); return s; },
    // RECORD event handlers so the test can fire them AFTER the page has finished building —
    // which is when a user actually hovers or clicks. The hover card and click-to-inspect panel
    // read ~20 fields per record, and a null among them must not throw; a no-op `on` would let
    // this test pass while the real page broke on the first mouseover. Firing at bind time
    // instead would be unrealistic (it would run a handler before its own closure is live).
    on: (type, fn) => {
      if (typeof fn === "function" && /^(mousemove|mouseleave|click)$/.test(type)) {
        global.__pending.push({ type, fn, data: bound });
      }
      return s;
    },
    attr: (_k, v) => { invoke(v); return s; },     // run attr accessors (x(r.cl), r=>r.pu ...)
    text: (v) => { invoke(v); return s; },          // run text accessors (titles -> r.pu.toFixed)
    classed: (_c, v) => { invoke(v); return s; },
    selectAll: () => sel(bound), select: () => sel(bound),
    data: (d, _key) => sel(typeof d === "function" ? bound.flatMap(d) : d),
    join: (enter) => { if (typeof enter === "function") enter(sel(bound)); return s; },
    call: (fn) => { if (typeof fn === "function") fn(s); return s; },
    each: (fn) => { bound.forEach(function (d, i) { if (typeof fn === "function") fn.call(sel(d), d, i); }); return s; },
    // the hover card measures itself to stay on screen
    node: () => ({ getBoundingClientRect: () => ({ width: 240, height: 130 }) }),
    remove: () => s,
  };
  return s;
}
// A continuous scale that actually maps and inverts, so the brush path computes real ranges
// (a scale stubbed to 0 would let the histogram code pass vacuously).
function contScale() {
  let d = [0, 1], r = [0, 1];
  const f = (v) => r[0] + ((v - d[0]) / ((d[1] - d[0]) || 1)) * (r[1] - r[0]);
  f.domain = (v) => (v === undefined ? d : ((d = v), f));
  f.range = (v) => (v === undefined ? r : ((r = v), f));
  f.invert = (v) => d[0] + ((v - r[0]) / ((r[1] - r[0]) || 1)) * (d[1] - d[0]);
  f.nice = () => f;
  f.ticks = (n) => { const out = []; const k = n || 10; for (let i = 0; i <= k; i++) out.push(d[0] + (d[1] - d[0]) * i / k); return out; };
  return f;
}
const d3 = {
  select: () => sel(), selectAll: () => sel(),
  scaleLog: contScale, scaleLinear: contScale,
  scaleOrdinal: () => { const f = () => "#000"; f.domain = () => f; f.range = () => f; return f; },
  axisBottom: () => { const f = () => f; f.ticks = () => f; f.tickValues = () => f; f.tickFormat = () => f; f.tickSize = () => f; return f; },
  axisLeft: () => { const f = () => f; f.ticks = () => f; return f; },
  format: () => (v) => (v == null ? "null" : String(v)),
  range: (n) => [...Array(n).keys()],
  min: (arr, acc) => { const xs = arr.map(acc || (v => v)).filter(v => v != null); return xs.length ? Math.min(...xs) : undefined; },
  max: (arr, acc) => { const xs = arr.map(acc || (v => v)).filter(v => v != null); return xs.length ? Math.max(...xs) : undefined; },
  // bin(): buckets values into [x0,x1) arrays the way d3.bin does, enough for the histogram.
  bin: () => {
    let dom = [0, 1], th = [];
    const b = (vals) => {
      const edges = [dom[0], ...th.filter(t => t > dom[0] && t < dom[1]), dom[1]];
      const out = [];
      for (let i = 0; i < edges.length - 1; i++) {
        const x0 = edges[i], x1 = edges[i + 1];
        const arr = vals.filter(v => v >= x0 && (i === edges.length - 2 ? v <= x1 : v < x1));
        arr.x0 = x0; arr.x1 = x1;
        out.push(arr);
      }
      return out;
    };
    b.domain = (v) => ((dom = typeof v === "function" ? dom : v), b);
    b.thresholds = (v) => ((th = Array.isArray(v) ? v : th), b);
    return b;
  },
  // brushX(): records the handler and FIRES it, once with a real selection and once with null,
  // so both the "range applied" and "range cleared" code paths run against the null-bearing row.
  brushX: () => {
    const handlers = {};
    const br = (g) => {
      const ev = { selection: [8, 90] };
      if (handlers["brush end"]) handlers["brush end"](ev);
      if (handlers["brush end"]) handlers["brush end"]({ selection: null });
      return g;
    };
    br.extent = () => br;
    br.on = (name, fn) => { handlers[name] = fn; return br; };
    br.move = () => br;
    return br;
  },
};

// --- minimal document shim: #xf exists so the explorer mounts ---
const nodes = { xf: {} };
global.document = {
  getElementById: (id) => (id === "xf" ? nodes.xf : null),
  querySelectorAll: () => [],
  createElement: () => sel(),
  body: {},
};
global.window = { innerWidth: 1200, innerHeight: 900 };
global.d3 = d3;
global.crossfilter = crossfilter;
global.__pending = [];      // event handlers recorded during build, fired after it (see `on`)

try {
  // Guarantee a null-bearing row is present even if the fixture has none, by appending one to the
  // embedded RECORDS before executing: redefine RECORDS via a prelude.
  const nullRow = `;RECORDS.push({id:"__nulltest__",conc:999,tp:4,ni:4,series:"X",route:"r",pd:"colocated",` +
    `mtp:0,mnbt:8192,mns:256,a2a:"n/a",epw:null,pu:null,pu_is_step_estimate:true,cl:null,itl:null,` +
    `ttft:null,hit:null,to:0,drop:0,p95:null,ok:false,viol:["rejected"]});`;
  const patched = body.replace(/const RECORDS = (\[[\s\S]*?\]);/, (m, arr) => `const RECORDS = ${arr}${nullRow}`);
  new Function(patched)();
  // Now drive the interactions a user would: hover and click points and rows, including the
  // null-bearing one. The hover card and deployment panel must survive a record whose metrics,
  // engine knobs or recorded command are absent.
  //
  // SNAPSHOT the queue first and stop recording: a real handler calls redraw(), which re-binds
  // every handler and would append to the queue while we iterate it — unbounded growth. One
  // representative record per binding is enough to exercise each code path.
  const queued = global.__pending.slice();
  global.__pending = { push() {} };
  let fired = 0;
  for (const { type, fn, data } of queued) {
    for (const d of data.slice(0, 3)) {
      try { fn({ clientX: 40, clientY: 40, stopPropagation() {} }, d); fired++; }
      catch (e) { throw new Error(`'${type}' threw on a record: ${e.message}`); }
    }
  }
  console.log(`ok: explorer rendered with a null-bearing record and survived ${fired} hover/click events`);
  process.exit(0);
} catch (e) {
  console.error("explorer threw on a null-bearing record: " + (e && e.message));
  process.exit(1);
}
