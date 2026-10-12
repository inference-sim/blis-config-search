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
    append: () => sel(bound), style: () => s, html: () => s, on: () => s,
    attr: (_k, v) => { invoke(v); return s; },     // run attr accessors (x(r.cl), r=>r.pu ...)
    text: (v) => { invoke(v); return s; },          // run text accessors (titles -> r.pu.toFixed)
    classed: (_c, v) => { invoke(v); return s; },
    selectAll: () => sel(bound), select: () => sel(bound),
    data: (d, _key) => sel(typeof d === "function" ? bound.flatMap(d) : d),
    join: (enter) => { if (typeof enter === "function") enter(sel(bound)); return s; },
    call: (fn) => { if (typeof fn === "function") fn(s); return s; },
    each: (fn) => { bound.forEach(function (d, i) { if (typeof fn === "function") fn.call(sel(d), d, i); }); return s; },
    node: () => null, remove: () => s,
  };
  return s;
}
const d3 = {
  select: () => sel(), selectAll: () => sel(),
  scaleLog: () => { const f = () => 0; f.range = () => f; f.domain = () => f; return f; },
  scaleLinear: () => { const f = () => 0; f.range = () => f; f.domain = () => f; return f; },
  scaleOrdinal: () => { const f = () => "#000"; f.domain = () => f; f.range = () => f; return f; },
  axisBottom: () => { const f = () => f; f.ticks = () => f; f.tickValues = () => f; f.tickFormat = () => f; f.tickSize = () => f; return f; },
  axisLeft: () => { const f = () => f; f.ticks = () => f; return f; },
  format: () => (v) => (v == null ? "null" : String(v)),
  min: (arr, acc) => { const xs = arr.map(acc || (v => v)).filter(v => v != null); return xs.length ? Math.min(...xs) : undefined; },
  max: (arr, acc) => { const xs = arr.map(acc || (v => v)).filter(v => v != null); return xs.length ? Math.max(...xs) : undefined; },
};

// --- minimal document shim: #xf exists so the explorer mounts ---
const nodes = { xf: {} };
global.document = {
  getElementById: (id) => (id === "xf" ? nodes.xf : null),
  querySelectorAll: () => [],
  createElement: () => sel(),
};
global.window = { innerWidth: 1200 };
global.d3 = d3;
global.crossfilter = crossfilter;

try {
  // Guarantee a null-bearing row is present even if the fixture has none, by appending one to the
  // embedded RECORDS before executing: redefine RECORDS via a prelude.
  const nullRow = `;RECORDS.push({id:"__nulltest__",conc:999,tp:4,ni:4,series:"X",route:"r",pd:"colocated",` +
    `mtp:0,mnbt:8192,mns:256,a2a:"n/a",epw:null,pu:null,pu_is_step_estimate:true,cl:null,itl:null,` +
    `ttft:null,hit:null,to:0,drop:0,p95:null,ok:false,viol:["rejected"]});`;
  const patched = body.replace(/const RECORDS = (\[[\s\S]*?\]);/, (m, arr) => `const RECORDS = ${arr}${nullRow}`);
  new Function(patched)();
  console.log("ok: explorer rendered with a null-bearing record, no throw");
  process.exit(0);
} catch (e) {
  console.error("explorer threw on a null-bearing record: " + (e && e.message));
  process.exit(1);
}
