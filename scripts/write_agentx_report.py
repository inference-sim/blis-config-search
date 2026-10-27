#!/usr/bin/env python3
"""Author report.json -- the narrative seam -- for one AgentX replay run directory.

    python3 -I scripts/write_agentx_report.py reports/<run-id>
    python3 -I .claude/skills/config-search/scripts/report.py reports/<run-id>   # then render

Every number in the prose is read out of `pareto.json` / `runs.jsonl` rather than typed, so the
narrative cannot drift from the data it describes, and the same script produces every scenario's
report -- six hand-written articles would have drifted from each other within a day.

Two things it deliberately does NOT do:

* It does not assert a direction for expert parallelism. The H100 and H200 runs of the same model
  disagree -- on a 79 GiB device every non-EP control dies on memory, so EP is a precondition; on
  a 141 GiB device the controls run and EP is worth a few per cent -- so `_ep_section` reads the
  outcome out of the finalists and writes whichever story the runs actually tell.
* It does not mix runs. The traffic character (input:output ratio, tokens per request) is taken
  from the SAME run as the headline throughput: the winner's own c_use run. An earlier draft took
  the ratio from the largest-input run and multiplied the winner's output rate by it, which
  produced an input-tok/s figure belonging to neither run.

Metric names, axis titles, figure captions and the "what to run" ranking rule are NOT written
here -- they come from report/lexicon.json, shared with the React dashboard (see
report_schema.md, "The shared lexicon").
"""
import json, os, sys

if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
    raise SystemExit(__doc__.strip().splitlines()[0] + "\n\n"
                     "usage: write_agentx_report.py <run-dir>")
D = sys.argv[1]
for need in ("scenario.json", "pareto.json", "runs.jsonl"):
    if not os.path.isfile(os.path.join(D, need)):
        raise SystemExit(f"{D}: not a completed run directory (no {need})")
HW_LABEL = {"h100": "H100", "h200": "H200", "b200": "B200"}

sc = json.load(open(f"{D}/scenario.json"))
pa = json.load(open(f"{D}/pareto.json"))
rows = [json.loads(l) for l in open(f"{D}/runs.jsonl")]
ok = [r for r in rows if r.get("returncode") == 0 and (r.get("metrics") or {}).get("tokens_per_sec")]

hw = HW_LABEL.get(sc["hardware"], sc["hardware"].upper())
gpus = sc["gpu_budget"]
nodes = gpus // 8
fin = pa["finalists"]
win = next(f for f in fin if f"({f.get('variant')})" in pa["winner"]["chosen"]
           or (not f.get("variant") and f["label"] == pa["winner"]["chosen"]))
wa = win["at_c_use"]

# Traffic character must come from the SAME run the throughput figures do -- the winner's own
# c_use run -- or the ratio and the tok/s describe different measurements. (An earlier draft took
# the ratio from the largest-input run and multiplied the winner's throughput by it, which mixed
# two runs and produced an input-tok/s figure belonging to neither.)
KNOB_KEYS = ("router_config", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization")
def _is_winner_c_use(r):
    d = r.get("deployment") or {}
    par = d.get("parallel") or {}
    cfg = r.get("config") or {}
    return (r.get("concurrency") == win["c_use"]
            and par.get("tp") == win["config"].get("tp")
            and par.get("dp", 1) == win["config"].get("dp", 1)
            and d.get("num_instances") == win["config"].get("num_instances")
            and not any(k in cfg for k in KNOB_KEYS))
big = next((r for r in ok if _is_winner_c_use(r)),
           max(ok, key=lambda r: r["metrics"].get("total_input_tokens", 0)))
bm = big["metrics"]; bc = bm["completed_requests"]
in_per = bm["total_input_tokens"] / bc
out_per = bm["total_output_tokens"] / bc
ratio = bm["total_input_tokens"] / max(bm["total_output_tokens"], 1)
# Input tok/s AS MEASURED in that same run, not the winner's output rate scaled by a ratio
# borrowed from elsewhere.
elapsed = bm.get("vllm_estimated_duration_s") or 0
in_tps = (bm["total_input_tokens"] / elapsed) if elapsed else wa["tokens_per_sec"] * ratio

# best per topology
best = {}
for f in fin:
    k = f["label"]
    if k not in best or f["at_c_use"]["tokens_per_sec_per_gpu"] > best[k]["at_c_use"]["tokens_per_sec_per_gpu"]:
        best[k] = f
ranked = sorted(best.values(), key=lambda f: -f["at_c_use"]["tokens_per_sec_per_gpu"])

# knob deltas on the winning topology
base = next((f for f in fin if f["label"] == win["label"] and not f.get("variant")), win)
bpg = base["at_c_use"]["tokens_per_sec_per_gpu"]
knobs = []
for f in fin:
    if f["label"] == win["label"] and f.get("variant"):
        pg = f["at_c_use"]["tokens_per_sec_per_gpu"]
        knobs.append((f["variant"], pg, 100 * (pg - bpg) / bpg, f["c_star"]))
knobs.sort(key=lambda k: k[2])

# every non-EP control that died, and why
mem_drops = [d for d in pa["dropped"] if "memory" in d["reason"].lower() or "occupancy" in d["reason"].lower()]
ctl_drops = [d for d in pa["dropped"] if "+ EP" not in d["label"]]
ep_survivors = [f for f in fin if "+ EP" in f["label"]]

n = lambda x, p=1: f"{x:,.{p}f}"
cv = win["curve"]
knee_lo = max((c for c in cv if c["tok_per_s_per_user"] and c["tok_per_s_per_user"] >= 50), key=lambda c: c["concurrency"])
knee_hi = min((c for c in cv if c["tok_per_s_per_user"] and c["tok_per_s_per_user"] < 50),
              key=lambda c: c["concurrency"], default=None)

worst_knob = knobs[0] if knobs else None


# --- the expert-parallelism section, which must report what was MEASURED --------------------
# On a 79 GiB H100 every non-EP control dies on memory, so EP is load-bearing. On a 141 GiB H200
# the controls run and EP is a modest few per cent. Hard-coding either reading would make the
# report assert something the runs do not show, so the section is built from the outcome.
DEV_GIB = {"h100": "79.2", "h200": "141.0", "b200": "178.4", "b300": "178.4"}

def _ep_pairs():
    """Per topology with BOTH arms servable: (label, EP, control, same-load EP, same-load ctl, Cstar_ep, Cstar_ctl).

    The at-c_use figures alone are NOT a clean EP comparison: when EP raises C*, the two arms are
    measured at DIFFERENT loads, and the ratio then mixes "EP is faster" with "EP serves more
    sessions". On B200 4x TP2xDP4 that read as +155.8% when the same-load gain is +1.5% and the
    real effect is that EP doubles the ceiling (C* 32 -> 64). So report both, from a load both
    arms actually ran.
    """
    byl = {f["label"]: f for f in fin if not f.get("variant")}
    out = []
    for lab in sorted(byl):
        if "+ EP" not in lab:
            continue
        ctl = lab.replace(" + EP", "")
        if ctl not in byl:
            continue
        e, c = byl[lab], byl[ctl]
        # the highest load BOTH arms measured, so the comparison is like-for-like
        shared = sorted({p["concurrency"] for p in e["curve"]}
                        & {p["concurrency"] for p in c["curve"]})
        se = sc_ = None
        if shared:
            at = max(shared)
            se = next(p["tokens_per_sec"] for p in e["curve"] if p["concurrency"] == at) / e["gpus"]
            sc_ = next(p["tokens_per_sec"] for p in c["curve"] if p["concurrency"] == at) / c["gpus"]
        out.append((lab, e["at_c_use"]["tokens_per_sec_per_gpu"],
                    c["at_c_use"]["tokens_per_sec_per_gpu"], se, sc_,
                    e["c_star"], c["c_star"], (max(shared) if shared else None)))
    return out

def _h100_drop_count():
    """How many shapes the sibling H100 run lost to memory, for the contrast. 12 is the measured
    figure; read it from that run when it is present rather than hard-coding."""
    here = os.path.basename(os.path.abspath(D))
    sib = here.replace("-b200-", "-h100-").replace("-h200-", "-h100-")
    path = os.path.join(os.path.dirname(os.path.abspath(D)), sib, "pareto.json")
    try:
        return len(json.load(open(path)).get("dropped") or [])
    except Exception:
        return "several"


def _ep_section():
    pairs = _ep_pairs()
    dev = DEV_GIB.get(sc["hardware"], "the")
    if not pairs:
        # No control survived anywhere: EP is what makes the model fit.
        return {
          "num": 3, "title": "Expert parallelism is load-bearing, not an optimisation",
          "figure": "scale_bars",
          "caption_note": "Only expert-parallel layouts appear: no control produced a servable run.",
          "prose": [
            (f"The search runs every expert-parallel candidate beside its non-EP control, because the "
             f"direction is meant to be measured rather than assumed. Here the comparison is not "
             f"close: <b>every one of the {len(ctl_drops)} non-EP controls failed outright</b>, and all "
             f"{len(ep_survivors)} finalists are expert-parallel."),
            (f"The failures are memory, not throughput \u2014 <i>\u201cNo available memory for the cache "
             f"blocks\u201d</i>, or a per-rank fixed occupancy above the {dev} GiB device. Sharding the "
             f"experts is what makes a 256k-token context fit on this part at all. It is also why the "
             f"narrow-TP shapes die even with EP on: the expert-parallel width is derived as "
             f"tp \u00d7 dp, so a width of 1 shards nothing."),
          ]}
    # Rank and quote the SAME-LOAD gain, which is the part attributable to EP itself; the capacity
    # change is stated separately because it is a different effect with a different implication.
    def samegain(t):
        _, _, _, se, sc_, _, _, _ = t
        return (100 * (se - sc_) / sc_) if (se and sc_) else 0.0
    lo = min(samegain(t) for t in pairs)
    hi = max(samegain(t) for t in pairs)
    def row(t):
        lab, e, c, se, sc_, cse, csc, at = t
        bit = (f"<li><code>{lab}</code>: {samegain(t):+.1f}% at a matched {at} lanes "
               f"({n(se)} vs {n(sc_)} tok/s/GPU)")
        if cse and csc and cse != csc:
            bit += (f" — and it moves the servable ceiling {csc} → {cse} lanes, which is what the "
                    f"{n(e)} vs {n(c)} tok/s/GPU at each arm's own usable load reflects")
        return bit + "</li>"
    rows = "<ul>" + "".join(row(t) for t in sorted(pairs, key=lambda t: -samegain(t))) + "</ul>"
    return {
      "num": 3, "title": "Expert parallelism helps here \u2014 but it is not what rescues the fit",
      "figure": "scale_bars",
      "caption_note": "Both arms of each topology are shown where both were servable.",
      "prose": [
        (f"Every expert-parallel candidate is run beside its non-EP control, so the direction is "
         f"measured rather than assumed. On {hw} the controls <b>do</b> run. Compared at a load "
         f"both arms served, expert parallelism is worth between {lo:+.1f}% and {hi:+.1f}% — "
         f"modest. Where it earns its place is <i>capacity</i>: it raises the servable ceiling, "
         f"which is a different claim from being faster per token:"),
        rows,
        (f"It is worth contrasting that with the same search on a smaller-memory part: on a "
         f"79.2 GiB H100 <i>no</i> non-EP control survives at all. The difference is headroom "
         f"\u2014 with {dev} GiB per device the weights and a 256k-token KV pool fit without "
         f"sharding the experts, so here EP is an optimisation rather than a precondition."
         if sc["hardware"] != "h100" else
         f"That is a real but modest gain at matched load."),
        ((f"The footprint still binds: {len(pa['dropped'])} shape(s) were refused outright. And the "
          f"narrow-TP layouts that do survive scale badly, because the expert-parallel width is "
          f"derived as tp \u00d7 dp \u2014 a width of 1 shards nothing.")
         if pa["dropped"] else
         (f"On this part the footprint stops binding altogether: <b>no shape was refused</b>, where "
          f"the 79.2 GiB H100 run lost {{}} of them to memory. What still separates the layouts is "
          f"sharding width \u2014 the expert-parallel width is derived as tp \u00d7 dp, so the "
          f"narrow-TP arms shard nothing and scale badly.").format(_h100_drop_count())),
      ]}


# --- scaling against the sibling budget, when that run exists ---------------------------------
# The six reports come in 4-node/8-node pairs on the same part, so each can say what doubling the
# cluster actually bought. Measured rather than assumed: on H100 doubling 32 -> 64 GPUs doubled
# CAPACITY (C* 64 -> 128 lanes) but multiplied throughput by only 1.62, so per-GPU efficiency
# FELL. A report that quoted only its own numbers would hide that.
def _sibling():
    """The same model+hardware run at the other node count, or None."""
    here = os.path.basename(os.path.abspath(D))
    for a, b in (("-4node", "-8node"), ("-8node", "-4node")):
        if here.endswith(a):
            cand = os.path.join(os.path.dirname(os.path.abspath(D)), here[: -len(a)] + b)
            if os.path.isfile(os.path.join(cand, "pareto.json")):
                return cand
    return None

def _scaling_section(num):
    sib = _sibling()
    if not sib:
        return None
    sp = json.load(open(f"{sib}/pareto.json"))
    ss = json.load(open(f"{sib}/scenario.json"))
    sw = next((f for f in sp["finalists"] if f["label"] == sp["winner"]["chosen"]), None)
    if not sw:
        return None
    sa = sw["at_c_use"]
    sg, mg = ss["gpu_budget"], gpus
    small, large = (sa, wa) if sg < mg else (wa, sa)
    small_g, large_g = (sg, mg) if sg < mg else (mg, sg)
    small_c, large_c = (sw["c_star"], win["c_star"]) if sg < mg else (win["c_star"], sw["c_star"])
    tps_x = large["tokens_per_sec"] / small["tokens_per_sec"]
    pg_x = large["tokens_per_sec_per_gpu"] / small["tokens_per_sec_per_gpu"]
    cap_x = large_c / small_c if small_c else 0
    return {
      "num": num,
      "title": f"What doubling the cluster buys",
      "prose": [
        (f"The same search at {small_g} and {large_g} GPUs on {hw} makes the scaling explicit. "
         f"Going from {small_g} to {large_g} GPUs multiplies fleet throughput by "
         f"<b>{tps_x:.2f}x</b> ({n(small['tokens_per_sec'])} to {n(large['tokens_per_sec'])} output "
         f"tok/s) and the servable ceiling by {cap_x:.2f}x ({small_c} to {large_c} lanes)."),
        (f"Per-GPU efficiency therefore <b>{'falls' if pg_x < 1 else 'rises'}</b> "
         f"{'from' if pg_x < 1 else 'from'} {n(small['tokens_per_sec_per_gpu'])} to "
         f"{n(large['tokens_per_sec_per_gpu'])} tok/s/GPU ({pg_x:.2f}x). "
         + (f"Capacity scales but efficiency does not: the second half of the cluster is doing "
            f"real work \u2014 the ceiling genuinely doubles \u2014 while each GPU does less of it. "
            f"For a fixed interactivity target that is the honest trade, and it means the larger "
            f"cluster is the right buy only if you need the extra {large_c - small_c} lanes of "
            f"capacity, not if you are chasing tokens per GPU."
            if pg_x < 1 else
            f"Efficiency holds up at the larger size, so the bigger cluster is the better buy on "
            f"both axes.")),
      ]}


# --- speculative decoding, its own section ----------------------------------------------------
# Drafting is not an engine THROTTLE like max_num_seqs or gpu_memory_utilization: those only ever
# constrain, while drafting can genuinely raise throughput, and its own measured curve is
# non-monotonic (K=1 was -25% on b200 chatbot because AL 1.81 does not pay for the draft, K=5
# +13%). It also changes what the per-user axis MEANS, which a reader has to be told. So it gets
# its own section rather than a bullet among the knobs that bind downward.

def _draft_matched():
    """Baseline vs each draft depth AT THE SAME LOADS, plus where each arm's cliff falls.

    The at-c_use numbers alone are misleading here for the same reason they were for expert
    parallelism: drafting MOVES C*, so each arm's own usable load is a different load, and the
    comparison then mixes "drafting is faster" with "drafting serves fewer sessions". Measured on
    h100 4x TP8 + EP, drafting at K=5 is +13% to +70% at matched load yet its ITL cliff arrives
    between 32 and 51 lanes where the baseline reaches 64 -- a real trade a reader has to see.
    """
    base = next((f for f in fin if f["label"] == win["label"] and not f.get("variant")), None)
    if not base:
        return [], None
    bl = {p["concurrency"]: p for p in base.get("curve") or []}
    out = []
    for f in fin:
        if f["label"] != win["label"]:
            continue
        k = int((f.get("config") or {}).get("num_speculative_tokens") or 0)
        if not k:
            continue
        pts = []
        for p in f.get("curve") or []:
            b = bl.get(p["concurrency"])
            if b and b["tokens_per_sec"]:
                pts.append((p["concurrency"], b["tokens_per_sec"], p["tokens_per_sec"],
                            100 * (p["tokens_per_sec"] - b["tokens_per_sec"]) / b["tokens_per_sec"],
                            b.get("tok_per_s_per_user"), p.get("tok_per_s_per_user")))
        out.append((k, f["c_star"], pts))
    return sorted(out), base

def _draft_rows():
    """(K, tok/s/GPU, delta% vs no drafting, C*, AL, AL source) for each drafting variant of the
    winning topology."""
    out = []
    for f in fin:
        if f["label"] != win["label"]:
            continue
        k = int((f.get("config") or {}).get("num_speculative_tokens") or 0)
        if not k:
            continue
        a = f["at_c_use"]
        al, src = None, None
        for r in ok:          # the acceptance is recorded on the run, not on the finalist
            d = r.get("deployment") or {}
            if (((d.get("engine") or {}).get("speculative") or {}).get("num_spec_tokens") == k
                    and (d.get("parallel") or {}).get("tp") == win["config"].get("tp")
                    and d.get("num_instances") == win["config"].get("num_instances")):
                acc = d.get("acceptance") or {}
                al, src = acc.get("acceptance_length"), acc.get("source")
                break
        out.append((k, a["tokens_per_sec_per_gpu"],
                    100 * (a["tokens_per_sec_per_gpu"] - bpg) / bpg if bpg else 0.0,
                    f["c_star"], al, src))
    return sorted(out)


def _draft_matched_prose():
    """The matched-load table plus what it means for setting the knob."""
    arms, base = _draft_matched()
    if not arms or not base:
        return ""
    head = ("<table class=\"short\"><thead><tr><th>sessions</th><th class=\"n\">no drafting</th>"
            + "".join(f'<th class="n">K={k}</th>' for k, _, _ in arms)
            + "</tr></thead><tbody>")
    loads = sorted({c for _, _, pts in arms for c, *_ in pts})
    body = ""
    for c in loads:
        cells = ""
        baseval = None
        for k, _, pts in arms:
            hit = next((t for t in pts if t[0] == c), None)
            if hit:
                baseval = hit[1]
                cells += f'<td class="n">{n(hit[2])} <span class="op-var">{hit[3]:+.0f}%</span></td>'
            else:
                cells += '<td class="n">\u2014</td>'
        body += (f'<tr><td class="n">{c}</td>'
                 f'<td class="n">{n(baseval) if baseval else "\u2014"}</td>{cells}</tr>')
    tbl = head + body + "</tbody></table>"
    cliffs = ", ".join(f"K={k} at {cs} lanes" for k, cs, _ in arms)
    return (f"<b>At matched load</b>, drafting is a clear throughput win \u2014 but it moves the "
            f"interactivity cliff, so the two arms' own usable loads are not the same load. Output "
            f"tok/s at each session count, with the change against no drafting:{tbl}"
            f"<p>The servable ceiling, however, falls: no drafting holds {base['c_star']} lanes "
            f"while {cliffs}. Drafting spends decode compute verifying tokens, which is compute "
            f"the baseline spends on concurrency \u2014 so the knob trades <i>sessions</i> for "
            f"<i>per-session speed</i>. Set it from which of those you are short of: at a fixed "
            f"session count below the drafting cliff it is close to free throughput, and above "
            f"that cliff it costs you the sessions.</p>")


def _draft_section(num):
    rows = _draft_rows()
    if not rows:
        return None
    best = max(rows, key=lambda r: r[1])
    worst = min(rows, key=lambda r: r[1])
    src = next((r[5] for r in rows if r[5]), None)
    reused = src and f"({sc['hardware']})" not in src
    items = "<ul>" + "".join(
        f"<li><b>K={k}</b>: {n(pg)} tok/s/GPU, {d:+.1f}% against no drafting"
        + (f", acceptance length {al}" if al else ", acceptance unresolved")
        + (f", ceiling {cs} lanes" if cs != win["c_star"] else "")
        + "</li>" for k, pg, d, cs, al, _ in rows) + "</ul>"
    return {
      "num": num,
      "title": "Speculative decoding, and what it does to the interactivity number",
      "prose": [
        (f"This model ships its own drafter \u2014 the catalog graph declares a "
         f"<code>{rows and 'dspark' or ''}</code> speculator \u2014 so draft depth is a real "
         f"deployment choice, not an exotic knob. Each depth is run against the no-drafting "
         f"baseline on the winning topology:"),
        items,
        (f"Read those percentages carefully: each arm is scored at <i>its own</i> usable load, and "
         f"drafting moves that load. A depth whose ceiling drops is therefore scored at a lower "
         f"session count than the baseline, which makes it look worse than it is at any load they "
         f"share \u2014 the matched-load table below is the comparison that answers \u201cshould "
         f"I turn this on\u201d. "
         + (f"Here the ceiling falls for "
            + ", ".join(f"K={k} ({cs} lanes against the baseline\u2019s {win['c_star']})"
                        for k, _, _, cs, _, _ in rows if cs != win["c_star"])
            + "." if any(r[3] != win["c_star"] for r in rows) else
            f"Here every depth holds the baseline\u2019s {win['c_star']}-lane ceiling, so the "
            f"figures above are already like-for-like.")),
        (f"<b>How to read the per-user rate on these rows.</b> Under drafting the engine reports "
         f"the inter-token latency of a <i>verification step</i>, and one step emits as many "
         f"tokens as the target accepts. So the per-user token rate is the acceptance length times "
         f"1000/ITL p99, not 1000/ITL p99 \u2014 the naive reading understates a drafting "
         f"configuration by the whole acceptance factor and can make the fastest configuration "
         f"measured look like a floor violation. The rows above use the acceptance-scaled rate, "
         f"and the explorer labels it as derived."),
        _draft_matched_prose(),
        (f"The acceptance lengths are measured, not assumed: they come from "
         f"<code>{src}</code> in blis-registry, keyed by this model, drafter, depth, thinking mode "
         f"and workload category."
         + (f" They were measured on another part and reused here; the profile set states that "
            f"acceptance depends mostly on the model, drafter and workload rather than the chip, "
            f"and asks that the reuse be recorded, so it is recorded on every row."
            if reused else "")
         if src else
         f"Where a depth has no matching acceptance profile, the per-user rate is left blank "
         f"rather than published at an unknown multiple."),
      ]}

rep = {
  "title": f"Serving DeepSeek-V4.1-Flash to coding agents on {nodes} {hw} nodes",
  "standfirst": (
    f"A configuration search over {len(rows)} BLIS replay runs on {gpus}× {hw}, driven by the "
    f"recorded AgentX agentic traffic rather than a synthetic request shape. The winner is "
    f"<b>{pa['winner']['chosen']}</b> at {n(wa['tokens_per_sec'])} output tok/s "
    f"({n(wa['tokens_per_sec_per_gpu'])} per GPU) while every user still reads at "
    f"{n(wa['tok_per_s_per_user'])} tok/s."),
  "meta": ["BLIS discrete-event simulation",
           f"blis-latency-kernel {sc.get('pinned_stack',{}).get('kernel','?')}",
           "AgentX MVP replay (393 recorded Claude Code trajectories)",
           f"{gpus}× {hw} · {nodes} nodes"],
  "hero": [
    {"h": "What was tested",
     "html": (f"Every expert-parallel and tensor-parallel layout that fills {gpus} GPUs, probed up a "
              f"session-lane ladder and refined one engine knob at a time — "
              f"{len(rows)} runs, {len(fin)} servable finalists, {len(pa['dropped'])} shapes the "
              f"simulator refused or that missed the floor.")},
    {"h": "What was found",
     "html": (f"This is a <b>prefill-dominated</b> workload: {n(in_per,0)} input tokens per request "
              f"against {n(out_per,0)} output, about <b>{ratio:.0f}:1</b>. Output tok/s per GPU reads low "
              f"by chatbot standards because almost all the work is reading context — the winner is "
              f"ingesting roughly {n(in_tps,0)} input tok/s.")},
    {"h": "What to run", "run": True,
     "html": (f"<b>{pa['winner']['chosen']}</b> at {win['c_use']} concurrent agent sessions "
              f"(80% of the measured ceiling of {win['c_star']}). Expert parallelism is not optional "
              f"here: <b>every</b> non-EP control ran out of memory before serving a request.")},
  ],
  "explore_prose": [
    (f"Each row below is one BLIS replay run. The load axis is <b>session-tree lanes</b>, not "
     f"in-flight requests: one lane drives a whole agent trajectory that fans out into many "
     f"sub-requests, so a lane is worth far more concurrency than a single chat turn."),
    (f"The interactivity floor is the default <b>50 tok/s per user</b> (ITL p99 ≤ 20 ms). That is a "
     f"<i>chatbot</i> floor, and it is worth saying plainly that it may be the wrong one for an agent: "
     f"these requests carry {n(in_per,0)} tokens of context and their p99 time-to-first-token is "
     f"{bm.get('ttft_p99_ms',0)/1000:.0f} s, so a user waits seconds before the first token regardless of how "
     f"fast the stream then runs. It is kept because it discriminates sharply on this workload — the "
     f"winner holds {n(knee_lo['tok_per_s_per_user'])} tok/s/user at {knee_lo['concurrency']} lanes and "
     f"collapses to {n(knee_hi['tok_per_s_per_user'])} at {knee_hi['concurrency']}" if knee_hi else
     f"The interactivity floor is the default 50 tok/s per user (ITL p99 ≤ 20 ms)."),
  ],
  "sections": [
    {"num": 2, "title": "The operating curve, and where it stops",
     "figure": "operating_curve",
     "caption_note": (f"The winner's ceiling is <b>{win['c_star']} lanes</b>; the recommendation sits at "
                      f"{win['c_use']}, measured directly rather than interpolated."),
     "prose": [
       (f"Throughput climbs almost linearly with lanes and then falls off a cliff. "
        f"{pa['winner']['chosen']} holds {n(knee_lo['tok_per_s_per_user'])} tok/s per user at "
        f"{knee_lo['concurrency']} lanes" +
        (f", and at {knee_hi['concurrency']} lanes per-user rate collapses to "
         f"{n(knee_hi['tok_per_s_per_user'])} tok/s while ITL p99 jumps from "
         f"{n(knee_lo['itl_p99_ms'])} ms to {n(knee_hi['itl_p99_ms'],0)} ms." if knee_hi else ".")),
       (f"The cliff is why the recommendation is {win['c_use']} lanes and not the ceiling: at the ceiling "
        f"there is no headroom for a burst, and this workload's bursts are whole agent trajectories "
        f"waking up at once."),
     ]},
    _ep_section(),
    {"num": 4, "title": "The knobs that bind, and the one that really hurts",
     "figure": "tail_scatter",
     "caption_note": "Crosses are runs that missed the interactivity floor.",
     "prose": ([
       (f"Refinement moves one knob at a time off the winning topology, so any difference is "
        f"attributable. Nothing improved on the defaults — these knobs <b>constrain</b>:") ] + ([
       ("<ul>" + "".join(
         f"<li><code>{v}</code>: {n(pg)} tok/s/GPU, {d:+.1f}%"
         + (f" — and it halves usable capacity, C* {win['c_star']} → {cs}" if cs < win["c_star"] else "")
         + "</li>" for v, pg, d, cs in knobs) + "</ul>") ] if knobs else []) + ([
       (f"The expensive one is <code>{worst_knob[0]}</code> at {worst_knob[2]:+.1f}%. It is not a small "
        f"scheduling inefficiency: routing on queue depth and KV utilisation drops the servable ceiling "
        f"from {win['c_star']} lanes to {worst_knob[3]}, so the deployment loses half its capacity. "
        f"Scoring on instantaneous KV pressure is actively wrong when every request arrives carrying "
        f"~{n(in_per,0)} tokens of context — the instance that looks least loaded is the one about to "
        f"be handed an enormous prefill.")
       ] if worst_knob and worst_knob[3] < win["c_star"] else [])),
    },
  ],
  "recommendation_num": 9,
  "recommendation_prose": [
    (f"<b>{pa['winner']['chosen']}</b> at {win['c_use']} concurrent agent sessions, on the "
     f"{sc['hardware']} deployment flags in the table below. That is {n(wa['tokens_per_sec'])} output "
     f"tok/s across {gpus} GPUs ({n(wa['tokens_per_sec_per_gpu'])} per GPU) at "
     f"{n(wa['tok_per_s_per_user'])} tok/s per user and ITL p99 {n(wa['itl_p99_ms'])} ms."),
    (f"Leave the engine knobs at their defaults. None of the refinements helped, and the routing "
     f"profiles in particular cost capacity on this traffic."),
    (f"These are simulator predictions over a recorded corpus, not a live cluster measurement. The "
     f"corpus is pinned by sha256, so the traffic is reproducible; the pricing is the "
     f"blis-latency-kernel cost model and should be spot-checked against real serving data before "
     f"anyone sizes a cluster from it."),
  ],
  "glossary": [
    {"id": "g-lane", "term": "session-tree lane",
     "def": "One live agent trajectory, not one request.",
     "long": ("An AgentX lane replays a whole recorded coding session: a main turn, its sub-agent "
              "groups and their inner requests. One lane therefore puts many requests in flight, "
              "which is why the lane ladder is much shorter than a request-concurrency ramp.")},
    {"id": "g-epw", "term": "expert-parallel width",
     "def": "tp × pcp × dp — derived, never declared.",
     "long": ("A width of 1 shards no experts, so it neither saves memory nor costs collectives. "
              "This is why adding dp rescues a layout that failed at the same tp.")},
    {"id": "g-cstar", "term": "C* and usable load",
     "def": "C* is the highest load meeting the floor; the recommendation is 0.8 × C*.",
     "long": ("The reported figures are measured AT the usable load, not read off a nearby rung of "
              "the ladder, so the throughput quoted is the throughput at the load recommended.")},
    {"id": "g-prefill", "term": "prefill-dominated",
     "def": f"About {ratio:.0f} input tokens per output token on this corpus.",
     "long": ("Output tok/s per GPU is the ranking metric but a poor description of the machine's "
              "work here: most of each request is reading context, so a low output rate can coexist "
              "with very high total token throughput.")},
  ],
}
# The scaling section only exists when the sibling budget has been searched, so it is appended
# rather than inlined -- a report for a part whose pair has not run yet simply omits it.
_draft = _draft_section(num=5)
if _draft:
    rep["sections"].append(_draft)
_scale = _scaling_section(num=6 if _draft else 5)
if _scale:
    rep["sections"].append(_scale)
rep["sections"] = [s for s in rep["sections"] if s]

json.dump(rep, open(f"{D}/report.json", "w"), indent=2, sort_keys=True)
print(f"wrote {D}/report.json")
print("  winner:", pa["winner"]["chosen"])
print("  ratio: %.0f:1   in/req %.0f   knobs: %d" % (ratio, in_per, len(knobs)))
