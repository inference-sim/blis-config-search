#!/usr/bin/env python3
# figures.py -- deterministic inline-SVG figure builders for the config-search report.
#
# Each builder takes the flat RECORDS list (see report.build_records) + options and returns an
# <svg> string, sized like the reference "glm53flash" report and themed entirely through the
# stylesheet's --fig-* CSS variables (so light/dark follows the page). No matplotlib, no numpy --
# just string assembly, so the output is byte-stable and the HTML stays self-rendering.
#
# A figure whose keying knob was not varied in this run (e.g. routing, when the v1 engine only
# sweeps concurrency + tp) returns a captioned placeholder instead of a misleading empty chart.
import math

VIEW_W, VIEW_H = 840, 470
PAD = {"t": 36, "r": 24, "b": 74, "l": 64}


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _plot_box():
    return (PAD["l"], PAD["t"], VIEW_W - PAD["r"], VIEW_H - PAD["b"])


def _lin(v, lo, hi, a, b):
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def _log(v, lo, hi, a, b):
    v, lo, hi = max(v, 1e-9), max(lo, 1e-9), max(hi, 1e-9)
    return _lin(math.log10(v), math.log10(lo), math.log10(hi), a, b)


def _nice_top(mx):
    if mx <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(mx))
    n = mx / p
    return (1 if n <= 1 else 2 if n <= 2 else 5 if n <= 10 else 10) * p


def _fmt(v):
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v/1000:.0f}k" if abs(v) >= 10000 else f"{v/1000:.1f}k"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}"


def _svg(inner, label):
    return (f'<svg viewBox="0 0 {VIEW_W} {VIEW_H}" width="100%" role="img" '
            f'aria-label="{esc(label)}" font-family="inherit">'
            f'<rect width="{VIEW_W}" height="{VIEW_H}" fill="var(--fig-bg,#fff)"/>'
            f'{inner}</svg>')


def placeholder(message):
    """A captioned 'nothing to plot' box. The caller supplies the specific reason (unknown figure
    name, or a keying knob that was not varied), so the message is shown verbatim with no fixed
    second line that could contradict it."""
    return (f'<svg viewBox="0 0 {VIEW_W} {VIEW_H}" width="100%" role="img" '
            f'aria-label="{esc(message)}" font-family="inherit">'
            f'<rect width="{VIEW_W}" height="{VIEW_H}" fill="var(--fig-bg,#fff)"/>'
            f'<text x="{VIEW_W/2}" y="{VIEW_H/2+4}" text-anchor="middle" font-size="14" '
            f'fill="var(--fig-ink3,#8a8a8a)">{esc(message)}</text></svg>')


def _grid_y(ticks, x0, x1, scale):
    out = []
    for t, yy in ticks:
        out.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" '
                   f'stroke="var(--fig-grid,#ececec)" stroke-width="1"/>')
        out.append(f'<text x="{x0-9:.1f}" y="{yy+3.5:.1f}" text-anchor="end" font-size="11" '
                   f'fill="var(--fig-ink2,#555)">{esc(t)}</text>')
    return "".join(out)


def _axis_titles(xt, yt):
    xx = PAD["l"] + (VIEW_W - PAD["r"] - PAD["l"]) / 2
    yy = PAD["t"] + (VIEW_H - PAD["b"] - PAD["t"]) / 2
    return (f'<text x="{xx:.1f}" y="{VIEW_H-24:.1f}" text-anchor="middle" font-size="12.5" '
            f'fill="var(--fig-ink2,#555)">{esc(xt)}</text>'
            f'<text transform="rotate(-90 18 {yy:.1f})" x="18" y="{yy:.1f}" text-anchor="middle" '
            f'font-size="12.5" fill="var(--fig-ink2,#555)">{esc(yt)}</text>')


# --- series grouping -------------------------------------------------------

def _config_key(r):
    """A stable label for the configuration a record belongs to (everything but load)."""
    return r.get("series") or f"TP{r.get('tp')} x{r.get('dp') or 1}"


def _by_config(records):
    groups = {}
    for r in records:
        groups.setdefault(_config_key(r), []).append(r)
    for pts in groups.values():
        pts.sort(key=lambda r: r.get("conc", 0))
    # deterministic order: by GPUs then label
    return sorted(groups.items(), key=lambda kv: (kv[1][0].get("gpus", 0), kv[0]))


_SERIES_COLORS = ["#1b4f8f", "#c2410c", "#1e5c33", "#7b3fb5", "#9a6a00", "#0e7490"]


# --- Figure 3: the operating curve (pu vs cl, one line per config) ---------

def fig_operating_curve(records, floor_pu=None):
    """Per-user token rate (y) against aggregate cluster output (x, log), a line per config.
    Filled markers are servable (ok), hollow ones break a limit; markers labelled with concurrency."""
    pts_all = [r for r in records if r.get("cl") and r.get("pu") is not None]
    if not pts_all:
        return placeholder("no runs with throughput + per-user rate to plot")
    x0, y0t, x1, y1b = _plot_box()
    xs = [r["cl"] for r in pts_all]
    xlo, xhi = min(xs) / 1.3, max(xs) * 1.3
    ytop = _nice_top(max(r["pu"] for r in pts_all) * 1.08)

    yticks = []
    for i in range(5):
        val = ytop * i / 4
        yticks.append((_fmt(val), _lin(val, 0, ytop, y1b, y0t)))
    inner = [_grid_y(yticks, x0, x1, None)]
    # x ticks (log) at nice decades/halves within range
    xticks = []
    d = 10 ** math.floor(math.log10(xlo))
    while d <= xhi:
        for m in (1, 2, 5):
            v = d * m
            if xlo <= v <= xhi:
                xx = _log(v, xlo, xhi, x0, x1)
                xticks.append(f'<line x1="{xx:.1f}" y1="{y0t}" x2="{xx:.1f}" y2="{y1b}" '
                              f'stroke="var(--fig-grid2,#f4f4f4)" stroke-width="1"/>'
                              f'<text x="{xx:.1f}" y="{y1b+18:.1f}" text-anchor="middle" '
                              f'font-size="11" fill="var(--fig-ink2,#555)">{_fmt(v)}</text>')
        d *= 10
    inner.append("".join(xticks))

    if floor_pu:
        # The y-axis is a MEAN-based per-user rate (1000/mean-ITL); servability (filled vs hollow) is
        # judged on the p99 ITL. Those are different statistics, so a marker can sit above this
        # reference line yet be hollow (p99 over the cap) or vice versa -- the line is a mean-rate
        # REFERENCE, not the pass/fail boundary. Label it as such so the two are not conflated.
        fy = _lin(floor_pu, 0, ytop, y1b, y0t)
        if y0t <= fy <= y1b:
            inner.append(f'<line x1="{x0}" y1="{fy:.1f}" x2="{x1}" y2="{fy:.1f}" '
                         f'stroke="var(--fig-muted,#b9c0c8)" stroke-width="1.3" '
                         f'stroke-dasharray="5 3"/>'
                         f'<text x="{x1-2:.1f}" y="{fy-5:.1f}" text-anchor="end" font-size="10.5" '
                         f'fill="var(--fig-ink3,#8a8a8a)">floor {_fmt(floor_pu)} (mean ref; '
                         f'servability is p99)</text>')

    for i, (label, pts) in enumerate(_by_config(records)):
        pp = [r for r in pts if r.get("cl") and r.get("pu") is not None]
        if not pp:
            continue
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        path = " ".join(("M" if j == 0 else "L") +
                        f"{_log(r['cl'],xlo,xhi,x0,x1):.1f},{_lin(r['pu'],0,ytop,y1b,y0t):.1f}"
                        for j, r in enumerate(pp))
        inner.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
        for r in pp:
            cx = _log(r["cl"], xlo, xhi, x0, x1)
            cy = _lin(r["pu"], 0, ytop, y1b, y0t)
            ok = r.get("ok", True)
            inner.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" '
                f'fill="{color if ok else "var(--fig-bg,#fff)"}" stroke="{color}" '
                f'stroke-width="1.6"{"" if ok else " stroke-dasharray=\"2.5 2\""}>'
                f'<title>{esc(label)}: {r.get("conc")} sessions, {_fmt(r["pu"])} tok/s/user, '
                f'{_fmt(r["cl"])} tok/s fleet{"" if ok else " (NOT servable)"}</title></circle>')
            inner.append(f'<text x="{cx+7:.1f}" y="{cy-6:.1f}" font-size="9.5" '
                         f'fill="var(--fig-ink3,#8a8a8a)">{r.get("conc")}</text>')
        # label the series at its last point
        last = pp[-1]
        lx = _log(last["cl"], xlo, xhi, x0, x1)
        ly = _lin(last["pu"], 0, ytop, y1b, y0t)
        inner.append(f'<text x="{lx+10:.1f}" y="{ly+4:.1f}" font-size="11" font-weight="600" '
                     f'fill="{color}">{esc(label)}</text>')

    inner.append(_axis_titles("aggregate output tok/s delivered by the cluster (log)",
                              "tok/s per user"))
    return _svg("".join(inner), "per-user token rate against aggregate cluster output per configuration")


# --- Figure 7: the tail-vs-rate scatter (pu vs p95) ------------------------

def fig_tail_scatter(records):
    """Per-user rate (y) against the slowest-95% latency (x, log). Crosses break a limit --
    the independent check that a fast-looking config isn't fast only because it dropped requests."""
    pts = [r for r in records if r.get("p95") and r.get("pu") is not None]
    if not pts:
        return placeholder("no runs with a p95 latency to plot")
    x0, y0t, x1, y1b = _plot_box()
    xs = [r["p95"] for r in pts]
    xlo, xhi = min(xs) / 1.3, max(xs) * 1.3
    ytop = _nice_top(max(r["pu"] for r in pts) * 1.08)

    yticks = [(_fmt(ytop * i / 4), _lin(ytop * i / 4, 0, ytop, y1b, y0t)) for i in range(5)]
    inner = [_grid_y(yticks, x0, x1, None)]
    d = 10 ** math.floor(math.log10(xlo))
    while d <= xhi:
        for m in (1, 2, 5):
            v = d * m
            if xlo <= v <= xhi:
                xx = _log(v, xlo, xhi, x0, x1)
                inner.append(f'<line x1="{xx:.1f}" y1="{y0t}" x2="{xx:.1f}" y2="{y1b}" '
                             f'stroke="var(--fig-grid2,#f4f4f4)" stroke-width="1"/>'
                             f'<text x="{xx:.1f}" y="{y1b+18:.1f}" text-anchor="middle" font-size="11" '
                             f'fill="var(--fig-ink2,#555)">{_fmt(v/1000)}s</text>')
        d *= 10

    conc_colors = {}
    palette = ["#08306b", "#1a5296", "#2e74b5", "#5a9bd4", "#8fbce0", "#bcd7ec", "#dceaf6"]
    for i, c in enumerate(sorted({r.get("conc") for r in pts})):
        conc_colors[c] = palette[i % len(palette)]

    for r in pts:
        cx = _log(r["p95"], xlo, xhi, x0, x1)
        cy = _lin(r["pu"], 0, ytop, y1b, y0t)
        ok = r.get("ok", True)
        tip = (f'{r.get("conc")} sessions, {_fmt(r["pu"])} tok/s/user, '
               f'p95 {_fmt(r["p95"]/1000)}s{"" if ok else " (NOT servable)"}')
        if ok:
            inner.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" '
                         f'fill="{conc_colors[r.get("conc")]}" opacity="0.9">'
                         f'<title>{esc(tip)}</title></circle>')
        else:
            inner.append(f'<g stroke="var(--bad,#b2182b)" stroke-width="1.8">'
                         f'<line x1="{cx-4:.1f}" y1="{cy-4:.1f}" x2="{cx+4:.1f}" y2="{cy+4:.1f}"/>'
                         f'<line x1="{cx-4:.1f}" y1="{cy+4:.1f}" x2="{cx+4:.1f}" y2="{cy-4:.1f}"/>'
                         f'<title>{esc(tip)}</title></g>')

    inner.append(_axis_titles("slowest 95% of session latencies (log)", "tok/s per user"))
    return _svg("".join(inner), "per-user token rate against tail latency, crosses break a limit")


# --- Figure 6: how far each configuration scales (max servable concurrency) --

def fig_scale_bars(records):
    """One horizontal bar per config: the highest concurrency that stays inside every limit."""
    groups = _by_config(records)
    rows = []
    for label, pts in groups:
        servable = [r.get("conc", 0) for r in pts if r.get("ok", True)]
        broke = [r for r in pts if not r.get("ok", True)]
        maxc = max(servable) if servable else 0
        first_break = min((r.get("conc", 0) for r in broke), default=None)
        viol = next((r.get("viol") or [] for r in broke if r.get("conc") == first_break), [])
        rows.append((label, maxc, first_break, viol))
    if not rows or all(m == 0 for _, m, _, _ in rows):
        return placeholder("no servable runs to size")
    x0, y0t, x1, y1b = _plot_box()
    # The x-axis must span the largest servable concurrency AND the furthest first-break marker.
    # Collect both into one list so the axis fits every mark; guard against an all-zero/empty set
    # (e.g. when no run broke a limit, so every fb is None) rather than calling max() on a bare int.
    extents = [m for _, m, _, _ in rows] + [fb for _, _, fb, _ in rows if fb]
    maxv = max([v for v in extents if v] or [1])
    top = _nice_top(maxv * 1.1)
    inner = []
    # x gridlines
    for i in range(5):
        v = top * i / 4
        xx = _lin(v, 0, top, x0, x1)
        inner.append(f'<line x1="{xx:.1f}" y1="{y0t}" x2="{xx:.1f}" y2="{y1b}" '
                     f'stroke="var(--fig-grid,#ececec)" stroke-width="1"/>'
                     f'<text x="{xx:.1f}" y="{y1b+18:.1f}" text-anchor="middle" font-size="11" '
                     f'fill="var(--fig-ink2,#555)">{_fmt(v)}</text>')
    n = len(rows)
    band = (y1b - y0t) / n
    for i, (label, maxc, fb, viol) in enumerate(rows):
        cy = y0t + band * (i + 0.5)
        bw = _lin(maxc, 0, top, x0, x1)
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        inner.append(f'<rect x="{x0:.1f}" y="{cy-9:.1f}" width="{max(bw-x0,0):.1f}" height="18" '
                     f'rx="2" fill="{color}" opacity="0.82"><title>{esc(label)}: serves up to '
                     f'{maxc} sessions</title></rect>')
        inner.append(f'<text x="{x0+6:.1f}" y="{cy+4:.1f}" font-size="11.5" font-weight="600" '
                     f'fill="#fff">{esc(label)}</text>')
        inner.append(f'<text x="{bw+8:.1f}" y="{cy+4:.1f}" font-size="11" '
                     f'fill="var(--fig-ink2,#555)">{maxc}</text>')
        if fb is not None:
            fx = _lin(fb, 0, top, x0, x1)
            reason = (viol[0].split("'")[1] if viol and "'" in viol[0] else "limit")
            inner.append(f'<g stroke="var(--bad,#b2182b)" stroke-width="1.8">'
                         f'<line x1="{fx-4:.1f}" y1="{cy-4:.1f}" x2="{fx+4:.1f}" y2="{cy+4:.1f}"/>'
                         f'<line x1="{fx-4:.1f}" y1="{cy+4:.1f}" x2="{fx+4:.1f}" y2="{cy-4:.1f}"/>'
                         f'<title>first break at {fb}: {esc(reason)}</title></g>')
    inner.append(_axis_titles("highest concurrency served inside every limit", ""))
    return _svg("".join(inner), "maximum servable concurrency per configuration")


# Figure registry. Each entry declares the builder plus `keys`: the RECORDS field(s) the figure's
# story depends on. report.render_figure() checks that EVERY keyed field actually varies across the
# runs (more than one distinct value) and otherwise returns a captioned placeholder -- so a figure
# whose knob the search never swept degrades honestly instead of drawing a misleading flat chart.
# A figure keyed only on "conc" is always drawable (concurrency is the primary load axis, always
# swept); a hypothetical routing figure would set keys=("route",) and placeholder until routing varies.
FIGURES = {
    "operating_curve": {"builder": fig_operating_curve, "keys": ("conc",)},
    "tail_scatter":    {"builder": fig_tail_scatter,    "keys": ("conc",)},
    "scale_bars":      {"builder": fig_scale_bars,      "keys": ("series",)},
    # Examples ready for when the search varies these knobs (they placeholder until then):
    # "routing_feasibility": {"builder": fig_route_heatmap, "keys": ("route",)},
    # "spec_decode_trade":   {"builder": fig_spec_trade,   "keys": ("mtp",)},
}


def varies(records, key):
    """True iff `key` takes more than one distinct (non-None) value across records."""
    seen = {r.get(key) for r in records if r.get(key) is not None}
    return len(seen) > 1
