#!/usr/bin/env python3
"""Unit tests for the config-search report step. No network / no BLIS binary needed.

    python3 scripts/test_report.py           # or: python3 -m unittest -v  (from this dir)

Groups, all against the committed examples/sample-run/ fixture:
  * Records   -- runs.jsonl -> flat RECORDS mapping + feasibility classification;
  * Facts     -- the precomputed FACTS the narrative references;
  * Figures   -- each inline-SVG builder: well-formed SVG, real content, graceful degrade;
  * Narrative -- report.json tolerance (present / absent / partial) + render invariants;
  * Contract  -- seam-consistency checks on pareto.json (at_c_use==curve[c_use], c_use==round(h*c*)).
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report
import figures

FIXTURE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "sample-run"))


def _svg_wellformed(svg):
    return svg.count("<svg") == 1 and svg.count("</svg>") == 1 and svg.strip().endswith("</svg>")


class Records(unittest.TestCase):
    def setUp(self):
        self.sc, self.pa, self.runs, self.rep = report.load_run_dir(FIXTURE)
        self.recs = report.build_records(self.runs, self.sc)

    def test_one_record_per_run(self):
        self.assertEqual(len(self.recs), len(self.runs))
        self.assertEqual(len(self.recs), 27)

    def test_pu_is_per_token_from_mean_itl(self):
        # pu is derived from the raw itl_mean_ms (not the display-rounded r["itl"]).
        for r, run in zip(self.recs, self.runs):
            raw = (run.get("metrics") or {}).get("itl_mean_ms")
            if raw:
                self.assertAlmostEqual(r["pu"], round(1000.0 / raw, 2), places=2)

    def test_per_gpu_series_from_deployment(self):
        # 4x TP4 => gpus 16, read from deployment not the flat config
        r = next(r for r in self.recs if r["series"] == "4x TP4 + EP")
        self.assertEqual(r["gpus"], 16)
        self.assertEqual(r["tp"], 4)
        self.assertEqual(r["ni"], 4)

    def test_feasibility_split(self):
        ok = [r for r in self.recs if r["ok"]]
        self.assertEqual(len(ok), 19)
        self.assertEqual(len(self.recs) - len(ok), 8)
        # an unservable row carries a reason
        bad = next(r for r in self.recs if not r["ok"])
        self.assertTrue(bad["viol"])

    def test_itl_cliff_marks_unservable(self):
        # a run with itl_p99 well over 20 ms must be flagged not-ok
        over = [r for r, run in zip(self.recs, self.runs)
                if (run.get("metrics") or {}).get("itl_p99_ms", 0) > 20]
        self.assertTrue(over and all(not r["ok"] for r in over))


class Facts(unittest.TestCase):
    def setUp(self):
        sc, pa, runs, _ = report.load_run_dir(FIXTURE)
        self.facts = report.build_facts(report.build_records(runs, sc), sc, pa)

    def test_counts(self):
        self.assertEqual(self.facts["n_points"], 27)
        self.assertEqual(self.facts["n_admissible"], 19)
        self.assertEqual(self.facts["n_configs"], 3)
        self.assertEqual(self.facts["winner"], "4x TP4 + EP")

    def test_peak_and_servable(self):
        self.assertEqual(self.facts["peak_cluster_conc"], 32)   # highest fleet tok/s is past the cliff
        self.assertEqual(self.facts["max_servable_conc"], 25)   # but max servable is the 4xTP4 C*


class Figures(unittest.TestCase):
    def setUp(self):
        sc, _, runs, _ = report.load_run_dir(FIXTURE)
        self.recs = report.build_records(runs, sc)

    def test_operating_curve(self):
        svg = figures.fig_operating_curve(self.recs, floor_pu=50)
        self.assertTrue(_svg_wellformed(svg))
        self.assertIn("tok/s per user", svg)
        self.assertIn("interactivity floor", svg)        # floor guide drawn
        self.assertIn("4x TP4 + EP", svg)                # a series labelled

    def test_scale_bars(self):
        svg = figures.fig_scale_bars(self.recs)
        self.assertTrue(_svg_wellformed(svg))
        self.assertIn("4x TP4 + EP", svg)

    def test_tail_scatter(self):
        svg = figures.fig_tail_scatter(self.recs)
        self.assertTrue(_svg_wellformed(svg))
        self.assertIn("tail", svg.lower() + "slowest")   # axis title present

    def test_placeholder_is_wellformed(self):
        svg = figures.placeholder("routing was not varied")
        self.assertTrue(_svg_wellformed(svg))
        self.assertIn("not varied", svg)

    def test_empty_records_degrade(self):
        self.assertIn("no runs", figures.fig_operating_curve([]))

    def test_render_figure_unknown_name(self):
        sc, _, _, _ = report.load_run_dir(FIXTURE)
        self.assertIn("not available", report.render_figure("no_such_figure", self.recs, sc))


class Narrative(unittest.TestCase):
    def test_full_render_has_all_sections(self):
        sc, pa, runs, rep = report.load_run_dir(FIXTURE)
        self.assertTrue(rep, "fixture should ship a report.json")
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        for needle in ('class="standfirst"', 'class="panel"', 'id="xf"',
                       'class="gl-list"', 'class="short"', 'id="g-per-user-token-rate"'):
            self.assertIn(needle, html)
        self.assertNotIn("__BODY__", html)
        self.assertNotIn("__RECORDS__", html)
        self.assertEqual(html.count("<figure>"), 3)

    def test_absent_report_json_still_renders(self):
        sc, pa, runs, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, {}, "/abs/sample-run")
        self.assertIn('id="xf"', html)                   # explorer still mounts
        self.assertIn("What to run", html)               # auto operating table
        self.assertIn('class="short"', html)
        self.assertNotIn("__BODY__", html)

    def test_explorer_handles_empty_cross_filter(self):
        # The capacity probe visits a different concurrency ladder per topology, so many
        # conc x tp pairs have no run. The explorer must (a) ship an empty-state banner and
        # (b) dim/disable zero-count options so a cross-spec selection can't blank the plot.
        sc, pa, runs, rep = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn("No runs match this combination", html)   # empty-state banner text
        self.assertIn(".xf-opt.empty", html)                    # disabled-option style
        self.assertIn("classed('empty'", html)                  # redraw dims zero-count options
        # the data really is sparse across conc x tp (otherwise the guard is untested)
        recs = report.build_records(runs, sc)
        concs = {r["conc"] for r in recs}
        tps = {r["tp"] for r in recs}
        pairs = {(r["conc"], r["tp"]) for r in recs}
        self.assertLess(len(pairs), len(concs) * len(tps),
                        "fixture should have empty conc x tp cells to exercise the guard")

    def test_partial_report_json_no_crash(self):
        sc, pa, runs, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, {"title": "X", "sections": [
            {"num": 2, "title": "only prose", "prose": ["hi"]}]}, "/x")
        self.assertIn("only prose", html)

    def test_records_and_facts_valid_json_in_output(self):
        sc, pa, runs, rep = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        recs = json.loads(re.search(r"const RECORDS = (\[.*?\]);", html, re.S).group(1))
        facts = json.loads(re.search(r"const FACTS = (\{.*?\});", html, re.S).group(1))
        self.assertEqual(len(recs), 27)
        self.assertEqual(facts["n_configs"], 3)

    def test_only_two_external_refs(self):
        sc, pa, runs, rep = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        refs = sorted(set(re.findall(r"https://[^\"']+", html)))
        self.assertEqual(refs, [
            "https://cdnjs.cloudflare.com/ajax/libs/crossfilter/1.3.12/crossfilter.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"])

    def test_deterministic(self):
        sc, pa, runs, rep = report.load_run_dir(FIXTURE)
        a = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        b = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertEqual(a, b)


class Contract(unittest.TestCase):
    """Seam-consistency on pareto.json (unchanged from the search track)."""
    def setUp(self):
        self.sc, self.pa, self.runs, _ = report.load_run_dir(FIXTURE)

    def test_at_c_use_matches_curve(self):
        for f in self.pa["finalists"]:
            pt = next((p for p in f["curve"] if p["concurrency"] == f["c_use"]), None)
            self.assertIsNotNone(pt)
            for k in ("tokens_per_sec", "itl_p99_ms", "tok_per_s_per_user"):
                self.assertEqual(f["at_c_use"][k], pt[k])

    def test_c_use_is_headroom_times_c_star(self):
        h = self.pa.get("headroom", self.sc.get("headroom"))
        for f in self.pa["finalists"]:
            self.assertEqual(f["c_use"], round(h * f["c_star"]))

    def test_finalist_topologies_in_runs(self):
        shapes = {(r.get("deployment", {}).get("parallel", {}).get("tp"),
                   r.get("deployment", {}).get("num_instances")) for r in self.runs}
        for f in self.pa["finalists"]:
            self.assertIn((f["config"]["tp"], f["config"]["num_instances"]), shapes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
