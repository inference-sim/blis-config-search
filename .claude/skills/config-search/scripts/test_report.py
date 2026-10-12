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
        self.sc, self.pa, self.runs, self.rep, self.nmal = report.load_run_dir(FIXTURE)
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
        # 17 servable, not 19: two 4xTP4 runs (conc 24, 25) have an aggregate ITL under 20 ms but a
        # single instance at ~55-58 ms, so the per-instance parity check (QA G4) correctly rejects
        # them where an aggregate-only check would have passed them.
        ok = [r for r in self.recs if r["ok"]]
        self.assertEqual(len(ok), 17)
        self.assertEqual(len(self.recs) - len(ok), 10)
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
        sc, pa, runs, _, _ = report.load_run_dir(FIXTURE)
        self.facts = report.build_facts(report.build_records(runs, sc), sc, pa)

    def test_counts(self):
        self.assertEqual(self.facts["n_points"], 27)
        self.assertEqual(self.facts["n_admissible"], 17)   # per-instance parity rejects 2 (QA G4)
        self.assertEqual(self.facts["n_configs"], 3)
        self.assertEqual(self.facts["winner"], "4x TP4 + EP")

    def test_peak_and_servable(self):
        self.assertEqual(self.facts["peak_cluster_conc"], 32)   # highest fleet tok/s is past the cliff
        # max servable drops to 20: conc 24/25 have an instance over the ITL cap even though the
        # cluster aggregate is under it, so per-instance parity (QA G4) excludes them.
        self.assertEqual(self.facts["max_servable_conc"], 20)


class Figures(unittest.TestCase):
    def setUp(self):
        sc, _, runs, _, _ = report.load_run_dir(FIXTURE)
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
        sc, _, _, _, _ = report.load_run_dir(FIXTURE)
        self.assertIn("not available", report.render_figure("no_such_figure", self.recs, sc))


class Narrative(unittest.TestCase):
    def test_full_render_has_all_sections(self):
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        self.assertTrue(rep, "fixture should ship a report.json")
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        for needle in ('class="standfirst"', 'class="panel"', 'id="xf"',
                       'class="gl-list"', 'class="short"', 'id="g-per-user-token-rate"'):
            self.assertIn(needle, html)
        self.assertNotIn("__BODY__", html)
        self.assertNotIn("__RECORDS__", html)
        self.assertEqual(html.count("<figure>"), 3)

    def test_absent_report_json_still_renders(self):
        sc, pa, runs, _, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, {}, "/abs/sample-run")
        self.assertIn('id="xf"', html)                   # explorer still mounts
        self.assertIn("What to run", html)               # auto operating table
        self.assertIn('class="short"', html)
        self.assertNotIn("__BODY__", html)

    def test_explorer_handles_empty_cross_filter(self):
        # The capacity probe visits a different concurrency ladder per topology, so many
        # conc x tp pairs have no run. The explorer must (a) ship an empty-state banner and
        # (b) dim/disable zero-count options so a cross-spec selection can't blank the plot.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
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
        sc, pa, runs, _, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, {"title": "X", "sections": [
            {"num": 2, "title": "only prose", "prose": ["hi"]}]}, "/x")
        self.assertIn("only prose", html)

    def test_records_and_facts_valid_json_in_output(self):
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        recs = json.loads(re.search(r"const RECORDS = (\[.*?\]);", html, re.S).group(1))
        facts = json.loads(re.search(r"const FACTS = (\{.*?\});", html, re.S).group(1))
        self.assertEqual(len(recs), 27)
        self.assertEqual(facts["n_configs"], 3)

    def test_only_two_external_refs(self):
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        refs = sorted(set(re.findall(r"https://[^\"']+", html)))
        self.assertEqual(refs, [
            "https://cdnjs.cloudflare.com/ajax/libs/crossfilter/1.3.12/crossfilter.min.js",
            "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"])

    def test_deterministic(self):
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        a = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        b = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertEqual(a, b)


class Contract(unittest.TestCase):
    """Seam-consistency on pareto.json (unchanged from the search track)."""
    def setUp(self):
        self.sc, self.pa, self.runs, _, _ = report.load_run_dir(FIXTURE)

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


class QaRegressions(unittest.TestCase):
    """Regression tests for the cross-vendor qa-review findings on PR #6 (G1-G8)."""

    def setUp(self):
        self.sc, self.pa, self.runs, self.rep, _ = report.load_run_dir(FIXTURE)

    # G1 — pu under speculative decoding
    def test_g1_pu_prefers_authoritative_field(self):
        run = {"metrics": {"tok_per_s_per_user": 90.0, "itl_mean_ms": 5.0}}
        pu, is_step = report._per_user_rate(run, mtp=3)
        self.assertEqual(pu, 90.0)
        self.assertFalse(is_step)

    def test_g1_pu_null_under_mtp_not_wrong_number(self):
        run = {"metrics": {"itl_mean_ms": 5.0}}            # no authoritative field
        pu, is_step = report._per_user_rate(run, mtp=3)
        self.assertIsNone(pu)                              # NOT the known-wrong step rate (1000/5)
        self.assertTrue(is_step)                           # flagged so the report can explain the gap
        pu0, is_step0 = report._per_user_rate(run, mtp=0)
        self.assertEqual(pu0, 200.0)                       # no MTP -> step == token, real rate
        self.assertFalse(is_step0)

    def test_g1_records_carry_the_flag(self):
        recs = report.build_records(self.runs, self.sc)
        self.assertTrue(all("pu_is_step_estimate" in r for r in recs))
        self.assertTrue(all(r["pu_is_step_estimate"] is False for r in recs))  # fixture is mtp=0

    # G2 — determinism depends on CONTENTS only: neither the path NOR the directory name may leak,
    # because the footer identifies the run by the in-data run_id (present in the fixture scenario).
    def test_g2_identical_contents_relocate_to_identical_html(self):
        self.assertTrue(self.sc.get("run_id"), "fixture scenario should carry a run_id")
        a = report.render_report(self.sc, self.pa, self.runs, self.rep, "/tmp/aaa/run-x")
        b = report.render_report(self.sc, self.pa, self.runs, self.rep, "/elsewhere/deep/DIFFERENT-name")
        self.assertEqual(a, b)                             # different path AND different basename
        self.assertNotIn("/tmp/aaa", a)
        self.assertNotIn("run-x", a)
        self.assertNotIn("DIFFERENT-name", b)

    def test_g2_no_run_id_still_location_independent(self):
        # Even without an in-data run_id, the output must NOT depend on the directory name: it says
        # "unidentified" rather than leaking the basename, so two differently-named dirs still match.
        sc = {k: v for k, v in self.sc.items() if k != "run_id"}
        a = report.render_report(sc, {}, self.runs, {}, "/x/anon-one")
        b = report.render_report(sc, {}, self.runs, {}, "/y/anon-two")
        self.assertEqual(a, b)
        self.assertNotIn("anon-one", a)
        self.assertNotIn("anon-two", b)
        self.assertIn("unidentified", a)

    # G3 — malformed ledger rows are counted + surfaced, not silently dropped
    def test_g3_malformed_rows_counted_and_surfaced(self):
        with tempfile.TemporaryDirectory() as d:
            shutil.copy(os.path.join(FIXTURE, "scenario.json"), os.path.join(d, "scenario.json"))
            with open(os.path.join(d, "runs.jsonl"), "w") as f:
                f.write('{"concurrency":1,"metrics":{"itl_mean_ms":5,"tokens_per_sec":10}}\n')
                f.write('{ this is not json\n')
                f.write('{"concurrency":2,"metrics":{"itl_mean_ms":6,"tokens_per_sec":20}}\n')
            sc, pa, runs, rep, nmal = report.load_run_dir(d)
            self.assertEqual(len(runs), 2)
            self.assertEqual(nmal, 1)
            html = report.render_report(sc, pa, runs, rep, d, nmal)
            self.assertIn("1 malformed line(s)", html)

    # G4 — feasibility parity: absent SLO metric is a violation, per-instance worst is checked
    def test_g4_absent_slo_metric_is_violation(self):
        ok, viol = report._feasible({"metrics": {"itl_p99_ms": 5}}, {}, {"ttft_p99_ms": 100})
        self.assertFalse(ok)
        self.assertTrue(any("ttft_p99_ms missing" in v for v in viol))

    def test_g4_aggregate_under_cap_cannot_mask_bad_instance(self):
        # aggregate itl 10 <= 20, but one instance is 50 > 20: must be flagged (the adjudicator's gap)
        run = {"metrics": {"itl_p99_ms": 10},
               "metrics_all_instances": [{"itl_p99_ms": 10}, {"itl_p99_ms": 50}]}
        ok, viol = report._feasible(run, {"itl_p99_ms_max": 20}, {})
        self.assertFalse(ok)
        self.assertTrue(any("50" in v for v in viol))

    def test_g4_per_instance_worst_used_when_aggregate_missing(self):
        run = {"metrics": {}, "metrics_all_instances": [{"itl_p99_ms": 50}, {"itl_p99_ms": 10}]}
        ok, viol = report._feasible(run, {"itl_p99_ms_max": 20}, {})
        self.assertFalse(ok)                               # worst instance 50 > 20

    # G7/F3 — a figure keyed on an unvaried knob degrades to a placeholder (really enforced)
    def test_g7_unvaried_knob_figure_placeholders(self):
        recs = report.build_records(self.runs, self.sc)
        figures.FIGURES["_qa_probe_route"] = {"builder": figures.fig_operating_curve,
                                              "keys": ("route",)}  # route is constant in the fixture
        try:
            svg = report.render_figure("_qa_probe_route", recs, self.sc)
            self.assertIn("was not varied", svg)
            # a figure keyed on conc (always swept) still draws real content
            self.assertIn("<path", report.render_figure("operating_curve", recs, self.sc))
        finally:
            del figures.FIGURES["_qa_probe_route"]

    def test_g7_registry_entries_declare_keys(self):
        for name, spec in figures.FIGURES.items():
            self.assertIn("builder", spec)
            self.assertIn("keys", spec)

    # G8 — a literal template marker in brain prose must not crash rendering
    def test_g8_placeholder_literal_in_prose_survives(self):
        rep = {"title": "T", "sections": [
            {"num": 2, "title": "x", "prose": ["a literal __BODY__ and __TITLE__ in text"]}]}
        html = report.render_report(self.sc, self.pa, self.runs, rep, "/x")
        self.assertIn("literal __BODY__ and __TITLE__ in text", html)

    # G5 — record-level edge cases
    def test_g5_zero_servable_scale_bars_degrade(self):
        recs = [{"series": "A", "conc": 4, "ok": False, "viol": ["x"], "cl": 10, "pu": 5, "p95": 1,
                 "itl": 5, "tp": 8, "ni": 1, "gpus": 8}]
        self.assertIn("no servable", figures.fig_scale_bars(recs))

    def test_g5_none_and_zero_metrics_do_not_crash(self):
        recs = [{"series": "A", "conc": c, "ok": True, "viol": [], "cl": cl, "pu": pu, "p95": p95,
                 "itl": itl, "tp": 8, "ni": 1, "gpus": 8}
                for c, cl, pu, p95, itl in [(1, None, None, None, None), (2, 0, 0, 0, 0),
                                            (4, 100, 50, 1000, 10)]]
        # each builder tolerates None/0 without raising
        for fn in (figures.fig_operating_curve, figures.fig_tail_scatter, figures.fig_scale_bars):
            svg = fn(recs)
            self.assertTrue(svg.strip().endswith("</svg>"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
