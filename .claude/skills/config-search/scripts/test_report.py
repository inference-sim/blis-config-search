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
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report
import figures
import lexicon

FIXTURE = os.path.normpath(os.path.join(HERE, "..", "examples", "sample-run"))
SMOKETEST_JS = os.path.join(HERE, "explorer_smoketest.js")


def _svg_wellformed(svg):
    return svg.count("<svg") == 1 and svg.count("</svg>") == 1 and svg.strip().endswith("</svg>")


class Records(unittest.TestCase):
    def setUp(self):
        self.sc, self.pa, self.runs, self.rep, self.nmal = report.load_run_dir(FIXTURE)
        self.recs = report.build_records(self.runs, self.sc)

    def test_one_record_per_run(self):
        self.assertEqual(len(self.recs), len(self.runs))
        self.assertEqual(len(self.recs), 27)

    def test_pu_on_p99_basis_matching_floor_and_artifact(self):
        # pu is on the p99 basis (1000/itl_p99 when no authoritative field), the SAME basis as the
        # interactivity floor and pareto.json's tok_per_s_per_user -- so the page never shows two
        # incompatible per-user numbers for a point (review F3). The fixture runs carry no
        # tok_per_s_per_user, so this exercises the p99 fallback.
        for r, run in zip(self.recs, self.runs):
            p99 = (run.get("metrics") or {}).get("itl_p99_ms")
            if p99:
                self.assertAlmostEqual(r["pu"], round(1000.0 / p99, 2), places=2)

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

    def test_tpg_is_a_first_class_field(self):
        # Output per GPU is the headline efficiency axis AND a brush axis in the explorer. It was
        # only derivable (cl / gpus), so the explorer could not filter on it. Derived from the
        # UNROUNDED throughput, not from the display-rounded cl, so it does not compound rounding.
        for r, run in zip(self.recs, self.runs):
            self.assertIn("tpg", r)
            raw = (run.get("metrics") or {}).get("tokens_per_sec")
            if raw is not None and r["gpus"]:
                self.assertAlmostEqual(r["tpg"], round(raw / r["gpus"], 2), places=2)
            else:
                self.assertIsNone(r["tpg"], "tpg must be null, never 0, when throughput is unknown")

    def test_latency_views_share_one_percentile_basis(self):
        # The explorer lets you switch the latency axis between ITL / TTFT / E2E / scheduling
        # delay. Those must all be the SAME statistic or the switch silently changes basis under
        # the reader: `itl` used to carry the MEAN while ttft/p95 carried p99/p95, so flipping the
        # axis compared a mean against a tail. p99 is the basis the interactivity floor uses.
        for r, run in zip(self.recs, self.runs):
            m = run.get("metrics") or {}
            if m.get("itl_p99_ms"):
                self.assertAlmostEqual(r["itl"], round(m["itl_p99_ms"], 3), places=3,
                                       msg="itl must be p99, matching the floor and pu")
            if m.get("scheduling_delay_p99_ms") is not None:
                self.assertAlmostEqual(r["sched"], m["scheduling_delay_p99_ms"], places=3)
            # pu is derived from the same p99 ITL, so the two can never disagree
            if r["itl"] and r["pu"] and not r["pu_is_step_estimate"]:
                self.assertAlmostEqual(r["pu"], round(1000.0 / r["itl"], 2), places=1)

    def test_itl_mean_kept_separately(self):
        # The mean is still useful (it is what BLIS reports most precisely), just not on the same
        # axis as the tails. Keep it under its own key rather than dropping the data.
        for r, run in zip(self.recs, self.runs):
            m = run.get("metrics") or {}
            if m.get("itl_mean_ms"):
                self.assertAlmostEqual(r["itl_mean"], round(m["itl_mean_ms"], 3), places=3)

    def test_series_label_is_pure_topology(self):
        # The series label is the thing you scan: GPUs x TP x EP x instances — what matters most
        # in LLM inference. Engine knobs do NOT belong in it. Cramming them in produced
        # "64x TP4xDP2 + EP · ll · b8k · seq256 · mem0.9" and invented abbreviations ("b2k") that
        # explain nothing. Knobs are filters and panel fields instead.
        def mk(tp, **cfg):
            return {"concurrency": 8, "config": {"tp": tp, **cfg},
                    "deployment": {"gpus": 64, "num_instances": 8,
                                   "parallel": {"tp": tp, "dp": 1, "pp": 1,
                                                "enable_expert_parallel": True},
                                   "engine": {"max_num_seqs": cfg.get("max_num_seqs", 256),
                                              "max_num_batched_tokens":
                                                  cfg.get("max_num_batched_tokens", 8192)}},
                    "metrics": {"tokens_per_sec": 1.0, "itl_p99_ms": 5.0,
                                "injected_requests": 1, "completed_requests": 1}}
        runs = [mk(8), mk(8, routing_policy="weighted"), mk(8, max_num_batched_tokens=2048)]
        recs = report.build_records(runs, self.sc)
        for r in recs:
            self.assertEqual(r["series"], "8x TP8 + EP",
                             f"series must be topology only, got {r['series']!r}")
            for junk in ("·", "b2k", "seq", "mem", "ll"):
                self.assertNotIn(junk, r["series"])

    def test_variant_names_the_knob_delta_readably(self):
        # Distinguishing two same-topology runs is the VARIANT's job, and it must spell the knob
        # out: "batch tokens 2,048", never "b2k".
        def mk(**cfg):
            return {"concurrency": 8, "config": {"tp": 8, **cfg},
                    "deployment": {"gpus": 64, "num_instances": 8,
                                   "parallel": {"tp": 8, "dp": 1, "pp": 1},
                                   "engine": {"max_num_seqs": cfg.get("max_num_seqs", 256),
                                              "max_num_batched_tokens":
                                                  cfg.get("max_num_batched_tokens", 8192)}},
                    "metrics": {"tokens_per_sec": 1.0, "itl_p99_ms": 5.0,
                                "injected_requests": 1, "completed_requests": 1}}
        recs = report.build_records(
            [mk(), mk(routing_policy="least-loaded"), mk(max_num_batched_tokens=2048),
             mk(max_num_seqs=64)], self.sc)
        variants = [r["variant"] for r in recs]
        self.assertEqual(variants[0], "", "the all-defaults run has no variant")
        self.assertEqual(variants[1], "least-loaded routing")
        self.assertEqual(variants[2], "batch tokens 2,048")
        self.assertEqual(variants[3], "max seqs 64")
        for v in variants:
            self.assertNotRegex(v, r"\bb\d+k\b|seq\d|mem0|^ll$", f"no abbreviations in {v!r}")

    def test_facets_are_tiered_primary_then_detail(self):
        # Progressive disclosure: the topology facets that matter most show first; the engine and
        # routing detail sits behind a disclosure so the panel does not open as a wall of knobs.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn("tier:'primary'", html)
        self.assertIn("tier:'detail'", html)
        self.assertIn("xf-more", html)          # the disclosure control
        # the headline knobs are primary; the engine limits are detail
        self.assertRegex(html, r"key:'tp',[^}]*tier:'primary'")
        self.assertRegex(html, r"key:'epw',[^}]*tier:'primary'")
        self.assertRegex(html, r"key:'mnbt',[^}]*tier:'detail'")
        self.assertRegex(html, r"key:'gmu',[^}]*tier:'detail'")

    def test_records_carry_routing_and_engine_knobs(self):
        # The knobs must reach the records or no facet, label or hover can show them.
        runs = [{"concurrency": 8, "config": {"tp": 8, "routing_policy": "weighted",
                                              "routing_scorers": "queue-depth:2"},
                 "deployment": {"gpus": 8, "num_instances": 1,
                                "parallel": {"tp": 8, "dp": 1, "pp": 1},
                                "engine": {"max_num_seqs": 64, "block_size": 16}},
                 "metrics": {"tokens_per_sec": 1.0, "itl_p99_ms": 5.0,
                             "injected_requests": 1, "completed_requests": 1}}]
        r = report.build_records(runs, self.sc)[0]
        self.assertEqual(r["route"], "weighted")
        self.assertEqual(r["scorers"], "queue-depth:2")
        self.assertEqual(r["mns"], 64)

    def test_figure_series_group_per_config_not_per_topology(self):
        # The operating-curve figure draws one line per configuration walking the load up. Keying
        # only on `series` (topology) put every knob variant on ONE line: a refined shape has a
        # run per variant at EVERY load, so the curve zigzagged between unrelated configs.
        base = {"tp": 4, "dp": 2, "ni": 32, "gpus": 256, "series": "32x TP4xDP2 + EP",
                "ok": True, "pu": 80.0, "cl": 1000.0}
        recs = [dict(base, conc=128, variant="", id="a"),
                dict(base, conc=128, variant="least-loaded routing", id="b"),
                dict(base, conc=256, variant="", id="c"),
                dict(base, conc=256, variant="least-loaded routing", id="d")]
        groups = figures._by_config(recs)
        self.assertEqual(len(groups), 2, f"one group per config, got {[g[0] for g in groups]}")
        for label, pts in groups:
            concs = [p["conc"] for p in pts]
            self.assertEqual(len(concs), len(set(concs)),
                             f"{label} repeats a load — that is the zigzag")
        # and the label must say which variant, or the legend shows the same name twice
        labels = [label for label, _ in groups]
        self.assertEqual(len(set(labels)), 2, f"group labels are distinct: {labels}")

    def test_operating_table_states_the_whole_deployment(self):
        # "What to run" is the table a reader copies flags from, so an omission is a WRONG
        # deployment, not just a terse one. It showed only --tensor-parallel-size and
        # --num-instances: no EP, no DP, no routing — the very knobs that distinguish two rows
        # whose topology matches.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        recs = report.build_records(runs, sc)
        html = report._operating_table({}, recs)
        self.assertIn("--enable-expert-parallel", html)      # the fixture's EP runs
        self.assertIn("--max-num-seqs", html)
        # the flag list must cover every knob the search can vary
        keys = {k for k, _ in report._FLAG_LABELS}
        for k in ("tp", "dp", "epw", "ni", "mnbt", "mns", "gmu", "bs", "route"):
            self.assertIn(k, keys, f"operating table omits {k}")
        # a knob at its default is omitted (the fixture is dp=1 throughout): printing
        # "--data-parallel-size 1" on every row would bury the settings that matter
        self.assertNotIn("--data-parallel-size", html)
        # ... but shown when it is not 1. Checked on the DERIVED path, which is what a reader
        # sees: _operating_table now derives its rows from the records whenever there are any
        # (see test_operating_table_is_derived_not_authored).
        dp_runs = [{"concurrency": 1, "config": {"tp": 4, "num_instances": 4},
                    "deployment": {"gpus": 8, "num_instances": 4,
                                   "parallel": {"tp": 4, "dp": 2, "enable_expert_parallel": True}},
                    "metrics": {"tokens_per_sec": 100.0, "itl_p99_ms": 9.0,
                                "injected_requests": 1, "completed_requests": 1}}]
        multi = report._operating_table({}, report.build_records(dp_runs, sc))
        self.assertIn("--data-parallel-size", multi)
        self.assertIn("--enable-expert-parallel", multi)

    def test_config_facet_exists_so_variants_are_filterable(self):
        # With knobs out of the label, several runs legitimately share a series name. There must be
        # a way to pick between them: a `variant` facet listing the knob deltas.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertRegex(html, r"key:'variant'")
        self.assertRegex(html, r"key:'series'")

    def test_records_carry_deployment_detail_for_the_panel(self):
        # The click-to-inspect panel reads the run's OWN recorded deployment, never a
        # reconstruction, so it cannot drift from what BLIS consumed.
        for r in self.recs:
            for k in ("pp", "nodes", "bs", "mml", "gmu", "cg", "scen", "cmd"):
                self.assertIn(k, r)
            self.assertEqual(r["nodes"], max(1, -(-(r["gpus"] or 0) // 8)) if r["gpus"] else 1)
            # a multi-node pool must name the fabric it prices collectives over
            if r["nodes"] > 1:
                self.assertTrue(r["fabric"], "multi-node run must report a fabric")
            else:
                self.assertIsNone(r["fabric"])

    def test_cmd_absent_in_older_runs_is_none_not_a_guess(self):
        # The committed fixture predates the `cmd` field. The panel must show "not recorded"
        # rather than invent a command that was never run.
        self.assertTrue(all(r["cmd"] is None for r in self.recs))

    def test_cmd_is_shell_quoted_from_the_recorded_argv(self):
        runs = [{"concurrency": 1, "cmd": ["/p/blis", "run", "--scenario", "a b.yaml"],
                 "deployment": {"gpus": 8, "num_instances": 1,
                                "parallel": {"tp": 8, "dp": 1, "pp": 1}, "engine": {}},
                 "metrics": {"tokens_per_sec": 1.0, "itl_p99_ms": 5.0,
                             "injected_requests": 1, "completed_requests": 1}}]
        r = report.build_records(runs, self.sc)[0]
        self.assertIn("'a b.yaml'", r["cmd"])      # a path with a space stays one argument

    def test_series_label_distinguishes_dp(self):
        # A label must identify its shape: "32x TP1" and "32x TP1xDP8" are different deployments
        # on different GPU counts, and a label that hides dp makes two rows of the explorer's
        # series facet indistinguishable. dp=1 keeps the short form (no gratuitous "xDP1").
        self.assertEqual(report._series_label(4, 4, {"tp": 4, "dp": 1,
                                                     "enable_expert_parallel": True}),
                         "4x TP4 + EP")
        self.assertEqual(report._series_label(1, 32, {"tp": 1, "dp": 8,
                                                      "enable_expert_parallel": True}),
                         "32x TP1xDP8 + EP")
        self.assertEqual(report._series_label(2, 32, {"tp": 2, "dp": 4,
                                                      "enable_expert_parallel": False}),
                         "32x TP2xDP4")
        # distinct shapes on the same instance count must not collide
        a = report._series_label(4, 32, {"tp": 4, "dp": 1, "enable_expert_parallel": False})
        b = report._series_label(4, 32, {"tp": 4, "dp": 2, "enable_expert_parallel": False})
        self.assertNotEqual(a, b)

    def test_moe_knobs_are_recorded(self):
        # dp and the derived expert-parallel width must survive into the records: without them the
        # explorer cannot show the axis the MoE search now explores.
        for r, run in zip(self.recs, self.runs):
            par = ((run.get("deployment") or {}).get("parallel") or {})
            self.assertEqual(r["dp"], par.get("dp", 1))
            if par.get("enable_expert_parallel"):
                # width is tp*pcp*dp (pcp unset = 1), mirroring blis-schemas
                self.assertEqual(r["epw"], par.get("tp", 1) * par.get("dp", 1))
            else:
                self.assertIsNone(r["epw"])


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
        self.assertIn("interactivity floor 50", svg)     # floor = the servability boundary (p99 basis)
        self.assertIn("4x TP4 + EP", svg)                # a series labelled

    def test_scale_bars(self):
        svg = figures.fig_scale_bars(self.recs)
        self.assertTrue(_svg_wellformed(svg))
        self.assertIn("4x TP4 + EP", svg)

    def test_tail_scatter(self):
        svg = figures.fig_tail_scatter(self.recs)
        self.assertTrue(_svg_wellformed(svg))
        low = svg.lower()
        self.assertTrue("tail" in low or "slowest" in low)   # axis title / aria-label present

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

    def test_provenance_strip_is_a_semantic_footer(self):
        # The provenance strip closes the document, so it is a <footer>, not an anonymous <div>:
        # the generated report and the React dashboard (<footer class="sitefoot">) agree on the
        # outline. The .foot class is kept, so report/template.html's styling is untouched.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn('<footer class="foot">', html)
        self.assertNotIn('<div class="foot">', html)
        self.assertIn(".foot{", html)                    # template rule still matches the element
        self.assertEqual(html.count('<footer'), html.count('</footer>'))

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

    def test_explorer_has_brushable_metric_histograms(self):
        # The reference report's explorer is TWO-part: "filter by outcome" (brush a range on the
        # continuous metrics) alongside "filter by setting" (click a knob level). We shipped only
        # the click facets, so an outcome-first question ("show me everything over 60 tok/s/user")
        # had no entry point.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn("Filter by outcome", html)
        self.assertIn("Filter by setting", html)
        self.assertIn("brushX", html)                      # d3 brush, not a bespoke drag
        self.assertIn("xf-hist", html)                      # the histogram row
        # every brush axis the reference uses, on our own field names
        for key in ("pu", "tpg", "ttft", "itl"):
            self.assertRegex(html, rf"key:\s*'{key}'", f"no histogram declared for {key}")

    def test_null_facet_levels_are_labelled(self):
        # epw is null for every EP-off run, which crossfilter groups as its own level. Rendering
        # the key verbatim gave a BLANK clickable row ("" with a count of 45) -- a filter you
        # cannot read. Null levels must carry an explicit label.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn("LEVEL_LABELS", html, "no null-level label table in the explorer")
        self.assertRegex(html, r"epw:\s*\{", "epw declares no level labels")

    def test_explorer_exposes_routing_and_engine_facets(self):
        # Routing swung throughput 2.3x on kimi-k2.5, and the batch/KV limits bind downward
        # (max_num_seqs 64 cost 29%). Every one of them must be filterable, and must auto-hide
        # when the search did not sweep it.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        for key in ("route", "scorers", "bs", "gmu", "schedp", "admit"):
            self.assertRegex(html, rf"key:\s*'{key}'", f"no facet declared for {key}")
        # the facets are grouped so deployment knobs read as a block, not mixed into topology
        self.assertIn("xf-grp", html)

    def test_hover_card_names_the_distinguishing_knobs(self):
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn("r.knobs", html)      # the hover reads the per-row knob map

    def test_explorer_exposes_moe_knob_facets(self):
        # EP width / dp / all-to-all must be declarable facets. They auto-hide when a run set does
        # not vary them (the existing varies() guard), which is why the assertion is on the dim
        # declarations rather than on rendered options.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        for key in ("epw", "dp", "a2a"):
            self.assertRegex(html, rf"key:\s*'{key}'", f"no facet declared for {key}")

    def test_explorer_has_a_latency_view_switcher(self):
        # The scatter's y-axis should answer more than one latency question. ITL, TTFT, E2E and
        # scheduling delay are four DISTINCT metrics (TPOT is a synonym for ITL, so it is labelled
        # on the ITL view rather than offered as a fifth).
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn("VIEWS", html)
        for key in ("pu", "itl", "ttft", "e2e", "sched"):
            self.assertRegex(html, rf"key:\s*'{key}'", f"no latency view for {key}")
        self.assertIn("xf-views", html)                 # the switcher UI
        self.assertIn("TPOT", html)                     # named as the synonym it is

    def test_explorer_has_a_hover_card_and_click_panel(self):
        # Hover = a readable card (not a bare SVG <title> dump); click = the exact deployment.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertIn("xf-tip", html)                   # styled hover card
        self.assertIn("click to see the deployment", html)   # the hover invites the click
        self.assertIn("xf-panel", html)                 # click-to-inspect panel
        self.assertIn("Reproduce this run", html)       # the panel shows the recorded command
        self.assertIn("Expert parallel", html)          # ... and the parallelism block
        self.assertIn("showPanel", html)                # wired to a click handler
        # the panel must read the recorded cmd, and say so plainly when a run has none
        self.assertIn("r.cmd", html)
        self.assertIn("was not recorded", html)

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

    def test_self_contained_no_external_resource_loads(self):
        # The report must be fully self-contained / offline: NO tag may load a resource over the
        # network. d3 + crossfilter are inlined, not linked. (URLs that merely appear as string
        # constants inside the inlined libs -- d3js.org, w3.org namespaces -- are not resource loads
        # and are fine; we assert on the loading tags, not on substrings.)
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        loads = re.findall(r'<(?:script|link|img|iframe|source|audio|video)\b[^>]*\b'
                           r'(?:src|href)\s*=\s*["\']https?://[^"\']+["\']', html, re.I)
        self.assertEqual(loads, [], f"report loads external resources: {loads}")
        self.assertNotIn("cdnjs", html)                       # no CDN dependency at all
        self.assertIn("vendored: d3.min.js", html)            # d3 is inlined
        self.assertIn("vendored: crossfilter.min.js", html)   # crossfilter is inlined

    def test_inlined_libs_are_byte_exact_pinned_versions(self):
        # The splice must preserve every byte of the vendored libs -- a mangled backslash or a
        # truncation at a "</script>" would silently corrupt the explorer. Assert the inlined block
        # bodies match the committed vendor files, and those match the pinned SHA-384 (same bytes the
        # report previously loaded from the CDN).
        import base64
        import hashlib
        pinned = {
            "d3.min.js": "su5kReKyYlIFrI62mbQRKXHzFobMa7BHp1cK6julLPbnYcCW9NIZKJiTODjLPeDh",
            "crossfilter.min.js": "qIMAO7hiswCdbLggCimcQKL2VYr4shEpIgnKP+9GL4YQPHJH4tQ/lLHsaLtDFulA",
        }
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        for name, want in pinned.items():
            with open(os.path.join(report.VENDOR_DIR, name), encoding="utf-8") as f:
                vendored = f.read()
            got = base64.b64encode(hashlib.sha384(vendored.encode()).digest()).decode()
            self.assertEqual(got, want, f"vendored {name} drifted from the pinned version")
            m = re.search(r"<!-- vendored: " + re.escape(name) + r" -->\n<script>\n(.*?)\n</script>",
                          html, re.S)
            self.assertIsNotNone(m, f"{name} not inlined in output")
            self.assertEqual(m.group(1), vendored, f"inlined {name} differs from the vendor file")

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
        run = {"metrics": {"tok_per_s_per_user": 90.0, "itl_p99_ms": 5.0}}
        pu, is_step = report._per_user_rate(run, mtp=3)
        self.assertEqual(pu, 90.0)
        self.assertFalse(is_step)

    def test_g1_pu_null_under_mtp_not_wrong_number(self):
        run = {"metrics": {"itl_p99_ms": 5.0}}             # no authoritative field
        pu, is_step = report._per_user_rate(run, mtp=3)
        self.assertIsNone(pu)                              # NOT the known-wrong step rate (1000/5)
        self.assertTrue(is_step)                           # flagged so the report can explain the gap
        pu0, is_step0 = report._per_user_rate(run, mtp=0)
        self.assertEqual(pu0, 200.0)                       # no MTP -> step == token, real p99 rate
        self.assertFalse(is_step0)

    # F3 — pu fallback is on the p99 basis (matches floor + artifact), not mean
    def test_f3_pu_fallback_uses_p99_not_mean(self):
        run = {"metrics": {"itl_p99_ms": 20.0, "itl_mean_ms": 10.0}}   # mean != p99
        pu, _ = report._per_user_rate(run, mtp=0)
        self.assertEqual(pu, 50.0)                         # 1000/p99=50, NOT 1000/mean=100

    # F3 (docs) — the user-facing description of `pu` must say p99, not mean, so the generated
    # report and skill contract don't misdescribe the plotted metric (namasl re-review).
    def test_f3_docs_describe_pu_as_p99(self):
        base = os.path.dirname(report.VENDOR_DIR)          # .../report
        skill = os.path.join(base, "..", "SKILL.md")
        schema = os.path.join(HERE, "report_schema.md")
        rjson = os.path.join(FIXTURE, "report.json")
        for path in (skill, schema, rjson):
            text = open(path, encoding="utf-8").read()
            lo = text.lower()
            self.assertNotIn("mean inter-token latency", lo, f"{path} still says mean ITL for pu")
            self.assertNotIn("1000 / mean itl", lo, f"{path} still says 1000/mean ITL for pu")
            self.assertNotIn("1000/mean-itl", lo, f"{path} still says 1000/mean-ITL for pu")
        # the fixture narrative + schema should affirmatively mention the p99 basis
        self.assertIn("p99", open(rjson, encoding="utf-8").read().lower())

    # F2 — a floor given only as tok_per_s_per_user still enforces (derived p99 cap)
    def test_f2_rate_only_floor_is_enforced(self):
        self.assertEqual(report._floor_itl_p99_max({"tok_per_s_per_user": 50}), 20.0)
        # a run at itl_p99 25 ms (> 20 cap implied by 50 tok/s) must be rejected
        ok, viol = report._feasible({"metrics": {"itl_p99_ms": 25}}, {"tok_per_s_per_user": 50}, {})
        self.assertFalse(ok)
        self.assertTrue(any("itl_p99_ms" in v for v in viol))

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

    # F4 — the BROWSER path: the explorer must render with a null-bearing record (not just the
    # Python SVG builders). Driven headlessly via explorer_smoketest.js with d3/crossfilter shims;
    # skipped where node is unavailable.
    def test_f4_explorer_tolerates_null_records_in_browser(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            html = os.path.join(d, "index.html")
            with open(html, "w") as f:
                f.write(report.render_report(sc, pa, runs, rep, d))
            r = subprocess.run([node, SMOKETEST_JS, html], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, f"explorer smoketest failed: {r.stderr or r.stdout}")

    # jgchn review — robustness
    def test_auto_operating_table_tolerates_none_conc(self):
        # a servable record with conc=None must not blow up sorted() in the auto table path
        runs = [{"concurrency": None, "metrics": {"itl_mean_ms": 5, "tokens_per_sec": 10,
                 "itl_p99_ms": 8, "injected_requests": 1, "completed_requests": 1}},
                {"concurrency": 4, "metrics": {"itl_mean_ms": 5, "tokens_per_sec": 40,
                 "itl_p99_ms": 8, "injected_requests": 1, "completed_requests": 1}}]
        html = report.render_report({"interactivity_floor": {"itl_p99_ms_max": 20}}, {}, runs, {}, "/x")
        self.assertTrue(html.strip().endswith("</html>"))

    def test_feasible_tolerates_non_numeric_metrics(self):
        # string-valued metrics must not raise TypeError in the comparisons
        run = {"metrics": {"injected_requests": "100", "completed_requests": "100",
                           "itl_p99_ms": "9.0"}}
        ok, viol = report._feasible(run, {"itl_p99_ms_max": 20}, {})
        self.assertTrue(ok)
        bad = {"metrics": {"injected_requests": "100", "completed_requests": "90"}}
        ok2, viol2 = report._feasible(bad, {}, {})
        self.assertFalse(ok2)
        self.assertTrue(any("incomplete" in v for v in viol2))

    # F5 — inline JSON must be safe inside a <script> element
    def test_f5_script_close_sequence_is_neutralized(self):
        runs = [{"concurrency": 1, "config": {"routing_policy": "</script><script>alert(1)</script>"},
                 "metrics": {"itl_p99_ms": 8, "tokens_per_sec": 10,
                             "injected_requests": 1, "completed_requests": 1}}]
        html = report.render_report({"interactivity_floor": {"itl_p99_ms_max": 20}}, {}, runs, {}, "/x")
        # the literal closing tag must NOT survive in the embedded data...
        self.assertNotIn("</script><script>alert", html)
        # ...it is present only as the inert unicode-escaped form
        self.assertIn("\\u003c/script", html)

    def test_f5_card_index_also_escaped(self):
        card = {"model": "m", "winner": "</script><script>x</script>", "href": "a/index.html",
                "hw": "8x h100", "workload": "chatbot", "summary": "", "run_id": "r"}
        html = report.render_card_index([card])
        self.assertNotIn("</script><script>x", html)

    # F1 — a present-but-malformed scenario/pareto must fail loudly, not silently unconstrain
    def test_f1_malformed_scenario_raises_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "runs.jsonl"), "w") as f:
                f.write('{"concurrency":1,"metrics":{"itl_p99_ms":8,"tokens_per_sec":10}}\n')
            with open(os.path.join(d, "scenario.json"), "w") as f:
                f.write('{ truncated not valid json')
            with self.assertRaises(report.MalformedInput):
                report.load_run_dir(d)

    def test_f1_missing_optional_file_still_defaults(self):
        # a genuinely ABSENT optional file must still default quietly (not raise)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "runs.jsonl"), "w") as f:
                f.write('{"concurrency":1,"metrics":{"itl_p99_ms":8,"tokens_per_sec":10}}\n')
            sc, pa, runs, rep, nmal = report.load_run_dir(d)   # no scenario/pareto/report files
            self.assertEqual((sc, pa, rep), ({}, {}, {}))
            self.assertEqual(len(runs), 1)



# ---------------------------------------------------------------------------
# Issue #13 -- cross-surface parity. config-search renders the same runs twice (this standalone
# report and the React dashboard in web/). The two had drifted into two vocabularies and, worse,
# two ranking objectives for "What to run", so at high load they recommended DIFFERENT configs
# for one scenario. These tests pin the seam: one lexicon file, one objective, one set of words.
class Parity(unittest.TestCase):
    WEB_MIRROR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "web", "src",
                                               "lexicon.json"))

    def test_web_mirror_is_byte_identical(self):
        # The dashboard cannot import out of the read-only skill dir, so web/src/lexicon.json is a
        # copy. A copy that drifts is worse than no sharing at all -- it looks shared and is not.
        if not os.path.exists(self.WEB_MIRROR):
            self.skipTest("web/ not present in this checkout")
        with open(lexicon.PATH, "rb") as a, open(self.WEB_MIRROR, "rb") as b:
            self.assertEqual(a.read(), b.read(),
                             "web/src/lexicon.json has drifted from report/lexicon.json; "
                             "copy the skill's canonical file over it")

    def test_operating_points_rank_on_the_stated_objective(self):
        # 7a: the table must rank on tok/s per GPU -- the objective the search itself optimises
        # and the recommendation card quotes -- not on per-user rate.
        self.assertEqual(lexicon.OBJECTIVE["key"], "tpg")
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        recs = report.build_records(runs, sc)
        pts = report.operating_points(recs)
        self.assertTrue(pts)
        by_conc = {}
        for r in recs:
            if r.get("ok") and r.get("conc") is not None and r.get("tpg") is not None:
                by_conc.setdefault(r["conc"], []).append(r)
        for p in pts:
            best = max(x["tpg"] for x in by_conc[p["conc"]])
            self.assertEqual(p["tpg"], best,
                             f"conc {p['conc']}: picked {p['tpg']} tok/s/GPU, best is {best}")

    def test_a_per_user_winner_does_not_win_the_table(self):
        # The exact shape of the disagreement: at one load, the fastest-per-user run is NOT the
        # most efficient. The table must take the efficient one (what the dashboard takes).
        runs = []
        for tpg_tps, pu, ni in ((2300.0, 131.0, 1), (6700.0, 83.0, 2)):
            runs.append({"concurrency": 512,
                         "deployment": {"gpus": 100, "num_instances": ni,
                                        "parallel": {"tp": 8, "enable_expert_parallel": True}},
                         "metrics": {"tokens_per_sec": tpg_tps, "itl_p99_ms": 1000.0 / pu,
                                     "tok_per_s_per_user": pu,
                                     "injected_requests": 1, "completed_requests": 1}})
        recs = report.build_records(runs, {"interactivity_floor": {"tok_per_s_per_user": 50}})
        pts = report.operating_points(recs)
        self.assertEqual(len(pts), 1)
        self.assertEqual(pts[0]["pu"], 83.0, "ranked by per-user rate instead of per-GPU")
        self.assertEqual(pts[0]["tpg"], 67.0)

    def test_the_table_is_derived_not_authored(self):
        # A hand-authored operating_points list carried its own rows AND its own metrics, which is
        # exactly the drift the issue is about: the glm fixture authors 3 of its 10 servable loads.
        # With records present the table is derived, so both surfaces show the same rows.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        self.assertTrue(rep.get("operating_points"), "fixture should still author a list")
        recs = report.build_records(runs, sc)
        html = report._operating_table(rep, recs)
        body = html[html.index("<tbody>"):]
        self.assertEqual(body.count("<tr>"), len(report.operating_points(recs)))

    def test_authored_points_still_render_without_a_ledger(self):
        # ... but a run dir with no usable ledger falls back to the authored table: a brain-written
        # table beats no table at all.
        rows = [{"conc": 20, "flags": {"tp": 4, "ni": 4}, "pu": 70.0, "cl": 1471.5, "itl": 14.3}]
        html = report._operating_table({"operating_points": rows}, [])
        self.assertIn("--tensor-parallel-size", html)
        self.assertEqual(html[html.index("<tbody>"):].count("<tr>"), 1)

    def test_objective_is_stated_in_the_what_to_run_section(self):
        # The acceptance criterion is "the same, STATED objective" -- not merely the same code.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        recs = report.build_records(runs, sc)
        html = report._what_to_run(rep, recs)
        self.assertIn(lexicon.OBJECTIVE["blurb"], html)
        self.assertIn(lexicon.OBJECTIVE["label"], html)

    def test_operating_pick_is_deterministic_under_ties(self):
        # Two equally efficient runs at one load must resolve the same way on both surfaces, so the
        # tiebreak is explicit rather than ledger-order dependent.
        def rec(rid):
            return {"id": rid, "ok": True, "conc": 8, "tpg": 50.0, "cl": 400.0, "pu": 80.0}
        a, b = rec("zzz"), rec("aaa")
        self.assertEqual(report.operating_points([a, b])[0]["id"], "aaa")
        self.assertEqual(report.operating_points([b, a])[0]["id"], "aaa")

    def test_figure_captions_come_from_the_lexicon(self):
        # 7b: the shared figures must carry the shared caption, so a reader moving between the two
        # surfaces sees one vocabulary.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        for name in ("operating_curve", "scale_bars", "tail_scatter"):
            cap = lexicon.figure(name, "caption")
            self.assertIn(report.esc(cap), html, f"{name}'s shared caption is not in the report")

    def test_figure_axis_titles_come_from_the_lexicon(self):
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        recs = report.build_records(runs, sc)
        svg = figures.fig_operating_curve(recs)
        self.assertIn(figures.esc(lexicon.AXES["pu"]), svg)
        self.assertIn(figures.esc(lexicon.AXES["cl_log"]), svg)
        tail = figures.fig_tail_scatter(recs)
        self.assertIn(figures.esc(lexicon.AXES["p95_log"]), tail)

    def test_captions_are_numbered_and_take_a_run_specific_note(self):
        # The shared sentence says what the axes mean; an optional per-run note says what THIS
        # run's numbers do. Both appear, and the figure number is assigned by the renderer.
        section = {"figure": "scale_bars", "caption_note": "Four-fold spread here."}
        cap = report._caption(section, 2)
        self.assertIn("Figure 2.", cap)
        self.assertIn("Four-fold spread here.", cap)
        self.assertIn(report.esc(lexicon.figure("scale_bars", "caption")), cap)
        # a figure with no lexicon entry keeps its authored caption verbatim
        self.assertEqual(report._caption({"figure": "nope", "caption": "<b>Custom.</b>"}, 1),
                         "<b>Custom.</b>")

    def test_lexicon_is_baked_into_the_page_for_the_explorer(self):
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        html = report.render_report(sc, pa, runs, rep, "/abs/sample-run")
        self.assertNotIn(report.LEXICON_PH, html, "the lexicon placeholder was not substituted")
        self.assertIn("const LEX =", html)
        # the explorer resolves its labels through the shared lookup, not literals
        self.assertIn("label:lab(d.key)", html)

    def test_explorer_panel_names_metrics_from_the_lexicon(self):
        # The report's click-to-inspect panel is the twin of the dashboard's PointDetail. Both now
        # name their measurement rows from the lexicon, so one number is never labelled two ways.
        tpl = open(os.path.join(report.REPORT_DIR, "template.html"), encoding="utf-8").read()
        for key in ("conc", "pu", "cl", "tpg", "itl", "ttft", "e2e", "sched"):
            self.assertIn(f"lab('{key}')", tpl, f"the panel/hover still hardcodes {key}'s label")

    # --- review follow-ups (PR #16) ---------------------------------------------------------

    def test_operating_numbers_use_the_lexicon_digits(self):
        # The table rendered its numbers through _num's own heuristic (>=1000 -> 0dp, else 1dp)
        # while the dashboard used the metric's lexicon `digits`, so a reader saw cl as 143.9 here
        # and 144 there -- the same 7b drift class this change closes, on the values themselves.
        sc, pa, runs, rep, _ = report.load_run_dir(FIXTURE)
        recs = report.build_records(runs, sc)
        html = report._operating_table(rep, recs)
        first = report.operating_points(recs)[0]
        for c in lexicon.OPERATING_COLUMNS[2:]:
            want = report._fnum(first.get(c), lexicon.metric(c, "digits"))
            self.assertIn(f'<td class="n">{want}</td>', html,
                          f"{c} is not formatted to its lexicon digits")
        # and specifically NOT the old heuristic, where the two disagree
        self.assertIn('<td class="n">144</td>', html)        # cl, digits 0 (was 143.9)
        self.assertNotIn('<td class="n">143.9</td>', html)

    def test_fnum_matches_the_dashboard_formatter(self):
        # _fnum must reproduce web/src/lib.ts `fnum`: at most `digits` decimals with trailing zeros
        # stripped, thousands separators, and ties rounded HALF AWAY FROM ZERO. The last part is
        # why an f-string will not do -- format() rounds half to EVEN, so f"{0.5:.0f}" is "0"
        # where JS gives "1".
        for value, digits, want in [(143.9, 0, "144"), (18.0, 1, "18"), (6.72, 2, "6.72"),
                                    (7476.19, 0, "7,476"), (0.5, 0, "1"), (1.5, 0, "2"),
                                    (2.5, 0, "3"), (0.15, 1, "0.2"), (1e6, 0, "1,000,000"),
                                    (0.0, 2, "0"), (None, 1, "–")]:
            self.assertEqual(report._fnum(value, digits), want,
                             f"_fnum({value!r}, {digits}) should be {want!r}")

    def test_no_servable_runs_says_so_on_both_surfaces(self):
        # The dashboard renders objective.none when nothing holds the floor; the report returned ""
        # and printed the "one row per load ... read the flags off it" blurb above empty space.
        runs = [{"concurrency": 8,
                 "deployment": {"gpus": 8, "num_instances": 1, "parallel": {"tp": 8}},
                 "metrics": {"tokens_per_sec": 100.0, "itl_p99_ms": 80.0,
                             "injected_requests": 1, "completed_requests": 1}}]
        recs = report.build_records(runs, {"interactivity_floor": {"tok_per_s_per_user": 50}})
        self.assertFalse(any(r["ok"] for r in recs), "fixture should have nothing servable")
        html = report._what_to_run({"recommendation_prose": ["Pick a config."]}, recs)
        self.assertIn(lexicon.OBJECTIVE["none"], html)
        # ... and does NOT promise a table that is not there
        self.assertNotIn(lexicon.OBJECTIVE["blurb"], html)
        self.assertIn("Pick a config.", html)          # the brain's own prose still shows

    def test_empty_case_wording_is_direction_neutral(self):
        # The run log sits BELOW this table on the dashboard and ABOVE it in the report (section 1
        # is the explorer), so the shared sentence must not say "below".
        self.assertNotIn("below", lexicon.OBJECTIVE["none"].lower())

    def test_modules_do_not_require_python_312(self):
        # A backslash inside an f-string expression is a syntax error before 3.12 (PEP 701 lifted
        # it). CI pins 3.12, so only a compile check catches a floor nobody asked for.
        import ast
        for name in ("report.py", "figures.py", "lexicon.py"):
            path = os.path.join(HERE, name)
            with open(path, encoding="utf-8") as f:
                src = f.read()
            ast.parse(src)                     # sanity: valid on the running interpreter
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.JoinedStr):
                    for part in node.values:
                        if isinstance(part, ast.FormattedValue):
                            expr = ast.get_source_segment(src, part.value) or ""
                            self.assertNotIn("\\", expr,
                                             f"{name}: backslash in an f-string expression "
                                             f"({expr[:60]!r}) forces a Python 3.12 floor")

    def test_flag_list_is_shared_with_the_dashboard(self):
        # A flag printed on one surface and not the other is a wrong deployment either way.
        self.assertEqual(report._FLAG_LABELS, lexicon.FLAG_LABELS)
        keys = {k for k, _ in report._FLAG_LABELS}
        for k in ("tp", "dp", "epw", "ni", "bs", "mnbt", "mns", "gmu", "mtp", "route", "pd"):
            self.assertIn(k, keys)

    def test_every_operating_column_has_a_label_and_a_definition(self):
        # The dashboard builds its column heads AND their tooltips from these entries, so a
        # missing one renders an unnamed column there.
        for key in lexicon.OPERATING_COLUMNS:
            self.assertTrue(lexicon.label(key), f"{key} has no label")
            src = lexicon.METRICS.get(key) or lexicon.FACETS[key]
            self.assertTrue(src.get("help"), f"{key} has no help text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
