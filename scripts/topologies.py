#!/usr/bin/env python3
"""Candidate per-instance topologies for a GPU budget — the ONE enumeration both search paths use.

`run_search.py` (the deterministic sweep behind the Submit button) and the agentic search in
`.claude/skills/config-search/SKILL.md` both source their candidate shapes here, so the automated
path cannot silently explore less than the skill advertises. It did: the MoE expert-parallel arm
was documented as a searched axis but never enumerated, so every MoE report recommended the weaker
shape (deepseek-v4-pro on 256x H200: TP8+EP measured 1923.7 tok/s against the reported TP8's
1556.3, and 21 of 28 runs failed for a reason EP/dp fixes).

The rules below are BLIS's, verified against the pinned v0.10.0 binary, not guesses:

- **Dense models take neither dp nor EP.** `dp=2` on qwen3-14b is refused outright ("states dp 2
  for a dense model; DP placement is modelled for MoE models"), and expert parallelism shards
  nothing when there are no experts.
- **The expert-parallel width is derived, not declared:** `tp x pcp x dp`
  (blis-schemas spec/deployment/deployment.go). With pcp unset that is `tp x dp`, so EP over a
  width of 1 is a no-op the schema itself warns about — we skip those rather than burn a run.
- **A width of 1 is also what breaks big MoE models.** `tp=4` alone on deepseek-v4-pro dies with
  "191.2 GiB of per-rank fixed occupancy on a 141.0 GiB device (tp=4, expert-parallel width 1)",
  while `tp=4,dp=2` runs. The missing axis was `dp` as much as `enable_expert_parallel`.
- **One instance must fit one node** (`tp*dp <= GPUS_PER_NODE`); scale out with `num_instances`.

`all2all_backend` is deliberately NOT in the default grid. The kernel does price it (allgather
1127.7 vs DeepEP high-throughput 1105.0 tok/s on deepseek-v4-pro) but 2% does not earn a doubling
of the run count against EP's 24%. The report carries it as a facet, so a search that overrides
the grid to sweep it displays correctly.
"""

GPUS_PER_NODE = 8          # a per-instance footprint beyond one node needs a fabric (see run_blis)
TP_TIERS = (1, 2, 4, 8, 16, 32, 64)    # powers of two; past 8 the instance spans nodes
DP_TIERS = (1, 2, 4, 8)
# COST NOTE (measured on deepseek-v4.1-flash / AgentX replay / 32 GPUs, 2026-10-26). The narrow-tp
# arms are the expensive half of a search and have never placed: tp<=2 took 39% of total BLIS wall
# time on H100 and 30% on H200, and scored 1.8-5.9 tok/s/GPU against the winners' 21-31. The cause
# is structural -- expert-parallel width is derived as tp x dp, so a wide-dp/narrow-tp layout both
# shards poorly AND makes the simulation expensive (8 DP ranks step in lockstep).
#
# They are kept anyway, deliberately: "narrow TP collapses, and here is the curve" is a FINDING the
# reports state, and dropping the arm would leave a lone survivor looking like the whole search
# space (the thing `dropped[]` exists to prevent). If a campaign is wall-clock bound, bound the
# grid by raising the floor here rather than by truncating max_candidates, and say in the report
# that the grid was bounded.


def is_moe(model_entry):
    """True only when the catalog says so.

    `arch` is derived from the model's graph.yaml layer_kinds by build_catalog.py, which is
    authoritative. A missing entry (catalog miss) is treated as dense: enumerating dp/EP for a
    model that turns out to be dense means BLIS rejects every one of those runs, and a candidate
    list that fails is worse than a narrow one that works.
    """
    return (model_entry or {}).get("arch") == "MoE"


def expert_parallel_width(tp, dp):
    """The width BLIS derives for a layout: tp x pcp x dp, with pcp unset (1) in this runner.

    Mirrors Parallelism.ExpertParallelWidth() in blis-schemas. A width of 1 shards no experts.
    """
    return max(1, tp) * max(1, dp)


def candidate_topologies(gpu_budget, model_entry=None, max_candidates=12,
                         max_footprint=GPUS_PER_NODE):
    """Per-instance shapes worth probing for `gpu_budget` GPUs, best-first and deterministic.

    Each candidate is a dict of run_blis config keys: `tp`, `num_instances`, and for MoE also `dp`
    and `enable_expert_parallel`. Every candidate fills the whole budget
    (`tp * dp * num_instances == gpu_budget`), so a 256-GPU budget really explores 256 GPUs.

    `max_footprint` bounds ONE instance's GPU count. It defaults to a single node, because that is
    what a model which fits should use -- multi-node instances pay a fabric term for collectives.
    Raise it for a model whose weights do not fit in one node at any tp (kimi-k3 needs 332.3 GiB
    per rank at tp=8 on a 141 GiB H200 and only fits from tp=32 up); past one node a footprint
    must fill whole nodes, which run_blis also enforces.

    Ordering is by footprint (widest first — fewest, largest instances run fastest), then the
    expert-parallel arm, then widest `tp`. That matters because the list is truncated: the previous
    enumeration sorted by -tp and capped at 8, which would have cut every EP variant off the end
    even once EP was generated. Measured throughput on deepseek-v4-pro is monotonic in tp within a
    footprint (TP8 2375 > TP4xDP2 1987 > TP2xDP4 1456 > TP1xDP8 991, all +EP), so widest-tp-first
    puts the likeliest winner in front of any truncation.
    """
    budget = int(gpu_budget)
    if budget < 1:
        return []
    moe = is_moe(model_entry)
    dp_tiers = DP_TIERS if moe else (1,)

    cap = max(1, int(max_footprint))
    out = []
    for tp in TP_TIERS:
        for dp in dp_tiers:
            footprint = tp * dp
            if footprint > cap or footprint > budget:
                continue
            # Past one node the instance must fill whole nodes: a straddling footprint leaves the
            # remainder of a node idle and misprices the cross-node term (run_blis refuses it).
            if footprint > GPUS_PER_NODE and footprint % GPUS_PER_NODE:
                continue
            if budget % footprint:
                continue                      # a partially-used node is not a shape we recommend
            ni = budget // footprint
            # EP arm first (see docstring); skipped when it would shard nothing.
            ep_arms = [True, False] if moe and expert_parallel_width(tp, dp) > 1 else [False]
            for ep in ep_arms:
                cand = {"tp": tp, "num_instances": ni}
                if moe:
                    cand["dp"] = dp
                    cand["enable_expert_parallel"] = ep
                out.append(cand)

    # Widest footprint first; EP before non-EP within a footprint; widest tp last (see docstring).
    # Pure key sort on a deterministic list -> identical output for identical input.
    out.sort(key=lambda c: (-(c["tp"] * c.get("dp", 1)),
                            not c.get("enable_expert_parallel", False),
                            -c["tp"]))
    return out[:max_candidates]


# --- engine / routing knobs -------------------------------------------------------------------
#
# These are swept as a COORDINATE refinement on the best topologies, not crossed with every shape:
# a full cross product is 4 routing x 3 batch x 3 seqs x 12 topologies x 12 loads = 5184 runs of
# mostly identical results, where one-knob-at-a-time finds the same binding lever in ~1/9th.
#
# Measured on kimi-k2.5 / 512x H200 / multidoc at the winner shape (c=1024, 2048 requests):
#
#   routing_policy      round-robin 24349 tok/s | least-loaded ~same | weighted -2% (TTFT +29%)
#                       always-busiest 5104 tok/s (-58%), ITL 103 ms, TTFT 4637 ms  <- EXCLUDED
#   max_num_seqs        256 (default) 24349 | 64 -> 17335 (-29%) | 16 -> 8827 (-64%)
#   max_num_batched_tokens  8192 24349 | 32768 ~same | 2048 -> ITL 91 ms (+47%)
#   gpu_memory_utilization  0.9 24349 | 0.95 ~same | 0.5 -> 17846 (-27%), 1270 preemptions
#
# The lesson: these knobs bind DOWNWARD. Raising them past the default changed nothing measurable,
# so the grid probes the constrained side (where a real deployment is often forced to sit) rather
# than spending runs on headroom that does not exist.
# ROUTING ARMS MOVED. inference-sim v0.12.0 removed --routing-policy (and every other routing /
# admission flag): routing is the embedded llm-d router now, configured by an
# EndpointPickerConfig that the scenario's `router:` block names. So a routing arm is a FILE --
# see routing/README.md for each arm and the former flag it maps from. Upstream dropped
# round-robin / least-loaded / always-busiest outright ("the llm-d router has no such policy"),
# so the +84% `least-loaded` result from the kimi-k2.5 campaign has no exact equivalent here and
# that comparison cannot be reproduced arm-for-arm.
#
# Measured on deepseek-v4.1-flash / 4x TP8xDP4 / h100 / chatbot at 64 sessions: the router's
# default, `queue`, `load-aware`, `queue-kv` and `prefix-affinity` are all byte-identical
# (5333.7 tok/s), and `random` differs only slightly (5318.8). Uniform synthetic requests over
# four equal instances leave no imbalance for a scorer to exploit, so the arms are kept for the
# agentic replay -- whose trajectories vary hugely in size and re-send a growing shared prefix --
# rather than being presented as a lever this workload has shown.
ROUTING_ARMS = ("routing/prefix-affinity.yaml", "routing/queue-kv.yaml")

KNOB_TIERS = (
    # one entry per knob: (key, values to probe besides the default)
    ("router_config", ROUTING_ARMS),                    # the router's own default IS the baseline
    ("max_num_seqs", (64,)),                            # the batch-width constraint that binds
    ("max_num_batched_tokens", (2048,)),                # the prefill-admission constraint
    ("gpu_memory_utilization", (0.7,)),                 # a KV-starved deployment
)


def knob_variants(topology):
    """One-knob-at-a-time refinements of `topology`, in stable order.

    Each returned candidate is the topology plus EXACTLY ONE engine/routing knob, so a result that
    differs is attributable to that knob. The topology itself is never altered — these refine a
    shape the search has already found servable.
    """
    out = []
    for key, values in KNOB_TIERS:
        for v in values:
            out.append({**topology, key: v})
    return out


def label(cand):
    """Topology label for a candidate ("32x TP4xDP2 + EP").

    Deliberately topology-ONLY: instances x TP x DP x EP is what dominates an LLM serving
    decision and what a reader scans down a column. Engine and routing knobs are described
    separately by knob_phrase() — squeezing them in here produced
    `64x TP4xDP2 + EP · ll · b8k · seq256 · mem0.9`, unreadable and full of invented shorthand.
    """
    dp = cand.get("dp", 1)
    shape = f"TP{cand['tp']}" + (f"xDP{dp}" if dp > 1 else "")
    ep = " + EP" if cand.get("enable_expert_parallel") else ""
    return f"{cand['num_instances']}x {shape}{ep}"


# Readable knob names, spelled out on purpose: "b2k" for max_num_batched_tokens=2048 tells a
# reader nothing. Mirrors report.KNOB_NAMES, keyed by candidate-config key.
KNOB_NAMES = {"router_config": "routing", "routing_policy": "routing",
              "routing_scorers": "scorers",
              "max_num_batched_tokens": "batch tokens", "max_num_seqs": "max seqs",
              "block_size": "block size", "gpu_memory_utilization": "gpu mem util",
              "scheduler": "scheduler", "admission_policy": "admission"}


def routing_arm_name(path):
    """The readable name of a routing arm, from its EPP config path.

    "routing/prefix-affinity.yaml" -> "prefix-affinity". The file basename IS the arm's name (see
    routing/README.md), so a reader sees "prefix-affinity routing" rather than a path.
    """
    base = str(path).rsplit("/", 1)[-1]
    return base[:-5] if base.endswith(".yaml") else base


def knob_phrase(key, value):
    """Spelled-out description of one knob setting, e.g. "batch tokens 2,048"."""
    if key == "router_config":
        return f"{routing_arm_name(value)} routing"
    if key == "routing_policy":
        return f"{value} routing"
    name = KNOB_NAMES.get(key, key)
    if key in ("max_num_batched_tokens", "max_num_seqs", "block_size"):
        return f"{name} {value:,}"
    return f"{name} {value}"
