# Setup

What a local sweep (`scripts/run_experiment.py`) needs on the machine. CI builds the
same stack from `pins.env`; the clone lines below are kept in step by `scripts/blis_pin.py bump`.

## 1. Build the pinned BLIS binary (kernel backend)

```bash
git clone --branch m1/inference-sim --depth 1 https://github.com/inference-sim/inference-sim.git
cd inference-sim
# The m1/inference-sim tag is NOT configured immutable upstream, so verify it still resolves
# to the pinned commit before building; fail loudly if the tag has been moved.
EXPECT=3b84e0878e4c63f47d202de2ff0471ee7722a4cc   # inference-sim m1/inference-sim, the M1 draft's branch (inference-sim#2090); the release tag when tagged
test "$(git rev-parse HEAD)" = "$EXPECT" || { echo "m1/inference-sim no longer resolves to $EXPECT -- aborting" >&2; exit 1; }
go build -o blis main.go
```

Pass this `./blis` to the validator and the sweeper as `--blis`. Both read the engine's
settings from the pinned blis-vllm release's `fields.json`, which `scripts/fetch_fields.py FILE`
writes and `--fields FILE` names (the simulator prints it itself once it has `blis fields`), the pool-layout keys with `blis schema deployment`,
and model and chip names with `blis catalog resolve`; the sweeper validates each generated
scenario and each Result with `blis schema validate`. This repository keeps no copy of any of
them. The kernel cost-model
backend is selected by the scenario's `coefficients:` list (the generated scenario names the seven
`cost-model-*` sets), so **no `--latency-model` CLI flag is passed** — v0.10.0 removed that flag
and rejects it as unknown. `run_blis.py` writes the coefficients into the generated scenario.

**Routing moved in v0.12.0.** That release ("Merge llm-d-r into main: all routing through the
embedded llm-d router") removed every routing and admission CLI flag — `--routing-policy`,
`--routing-scorers`, `--admission-policy`, `--flow-control`, `--pd-decider` and ~25 more. Routing
is now a real llm-d **EndpointPickerConfig**, named by the scenario's deployment-level `router:`
block, so the sweep knob is `router_config` (a path to an EPP YAML). The per-flag mapping is
upstream in `docs/guide/routing.md`, "From BLIS's former routing features". Note that
`round-robin`, `least-loaded` and `always-busiest` are **dropped** there ("the llm-d router has no
such policy"): the nearest configurations are a `queue-scorer` or `load-aware-scorer` profile with
`max-score-picker`, or `random-picker`. So reports that compared routing policies before v0.12.0
cannot be reproduced arm-for-arm.

## 2. Catalog + registry (embedded; a checkout only for the scripts' file reads)

The simulator prices with the catalog and registry **embedded** in the blis-substrate release it
pins, so a run passes neither `--catalog` nor `--registry` to it and every Result records data
source `embedded`. A model runs only if that embedded catalog carries it (NS-6).

Some scripts read catalog and registry *files* (a workload's descriptor, a model's graph, the
speculative-decoding acceptance profiles). They read them from a checkout of the same release,
`SUBSTRATE_REF` in `pins.env`:

```bash
set -a; . ./pins.env; set +a
git clone --branch "$SUBSTRATE_REF" --depth 1 https://github.com/inference-sim/blis-substrate.git
export BLIS_CATALOG=$PWD/blis-substrate/catalog
export BLIS_REGISTRY=$PWD/blis-substrate/registry
```

These two variables (or `--catalog` / `--registry`) name where the scripts read those files; they
are not handed to the simulator. `--data-override` hands them to it, for an **experimental** run
priced with other data: its report is marked experimental and the sweeper refuses to write it into
`reports/`. The runner generates kernel scenario files into a `--scenarios` dir you choose (e.g.
`/tmp/cs-scenarios`).

## 2b. The AgentX replay corpus (only for a replay workload)

A replay workload names a corpus that the catalog deliberately does **not** vendor. For
`agentx-mvp`, fetch `traces.jsonl` from the Hugging Face dataset
`semianalysisai/cc-traces-weka-062126-256k` and check it against the sha256 the workload states:

```bash
# 568,864,747 bytes
shasum -a 256 traces.jsonl
# e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef
```

The digest is **required**, not optional: aiperf reads the dataset's default branch rather than a
pinned revision, so only the file's bytes tie a run to its corpus, and BLIS refuses a replay that
states no sha256. A report over a replay workload is therefore only regenerable where the corpus
is obtainable.

## 3. Smoke test

One run (kernel-priced; works from any cwd, since the runner finds `defaults.yaml` next to the binary):

```bash
python3 -I scripts/run_blis.py --blis /path/to/blis --scenarios /tmp/cs-scenarios \
  --model glm-5.3-flash --hardware h100 --workload chatbot --num-requests 100 --seed 0 \
  --config '{"tp":8,"enable_expert_parallel":true,"num_instances":1,"concurrency":8}'
```

Confirm the printed record has `returncode: 0`, `error: null`, and non-empty `metrics`
(`tokens_per_sec`, `itl_p99_ms`, ...). `--model`/`--hardware`/`tp`/`ep` are written into the generated
scenario; the kernel refuses them as CLI flags.

Then a whole spec, without BLIS first to check its run budget:

```bash
python3 -I scripts/run_experiment.py experiments/<name>.yaml --dry-run
python3 -I scripts/run_experiment.py experiments/<name>.yaml \
  --blis /path/to/blis --catalog "$BLIS_CATALOG" --registry "$BLIS_REGISTRY"
```

## 4. Unit tests (no BLIS / no network)

See the test list in `CLAUDE.md`; `scripts/test_run_blis.py` covers this runner.
