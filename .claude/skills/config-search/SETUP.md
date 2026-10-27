# Setup

What `config-search` needs on the machine before a run.

## 1. Build the pinned BLIS binary (kernel backend)

```bash
git clone --branch v0.12.0 --depth 1 https://github.com/inference-sim/inference-sim.git
cd inference-sim
# The v0.12.0 tag is NOT configured immutable upstream, so verify it still resolves
# to the pinned commit before building; fail loudly if the tag has been moved.
EXPECT=f199223d701d96dddeb1c314509bed003515b599   # release v0.12.0, "all routing through the embedded llm-d router" (kernel v0.3.0, schemas v0.5.0 in go.mod)
test "$(git rev-parse HEAD)" = "$EXPECT" || { echo "v0.12.0 no longer resolves to $EXPECT -- aborting" >&2; exit 1; }
go build -o blis main.go
```

Point the skill at this `./blis` (the `--blis` arg, default `./blis`). The kernel cost-model
backend is selected by the scenario's `coefficients:` list (the generated scenario names the six
`cost-model-*` sets), so **no `--latency-model` CLI flag is passed** — v0.10.0 removed that flag
and rejects it as unknown. `run_blis.py` writes the coefficients into the generated scenario.

**Routing moved in v0.12.0.** That release ("Merge llm-d-r into main: all routing through the
embedded llm-d router") removed every routing and admission CLI flag — `--routing-policy`,
`--routing-scorers`, `--admission-policy`, `--flow-control`, `--pd-decider` and ~25 more. Routing
is now a real llm-d **EndpointPickerConfig**, named by the scenario's deployment-level `router:`
block, so the search knob is `router_config` (a path to an EPP YAML). The per-flag mapping is
upstream in `docs/guide/routing.md`, "From BLIS's former routing features". Note that
`round-robin`, `least-loaded` and `always-busiest` are **dropped** there ("the llm-d router has no
such policy"): the nearest configurations are a `queue-scorer` or `load-aware-scorer` profile with
`max-score-picker`, or `random-picker`. So reports that compared routing policies before v0.12.0
cannot be reproduced arm-for-arm.

## 2. Catalog + registry (both required by the kernel)

```bash
git clone --branch 0.3.0 --depth 1 https://github.com/inference-sim/blis-catalog.git
export BLIS_CATALOG=$PWD/blis-catalog
git clone --branch v0.2.0 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_REGISTRY=$PWD/blis-registry      # MUST be v0.2.0: it carries cost-model-memory AND
                                             # acceptance/, the measured speculative-decoding
                                             # acceptance lengths (v0.1.1 has no acceptance/)
```

A model runs only if it is in the catalog (NS-6). `glm-5.3-flash`, `glm-5.3`, and `qwen3-14b` all
ship in catalog `0.3.0`, which also adds `deepseek-v4.1-flash` and the `agentx-mvp` replay
workload. The runner generates kernel scenario files into a `--scenarios` dir you choose (e.g.
`/tmp/cs-scenarios`).

The catalog pin and the binary pin move **together**: catalog `0.3.0`'s model graphs use schema
fields that v0.10.0 cannot decode (it fails with hundreds of `field ... not found in type
model.Node` errors), and v0.12.0 needs `0.3.0` for the entries above.

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

## 3. Smoke test the loop

Single run (kernel-priced; works from any cwd -- the runner finds `defaults.yaml` next to the binary):

```bash
S=.claude/skills/config-search/scripts
python3 -I $S/run_blis.py --blis /path/to/blis --scenarios /tmp/cs-scenarios \
  --model glm-5.3-flash --hardware h100 --workload chatbot --num-requests 100 --seed 0 \
  --config '{"tp":8,"enable_expert_parallel":true,"num_instances":1,"concurrency":8}'
```

Confirm the printed record has `returncode: 0`, `error: null`, and non-empty `metrics`
(`tokens_per_sec`, `itl_p99_ms`, ...). `--model`/`--hardware`/`tp`/`ep` are written into the generated
scenario; the kernel refuses them as CLI flags.

Parallel batch (proves the host-adaptive cap):

```bash
printf '[%s]' "$(printf '{"tp":8,"enable_expert_parallel":true,"num_instances":1,"concurrency":%d},' 1 4 8 16 | sed 's/,$//')" > /tmp/batch.json
python3 -I $S/sweep_pool.py --blis /path/to/blis --scenarios /tmp/cs-scenarios \
  --model glm-5.3-flash --hardware h100 --workload chatbot --num-requests 100 --seed 0 \
  --batch /tmp/batch.json --out /tmp/runs.jsonl
```

The summary line reports `parallel_cap`, `bound_by` (cores vs memory), and `speedup_x`.

## 4. Unit tests (no BLIS / no network)

```bash
python3 .claude/skills/config-search/scripts/test_run_blis.py
```

Covers `parse_metrics` (cluster selection, braces-in-strings, nested envelope, per-instance), the
`_deployment`/scenario rendering, the kernel `build_cmd`, and the `compute_cap` host-adaptive logic.
