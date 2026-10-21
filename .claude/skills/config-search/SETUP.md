# Setup

What `config-search` needs on the machine before a run.

## 1. Build the pinned BLIS binary (kernel backend)

```bash
git clone --branch v0.10.0 --depth 1 https://github.com/inference-sim/inference-sim.git
cd inference-sim
# The v0.10.0 tag is NOT configured immutable upstream, so verify it still resolves
# to the pinned commit before building; fail loudly if the tag has been moved.
EXPECT=0d092d4ef64f605046de1181adf8e2b6ee564725   # release v0.10.0, "blis-latency-kernel integration" (kernel v0.1.0 in go.mod)
test "$(git rev-parse HEAD)" = "$EXPECT" || { echo "v0.10.0 no longer resolves to $EXPECT -- aborting" >&2; exit 1; }
go build -o blis main.go
```

Point the skill at this `./blis` (the `--blis` arg, default `./blis`). The kernel cost-model
backend is selected by the scenario's `coefficients:` list (the generated scenario names the six
`cost-model-*` sets), so **no `--latency-model` CLI flag is passed** — v0.10.0 removed that flag
and rejects it as unknown. `run_blis.py` writes the coefficients into the generated scenario.

## 2. Catalog + registry (both required by the kernel)

```bash
git clone --branch 0.2.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
export BLIS_CATALOG=$PWD/blis-catalog
git clone --branch v0.1.1 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_REGISTRY=$PWD/blis-registry      # MUST be v0.1.1: it carries cost-model-memory
```

A model runs only if it is in the catalog (NS-6). `glm-5.3-flash`, `glm-5.3`, and `qwen3-14b` all
ship in catalog `0.2.1`. The runner generates kernel scenario files into a `--scenarios` dir you
choose (e.g. `/tmp/cs-scenarios`).

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
