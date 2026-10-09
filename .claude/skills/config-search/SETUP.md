# Setup

What `config-search` needs on the machine before a run.

## 1. Build the pinned BLIS binary

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
git fetch origin pull/1851/head:kernel-exclusive && git checkout kernel-exclusive
go build -o blis main.go
```

Point the skill at this `./blis` (the `--blis` arg to `run_blis.py`, default `./blis`).

## 2. Catalog + registry

```bash
git clone --branch 0.2.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
export BLIS_CATALOG=$PWD/blis-catalog
```

Registry (`v0.1.1`) is resolved the same way the pinned BLIS expects. A model runs only if it is in
the catalog (NS-6).

## 3. The example models are already in catalog 0.2.1

No overlay needed: `glm-5.3-flash` (the headline example), `glm-5.3`, and `qwen3-14b` all ship in
blis-catalog `0.2.1`, each with its committed `config.json` + derived `graph.yaml`. Verified: the
GLM-5.3-Flash headline example (2x TP8 + EP) runs on the pinned stack. (Only `0.2.0` lacked
`glm-5.3-flash`.)

## 4. Smoke test the loop

Single run (works from any cwd -- the runner finds `defaults.yaml` next to the binary):

```bash
S=.claude/skills/config-search/scripts
python3 -I $S/run_blis.py --blis /path/to/blis \
  --model qwen3-14b --hardware H100 --workload chatbot \
  --num-requests 500 --seed 0 --config '{"tp":1,"num_instances":1,"concurrency":16}'
```

Confirm the printed record has `returncode: 0` and non-empty `metrics` (`tokens_per_sec`,
`itl_p99_ms`, `ttft_p99_ms`, `preemption_count`, ...). Empty metrics means `blis run`'s output format
changed -- update `parse_metrics()` in `run_blis.py`.

Parallel batch (proves the host-adaptive cap):

```bash
printf '[%s]' "$(printf '{"tp":1,"num_instances":1,"concurrency":%d},' 8 16 32 64 | sed 's/,$//')" > /tmp/batch.json
python3 -I $S/sweep_pool.py --blis /path/to/blis \
  --model qwen3-14b --hardware H100 --workload chatbot --num-requests 500 --seed 0 \
  --batch /tmp/batch.json --out /tmp/runs.jsonl
```

The summary line reports `parallel_cap`, `bound_by` (cores vs memory), and `speedup_x`.
