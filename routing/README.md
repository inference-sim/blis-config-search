# Routing arms (llm-d EndpointPickerConfig)

inference-sim **v0.12.0** removed `--routing-policy` and every other routing/admission flag:
routing is now the embedded llm-d router, configured by a real **EndpointPickerConfig** that the
scenario's deployment-level `router:` block names. So a routing arm in this search is a *file*
here, passed as the `router_config` knob, not a policy name.

The per-flag mapping is upstream in inference-sim `docs/guide/routing.md`, section "From BLIS's
former routing features". The rows that matter for this search:

| former BLIS routing | here |
|---|---|
| `--routing-policy weighted --routing-scorers queue-depth,kv-utilization` | `queue-kv.yaml` |
| scorer `queue-depth` | `queue.yaml` (`queue-scorer` + `max-score-picker`) |
| scorer `load-aware` | `load-aware.yaml` |
| scorer `prefix-affinity` | `prefix-affinity.yaml` (`approx-prefix-cache-producer` + `prefix-cache-scorer`) |
| `--routing-policy round-robin` / `least-loaded` / `always-busiest` | **dropped upstream** — "the llm-d router has no such policy". `random-picker` (`random.yaml`) is the nearest neutral baseline. |

`least-loaded` was the biggest lever the earlier campaigns measured (+84% tok/s/GPU on
kimi-k2.5), and it has **no exact equivalent** on this stack. A pre-v0.12.0 report that compared
routing policies therefore cannot be reproduced arm-for-arm; the nearest comparison is
`queue.yaml` / `load-aware.yaml` against the router's default.

Omitting `router_config` entirely runs the router's built-in standalone configuration over every
pod, which is the baseline every arm here is measured against. Measured on
deepseek-v4.1-flash / 4x TP8xDP4 / 64 sessions: the default and `queue.yaml` are byte-identical
(the default profile already scores on queue depth), while `random.yaml` moves both throughput and
the per-instance request distribution — which is how you can tell these files are really applied.
