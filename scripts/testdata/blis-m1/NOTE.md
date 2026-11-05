# TEMPORARY fixture: the inference-sim commands

The hermetic tests need the output of `blis fields`, `blis metrics`, `blis schema`,
`blis schema validate` and `blis catalog resolve`. No released inference-sim has those commands yet: they are
being added in inference-sim#2077 (task H of milestone M1). Until a build that has them is
pinned in `pins.env`, the executable `blis` in this directory prints the same data, copied
from the commits the M1 branches pin:

| File | Copied from |
|---|---|
| `fields-vllm.json` | blis-vllm `fields.json` at e68846f2e51bf76f1c07a73f66f48d02b47af8a8 (blis-vllm#52) |
| `deployment.schema.json`, `scenario.schema.json`, `result.schema.json` | blis-substrate `contracts/schema/` at 800ae26398590cb4f4f5c41fb4d89a33930ecff2 |
| `metrics.json` | `metrics.All()` of blis-substrate `contracts/metrics` at 800ae263, plus `scopes` and `aggregates` from design §8.7, which that registry does not list yet |
| `catalog.json` | the names under blis-substrate `catalog/models/` and `catalog/hardware/` at 800ae263, and `packaging`: the `GPUsPerNode` and `GPUsPerRack` of the two entries that state them (`gb200-nvl72`, `gb300`), as `blis catalog resolve --chip` reports them from inference-sim m1/packaging 146e13d |

Only tests read this directory. Scripts take the real binary with `--blis`. Delete the
directory, and the tests' references to it, once a pinned build has the commands.
