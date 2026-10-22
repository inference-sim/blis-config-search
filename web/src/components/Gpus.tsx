import type { Catalog, HardwareEntry, Scenario } from '../types'
import { countReports, dash } from '../lib'
import { CatalogTable, type Col } from './CatalogTable'
import { ReportsLink } from './Models'

export function Gpus({ catalog, scenarios }: { catalog: Catalog; scenarios: Scenario[] }) {
  const cols: Col<HardwareEntry>[] = [
    { key: 'name', label: 'gpu', get: (h) => h.name, render: (h) => <span className="mono">{h.name}</span> },
    { key: 'mem', label: 'memory (GiB)', get: (h) => h.memory_gib, num: true },
    { key: 'bw', label: 'mem BW (TB/s)', get: (h) => h.bw_tbs, num: true },
    { key: 'bf16', label: 'BF16 TFLOPS', get: (h) => h.tflops_bf16, num: true },
    { key: 'fp8', label: 'FP8 TFLOPS', get: (h) => h.tflops_fp8, num: true },
    { key: 'fp4', label: 'NVFP4 TFLOPS', get: (h) => h.tflops_nvfp4, num: true },
    { key: 'sm', label: 'SMs', get: (h) => h.sm_count, num: true },
    { key: 'nv', label: 'intra-node BW (GB/s)', get: (h) => h.intra_node_bw_gbps, num: true },
    { key: 'pn', label: 'GPUs / node', get: (h) => h.gpus_per_node, num: true },
    { key: 'pr', label: 'GPUs / rack', get: (h) => h.gpus_per_rack, num: true },
    { key: 'prov', label: 'provenance', get: (h) => h.provenance },
    {
      key: 'reports',
      label: 'reports',
      get: (h) => countReports(scenarios, 'hardware', h.name),
      render: (h) => <ReportsLink n={countReports(scenarios, 'hardware', h.name)} params={{ hardware: h.name }} />,
    },
  ]
  return (
    <div>
      <h2 className="viewhead">GPUs</h2>
      <p className="dek"><b>{catalog.hardware.length}</b> hardware entries. Missing values are shown as {dash}.</p>
      <CatalogTable rows={catalog.hardware} cols={cols} rowKey={(h) => h.name} caption="Hardware catalog" />
    </div>
  )
}
