// Stage the Python-built site/ into web/public/data/ so the Vite SPA can fetch it as
// same-origin relative requests (works on GitHub Pages with no server) and `vite build`
// bundles it into dist/.
//
// Pipeline:  scripts/build_gallery.py  ->  site/{data.js,catalog.js,mode.js,<run_id>/...}
//            then this script lifts the JSON payloads + each report into web/public/data/.
//
// Produces:
//   web/public/data/scenarios.json     window.SCENARIOS payload (the gallery dataset)
//   web/public/data/catalog.json       window.CATALOG payload (models/hardware/workloads)
//   web/public/data/mode.json          { mode: "pages" | "local" }
//   web/public/data/reports/<run_id>/  the self-contained report index.html + records.json
//
// Run from web/ (npm run stage:data). It shells to python3 to (re)build site/ first, then
// copies. If site/ already exists and REUSE_SITE=1, it skips the python build (for CI that
// built site/ in a prior step).
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, rmSync, readFileSync, writeFileSync, cpSync, readdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const WEB = resolve(here, '..')
const ROOT = resolve(WEB, '..')
const SITE = join(ROOT, 'site')
const OUT = join(WEB, 'public', 'data')
const MODE = process.env.SITE_MODE === 'local' ? 'local' : 'pages'

function buildSite() {
  if (process.env.REUSE_SITE === '1' && existsSync(join(SITE, 'data.js'))) {
    console.log('stage-data: reusing existing site/ (REUSE_SITE=1)')
    return
  }
  console.log(`stage-data: building site/ (mode=${MODE}) via build_gallery.py`)
  const args = ['-I', join(ROOT, 'scripts', 'build_gallery.py'), '--mode', MODE, '--out', SITE]
  if (process.env.NO_MOCK === '1') args.push('--no-mock')
  execFileSync('python3', args, { cwd: ROOT, stdio: 'inherit' })
}

// site/ emits `window.X = {...};` files; lift the JSON payload out of each.
function liftWindowJson(file, varName) {
  const txt = readFileSync(join(SITE, file), 'utf8')
  const i = txt.indexOf('=')
  const semi = txt.lastIndexOf(';')
  const json = txt.slice(i + 1, semi === -1 ? undefined : semi).trim()
  return JSON.parse(json)
}

function main() {
  buildSite()
  if (!existsSync(join(SITE, 'data.js'))) {
    console.error('FATAL: site/data.js missing after build — cannot stage a dataless site')
    process.exit(1)
  }
  rmSync(OUT, { recursive: true, force: true })
  mkdirSync(join(OUT, 'reports'), { recursive: true })

  const scenarios = liftWindowJson('data.js', 'SCENARIOS')
  const catalog = liftWindowJson('catalog.js', 'CATALOG')

  // INLINE each scenario's per-run records into scenarios.json (they total ~40 KB), so the
  // detail explorer's cross-filter charts render from the SINGLE scenarios.json load with NO
  // per-scenario fetch. This is what makes the charts appear reliably in every viewing
  // context (the old separate records.json fetch was the fragile link). The full report and
  // its data files are still copied for the "full report ↗" link.
  let reports = 0
  for (const s of scenarios) {
    const src = join(SITE, s.run_id)
    const recPath = join(src, 'records.json')
    s.records = existsSync(recPath) ? JSON.parse(readFileSync(recPath, 'utf8')) : []
    if (!existsSync(src)) continue
    const dst = join(OUT, 'reports', s.run_id)
    mkdirSync(dst, { recursive: true })
    for (const f of ['index.html', 'records.json', 'pareto.json', 'scenario.json']) {
      if (existsSync(join(src, f))) cpSync(join(src, f), join(dst, f))
    }
    reports++
  }

  writeFileSync(join(OUT, 'scenarios.json'), JSON.stringify(scenarios))
  writeFileSync(join(OUT, 'catalog.json'), JSON.stringify(catalog))
  writeFileSync(join(OUT, 'mode.json'), JSON.stringify({ mode: MODE }))
  console.log(`stage-data: ${scenarios.length} scenarios (records inlined), ${reports} reports, `
    + `${catalog.models?.length ?? 0} models staged to web/public/data/ (mode=${MODE})`)
}

main()
