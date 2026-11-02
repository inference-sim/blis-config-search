// Build site/ with scripts/build_gallery.py, then copy it into web/public/data/ so the SPA can
// fetch it with same-origin relative requests (works on GitHub Pages with no server).
//
//   web/public/data/scenarios.json     one row per report (cards, coverage)
//   web/public/data/catalog.json       models / hardware / workloads
//   web/public/data/reports/<run_id>/  explore.json + commands.json for the explorer, plus scenario/pareto/report.json
//
// Run from web/ (npm run stage:data). REUSE_SITE=1 skips the Python build when site/ exists.
import { execFileSync } from 'node:child_process'
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const ROOT = resolve(WEB, '..')
const SITE = join(ROOT, 'site')
const OUT = join(WEB, 'public', 'data')

if (process.env.REUSE_SITE === '1' && existsSync(join(SITE, 'scenarios.json'))) {
  console.log('stage-data: reusing existing site/ (REUSE_SITE=1)')
} else {
  execFileSync('python3', ['-I', join(ROOT, 'scripts', 'build_gallery.py'), '--out', SITE],
    { cwd: ROOT, stdio: 'inherit' })
}

rmSync(OUT, { recursive: true, force: true })
mkdirSync(join(OUT, 'reports'), { recursive: true })
let reports = 0
for (const name of readdirSync(SITE)) {
  const src = join(SITE, name)
  if (statSync(src).isDirectory()) {
    cpSync(src, join(OUT, 'reports', name), { recursive: true })
    reports++
  } else {
    cpSync(src, join(OUT, name))
  }
}
console.log(`stage-data: ${reports} reports staged to web/public/data/`)
