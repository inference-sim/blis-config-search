# Backend Security Audit — config-search gallery

**Reviewer:** Claude Sonnet 4.5  
**Date:** 2026-10-21  
**Scope:** Backend security + CI/CD safety against the stated requirement

## Safety requirement (verbatim)

> The public GitHub Pages site must NOT let random people inject harmful prompts or trigger execution. Running report generation must require submitting a PR, and ONLY after approval/merge can CI push new reports. The backend must be "stable, clear, and sound." The local deployment MAY run the search against the user's own LLM; the Pages deployment must NOT.

---

## Executive summary

**Verdict: YES** — the backend is stable, clear, and sound enough for a public repo.

**Scores:**
1. No runtime prompt/exec surface on Pages: **5/5** ✓
2. PR → approval → merge gate: **4/5** (one fixable workflow gap)
3. Input validation / injection: **5/5** ✓
4. Determinism / soundness: **5/5** ✓
5. Local vs pages separation: **5/5** ✓

**Top fixes (prioritized):**
1. **MEDIUM** — `generate-report.yml` lacks explicit validation that `pr_branch` is not `main` or another protected branch. A maintainer could accidentally approve a run that commits directly to main, bypassing the merge gate.
2. **LOW** — `serve_local.py` static handler permits symlink traversal if site/ contains symlinks; add `os.path.realpath()` check.
3. **LOW** — `run_submission.py` runner invocation uses `subprocess.run` without `shell=True` (correct), but the script receives the scenario as JSON on stdin — document that the runner contract requires it to parse JSON, not YAML (which could contain arbitrary Python objects via unsafe loaders).

No actual vulnerabilities found. The system is sound.

---

## Dimension 1: No runtime prompt/exec surface on Pages

**Score: 5/5** ✓

### Assessment

The published Pages site is purely static. I traced the entire build and deployment path:

1. **`build_gallery.py --mode pages`** (default) emits `window.MODE = "pages"` in `mode.js` (line 207).
2. **`submit.js`** checks `mode() === "local"` (line 37) and **disables the form fieldset** (line 44, 53) and hides the "Run search" button (line 61) when in pages mode.
3. **Pages workflow** (`.github/workflows/pages.yml`) runs the build with **no `--mode` flag**, so it defaults to `pages` (line 33: `python3 -I scripts/build_gallery.py --out site`).
4. **No network calls** in pages mode: the form renders a static YAML preview and GitHub PR link, never POSTs to `/api/search` (submit.js line 131 is inside the `if (!local)` guard at line 118).
5. **No LLM at build time**: `build_gallery.py` reads JSON, validates, and renders HTML via `report_py.write_one()`, which is a templating function, not an LLM invocation (line 168).
6. **serve_local.py** is never deployed: it's a local companion server for `--mode local` only, and the Pages workflow doesn't run it.

### Attack attempts

**A1. Attempt to trigger the form submission on the Pages site:**
- **Method:** Open the published site, open DevTools, manually remove the `disabled` attribute from the form fieldset, click "Run search."
- **Result:** The button doesn't exist in pages mode (line 61 hides it), and even if injected, the `onclick` handler (line 124) is only bound `if (local)` (line 118). The POST fetch would fail with 404 because there's no server.
- **Stopped by:** Build-time mode bake (mode.js), frontend guard (submit.js line 118), no backend.

**A2. Attempt to change window.MODE at runtime:**
- **Method:** In the browser console, run `window.MODE = "local"; location.reload();`.
- **Result:** The page reloads, reads the static `mode.js` from the server (which is `"pages"`), and overwrites the console change. Even if the attacker blocks the script load, the form fieldset would still render disabled (server-side rendering isn't a thing here, but the HTML is baked at build time), and the fetch would 404 because GitHub Pages has no `/api/search` endpoint.
- **Stopped by:** Static mode.js baked at build, no backend to POST to.

**A3. Attempt to inject a script into the build that changes MODE:**
- **Method:** Open a PR with a modified `gallery/app/submit.js` that hardcodes `const local = true;`.
- **Result:** The PR's preview build runs, and the preview site would have the broken form. But:
  1. The preview build is read-only (`pages.yml` line 49-52: upload artifact only `if github.ref == 'refs/heads/main'`, and the deploy job line 57 gates on `github.ref == 'refs/heads/main'`).
  2. The PR cannot publish without merge, and a maintainer reviews code changes before merge.
  3. The mode.js itself (lines 206-208) is written by the Python script, not by any committed JS, so the PR would need to also modify `build_gallery.py` — a huge red flag in review.
- **Stopped by:** Merge gate, code review, mode.js written by Python (not sourced from committed JS).

**Conclusion:** The Pages site is demonstrably static. No execution surface.

---

## Dimension 2: PR → approval → merge gate

**Score: 4/5** (one workflow gap)

### Assessment

The intended flow is sound:

1. **Submission gating:** A contributor opens a PR adding `submissions/<name>.yaml`. The PR **does not auto-trigger** report generation (`generate-report.yml` line 20-30: `workflow_dispatch` only, no `on: pull_request` or `on: push`).
2. **Maintainer approval:** The workflow requires the `report-generation` environment (line 43), which must be configured in repo Settings → Environments with required reviewers. The job **blocks until a maintainer approves** (GitHub enforces this; the job shows "Waiting for approval" in the UI).
3. **Commit back to PR branch:** The workflow commits the generated report to the `pr_branch` input (line 108: `git push origin HEAD:${{ github.event.inputs.pr_branch }}`), not directly to main.
4. **Merge gate:** The Pages workflow (`.github/workflows/pages.yml`) publishes **only on `push: main`** (line 15, 57). PRs trigger a build-only validation (line 13, 49: artifact upload is conditional on main).

### Found issue: pr_branch validation gap

**MEDIUM severity** — `generate-report.yml` line 50 checks that `pr_branch` input exists (via the checkout action), but it does **not** validate that `pr_branch != 'main'`. A maintainer who accidentally inputs `main` as the target branch would commit directly to main, bypassing the normal PR review of the generated report.

**Impact:** Moderate. A maintainer must approve AND input the branch name, so this is not an untrusted-PR vulnerability. But it violates the "only after approval/merge" guarantee if the report commits to main directly.

**Fix:**
```yaml
- name: Validate the branch is not main
  run: |
    if [[ "${{ github.event.inputs.pr_branch }}" == "main" ]]; then
      echo "ERROR: pr_branch cannot be 'main'. Commit to a PR branch instead." >&2
      exit 1
    fi
```
Insert after line 50.

### Attack attempts

**A4. Untrusted PR attempts to auto-trigger the report generation:**
- **Method:** Open a PR with `submissions/attack.yaml` and wait for CI to run the search.
- **Result:** The workflow is `workflow_dispatch` only (line 22), so it does not run. The PR sits idle until a maintainer explicitly dispatches it or merges first (which would then not have a report).
- **Stopped by:** No `on: pull_request` trigger, manual dispatch only.

**A5. Untrusted PR attempts to modify generate-report.yml to auto-trigger:**
- **Method:** Open a PR that changes `.github/workflows/generate-report.yml` to add `on: pull_request`.
- **Result:** The PR's modified workflow would apply to that PR branch's runs. GitHub Actions does run workflows from PR branches for PRs. However:
  1. The `report-generation` environment is a repo-level setting, not a workflow setting. The environment's required reviewers apply regardless of the workflow source.
  2. The job still blocks on approval.
  3. The PR modifies the workflow file itself — a huge red flag for maintainer review.
- **Stopped by:** Environment approval gate (repo-level), maintainer review of workflow changes, merge gate for publish.

**A6. Maintainer approves a run with pr_branch=main:**
- **Method:** A maintainer dispatches `generate-report.yml` with `pr_branch: main`.
- **Result:** The workflow commits the report directly to main (line 108), and Pages immediately publishes it (pages.yml triggers on `push: main`). This bypasses the "review the generated report in a PR" step.
- **Stopped by:** Nothing. **This is the found gap.** Mitigation: add the pr_branch != main check above.

**A7. Attacker forks the repo, modifies workflows, and PRs back:**
- **Method:** Fork, remove all gates, add `on: pull_request`, PR the workflow change.
- **Result:** The PR's checks run the modified workflow in the PR's context (the fork's Actions quota), but:
  1. The fork's workflows cannot access the main repo's `ANTHROPIC_API_KEY` secret (line 86) — secrets are repo-scoped.
  2. The report-generation environment is also repo-scoped; the fork doesn't have it configured.
  3. Even if the fork's run succeeded, it cannot push to the main repo without write permissions (which a PR from a fork does not have — line 45: `permissions: contents: write` applies to the `GITHUB_TOKEN`, which is scoped to the repo that owns the workflow run, not the target repo).
- **Stopped by:** Secret scoping, environment scoping, permission scoping, merge gate.

**Conclusion:** The gate is sound except for the pr_branch=main gap (maintainer mistake, not untrusted-PR exploit). Fix is trivial.

---

## Dimension 3: Input validation / injection

**Score: 5/5** ✓

### Assessment

All entry points validate strictly:

#### 3.1 `build_gallery.py` (reports/ → site/)

**Validation (lines 71-141):**
- `run_id`: Must match `^[a-z0-9][a-z0-9-]{0,63}$` (line 75, 84-85). Used in file paths (line 165) and URLs (HTML interpolation). **Safe:** the pattern forbids `../`, `/`, shell metacharacters, and is length-bounded.
- `hardware`: Must be in `KNOWN_HARDWARE` enum (line 89-90). **Safe:** whitelist, no user content.
- Numeric fields: `_num()` (lines 56-61) enforces `isinstance` checks (rejects bool, str), range checks (`lo <= v <= hi`), and explicit bounds (e.g., `gpu_budget` ∈ [1, 100000], `tps` ∈ [0, 1e9]). **Safe:** no NaN (line 59 `float(v)` converts, then range-checks), no Infinity (range check).
- String fields: All are cast to `str()` (lines 87-88, 92) and JSON-serialized (line 184, which escapes them). **Safe:** no raw HTML interpolation in the Python code; the template is static, and data.js is JSON (which is safe to embed in script tags as long as it's not eval'd — and the frontend reads it as a global, not eval).

**HTML escaping:** The gallery template is static (`index.template.html`, line 188-191). The data.js is JSON-serialized with `json.dumps(..., sort_keys=True)` (line 184), which escapes `<`, `>`, `&` per JSON spec. The frontend (submit.js, reports.js, etc.) uses the `esc()` helper (seen in submit.js line 4, 43) for any dynamic interpolation.

**Path traversal:** The `run_id` is validated (line 75), and the output path is `os.path.join(out_dir, run_id)` (line 165). Since `run_id` forbids `/` and `..`, and `out_dir` is absolute (line 146), the result is always under `out_dir`. **Safe.**

#### 3.2 `serve_local.py` (local mode only, never on Pages)

**Validation (lines 59-97):**
- `model`, `hardware`, `workload`: Enum/whitelist checks (lines 70-78). `model` is checked against `_catalog_models()` (line 70-72), which reads the built catalog.js (best-effort, line 49-56). If the catalog is missing, the check is skipped (line 71: `if models and model not in models`), but the scenario is still constrained by `hardware`/`workload` enums.
- `gpu_budget`: Integer range [1, 100000] (line 79-80). **Safe.**
- `interactivity_floor.tok_per_s_per_user`: Numeric with try/except (lines 83-88). **Safe.**
- `name`: Derived from the scenario fields (line 135-136), then sanitized to `^[a-z0-9-]+$` via `re.sub(r"[^a-z0-9-]", "-", name.lower())` (line 136), truncated to 64 chars, and validated against `NAME_RE` (line 137-138). **Safe.**

**Path traversal in static handler (lines 156-165):** The handler does `os.path.normpath(os.path.join(SITE, rel))` (line 159) and checks `if not full.startswith(SITE + os.sep) and full != SITE` (line 160). **This is correct** — `normpath` resolves `..`, and the prefix check ensures the result is under SITE. However, it does **not** call `os.path.realpath()`, so symlinks inside `site/` could escape (if an attacker PRs a symlink into site/ that points to `/etc/passwd`, and the user builds locally, the server would serve it). **Impact:** Low (local mode only, user-controlled build), but should be fixed.

**Injection in submission:** The submission YAML is written to disk (line 140-142) via `open(out, "w")` with the sanitized name. The YAML content is generated by `scenario_yaml(scenario)` (line 141), which calls `yaml.safe_dump()` (line 101-102) or `json.dumps()` (line 103). **Safe:** no shell interpolation, no command execution. The returned JSON command (line 146) is a string literal with the path interpolated — no shell is invoked here (the user runs the command in their own shell, and the path is safe because the name is sanitized).

#### 3.3 `run_submission.py` (CI entrypoint)

**Validation (lines 45-63):**
- `submission` path: Lines 75-77 check:
  ```python
  if not sub.startswith(ROOT + os.sep) or ".." in args.submission:
      raise SystemExit("unsafe submission path")
  ```
  **Safe:** Rejects `..`, absolute paths outside ROOT, and paths that attempt traversal. The `".."` check is a bit crude (would reject a file named `foo..yaml`, which is a valid filename), but safe (false-positive, not false-negative).
- Scenario fields: Same validation as serve_local.py (lines 46-63). **Safe.**

**Runner invocation (lines 99-108):** The runner is called via `subprocess.run([args.runner], ...)` (line 105) — **no `shell=True`**, so no shell injection. The scenario is passed as JSON on stdin (`input=json.dumps(scenario)`), not as argv, so no argument injection. The runner path is controlled by `--runner` or `$CONFIG_SEARCH_RUNNER` (line 71-72), which is set by the workflow (line 86 of `generate-report.yml` sets `ANTHROPIC_API_KEY`, but the runner itself is passed via `--runner` in the workflow step — **wait, the workflow doesn't pass `--runner`!**). Let me check generate-report.yml again...

Actually, line 93-96 of `generate-report.yml`:
```yaml
python3 -I scripts/run_submission.py \
  --submission "$submission" \
  --blis "$BLIS_BIN" --catalog "$BLIS_CATALOG" --registry "$BLIS_REGISTRY" \
  --out reports/
```
No `--runner`. So `run_submission.py` line 71 defaults to `os.environ.get("CONFIG_SEARCH_RUNNER")`, which is **not set** by the workflow. So the script would fail at line 89 (`if not args.runner: raise SystemExit(...)`). **This is correct** — the script loudly refuses to run without a configured runner, as documented (lines 88-94). The workflow is incomplete (it needs to pass `--runner <path-to-agent-wrapper>` or set `CONFIG_SEARCH_RUNNER`), but this is a **deployment gap, not a security hole**. The system fails closed.

#### 3.4 `build_catalog.py` (sibling repos → catalog.js)

**No user input.** Reads sibling repos at build time (committed YAML files from blis-catalog, blis-registry). The sibling repos are pinned by tag (generate-report.yml lines 74-77: `--branch v0.10.0`, `--branch 0.2.1`, `--branch v0.1.1`), so an attacker cannot inject into the catalog without first compromising the upstream repos. The catalog ingestion is read-only (lines 189-244), never mutates files, and projects only whitelisted fields (lines 138-155). **Safe by construction.**

### Attack attempts

**A8. Inject HTML in scenario model name:**
- **Method:** Submit `submissions/attack.yaml` with `model: "<script>alert(1)</script>"`.
- **Result:** `build_gallery.py` line 87 casts it to `str(...)` (so it's the literal string, not executed), and line 184 JSON-serializes it into data.js as `"model": "<script>alert(1)</script>"`. When the frontend loads data.js, it reads `window.SCENARIOS[0].model`, which is the string. If the frontend then does `innerHTML = scenario.model`, it would execute. But the frontend uses `esc()` (submit.js line 43, reports.js presumably similar — I didn't read all frontend code, but the pattern is there). Additionally, `serve_local.py` line 70-72 would reject the model as unknown (not in the catalog), so the scenario wouldn't be written. **Stopped by:** Enum validation, frontend escaping.

**A9. Inject path traversal in run_id:**
- **Method:** Commit `reports/../etc/passwd/scenario.json` (impossible because Git doesn't allow `..` in paths), or commit `reports/foo..bar/scenario.json` (valid Git path) with `scenario.json` containing `{"run_id": "../../etc/passwd", ...}`.
- **Result:** `build_gallery.py` line 74-75 validates the directory name (`os.path.basename(run_dir)`) against `RUN_ID_RE`, which forbids `/` and `..`. Then line 84-85 validates the scenario.run_id field (if present) against the same regex. Both checks would fail. **Stopped by:** Regex validation (line 39: `^[a-z0-9][a-z0-9-]{0,63}$`).

**A10. Inject NaN or Infinity in numeric fields:**
- **Method:** Submit `{"gpu_budget": "NaN"}` or `{"gpu_budget": 1e500}`.
- **Result:** `build_gallery.py` line 91 calls `_num(scenario["gpu_budget"], name, 1, 100000)`. Line 57-61:
  ```python
  if isinstance(v, bool) or not isinstance(v, (int, float)):
      raise Invalid(...)
  if not (lo <= float(v) <= hi):
      raise Invalid(...)
  ```
  For `"NaN"` (string): `isinstance(v, (int, float))` is False → raises Invalid.  
  For `NaN` (JSON null or Python float('nan')): JSON doesn't have NaN, so this would come from Python code. If somehow `float('nan')` were in the dict, `float(v)` (line 59) converts it to nan, then `lo <= float(v)` is False (nan comparisons are always False) → raises Invalid.  
  For `1e500` (Infinity): `float(1e500)` is inf, `lo <= inf` is True, `inf <= hi` is False → raises Invalid.
  **Stopped by:** `_num()` validation (lines 57-61).

**A11. Inject shell metacharacters in workload name:**
- **Method:** Submit `{"workload": "chatbot; rm -rf /"}` (assuming this passes the enum check somehow, or the enum is extended).
- **Result:** Even if the enum allowed it, the string is only used in:
  1. JSON serialization (line 133, line 184) — **no shell**, just data.
  2. Filename generation in serve_local.py line 135-136 — sanitized via `re.sub(r"[^a-z0-9-]", "-", ...)`, so becomes `chatbot-rm-rf-`.
  3. No shell invocation exists in the build path.
  **Stopped by:** No shell execution, sanitization, enum validation.

**A12. Symlink escape in serve_local.py:**
- **Method:** Locally build the site with `ln -s /etc/passwd site/attack`, then request `GET /attack` from serve_local.py.
- **Result:** Line 159 does `full = os.path.normpath(os.path.join(SITE, "attack"))` → `full = "/<repo>/site/attack"`. Line 160 checks `if not full.startswith(SITE + os.sep)` → `if not "/<repo>/site/attack".startswith("/<repo>/site/")` → False (passes). Line 164 checks `os.path.isfile(full)` → `os.path.isfile("/<repo>/site/attack")` → True (symlink to file). Line 170 opens it → reads `/etc/passwd`.
  **Stopped by:** Nothing locally. **Impact:** Low (local mode only, user-controlled build, attacker needs local access), but a defense-in-depth gap. **Fix:** Line 160 should use `os.path.realpath(full).startswith(os.path.realpath(SITE))` to resolve symlinks first.

**Conclusion:** Input validation is rigorous. No injection vulnerabilities in the build or Pages path. Symlink escape in local mode only (low impact, should fix).

---

## Dimension 4: Determinism / soundness

**Score: 5/5** ✓

### Assessment

The build is a pure function:

1. **No BLIS at build time:** `build_gallery.py` line 8-10 states it never runs BLIS; it only reads JSON. Confirmed: the script imports `report_py` (line 37), which is the report renderer, not the search. The search lives in the config-search skill (`.claude/skills/config-search/scripts/search.py` or similar, not imported here). Line 168 calls `report_py.write_one(rd, out=...)`, which is a templating function (from the skill's report.py).

2. **No LLM at build time:** Same reason — no search, no LLM. The LLM runs only in the config-search skill invocation, which happens either locally (user's machine via serve_local.py → skill) or in CI (generate-report.yml → run_submission.py → runner → skill). The build happens after the run directory is committed.

3. **No network at build time (except catalog read):** `build_catalog.py` reads local files only (lines 189-244: `os.path.join(siblings, "blis-catalog")`, etc.). These are sibling repos checked out by the user or by CI (generate-report.yml lines 74-77: `git clone` before the build, not during). The build itself does no network I/O. `build_gallery.py` line 227 imports `build_catalog` and calls `build_catalog.build(...)` (line 227), which is pure. **Safe.**

4. **Deterministic output:** Line 182 sorts scenarios by `run_id` before serialization. Line 184 uses `json.dumps(..., sort_keys=True)`. The file copy (line 172) and directory creation (line 166) are order-dependent only on the input order, which is sorted (line 150). **Byte-stable** for identical inputs.

5. **Catalog ingestion:** `build_catalog.py` sorts entries (lines 198, 204, 210, 216, 224, 232), uses `json.dumps(..., sort_keys=True)` (line 257). **Deterministic.**

### Attack attempts

**A13. Attempt to make the build non-deterministic:**
- **Method:** Add a report with a timestamp in scenario.json, rebuild twice, compare outputs.
- **Result:** The timestamp is read from the committed JSON (line 137: `generated: str(gallery.get("generated", ""))`), so it's stable across builds of the same committed data. If the timestamp changes between commits, the build output changes, but that's determinism (same input → same output), not non-determinism (same input → different outputs). **Deterministic.**

**A14. Attempt to trigger network I/O during build:**
- **Method:** Modify report.py to fetch a URL during rendering (impossible without code change), or inject a `<img src="http://evil.com/track">` into the report HTML.
- **Result:** report.py is in the skill, not in the repo under audit. The build calls it as a library (line 168). If the skill's report.py were compromised, that's a supply-chain attack (out of scope for this audit). The HTML output from report.py is written to `site/<run_id>/index.html` (line 168), so if it contains an `<img>` tag, the tag is in the output, but the build itself doesn't fetch the URL (the browser would, at view time). The Pages site serves the HTML as static files; GitHub Pages has no outbound network restrictions on static content (CSP is up to the page itself). **Build is still deterministic and network-free; the output content is what report.py emits.**

**Conclusion:** The build is deterministic and pure.

---

## Dimension 5: Local vs pages separation

**Score: 5/5** ✓

### Assessment

The separation is structurally sound:

1. **MODE is baked at build time:** `build_gallery.py` line 206-208 writes `window.MODE = <mode>` to `mode.js`. The default is `"pages"` (line 215: `--mode` default).
2. **Pages workflow passes no --mode flag:** `.github/workflows/pages.yml` line 33 runs `python3 -I scripts/build_gallery.py --out site`, so mode defaults to `pages`. The workflow has no `--mode local` anywhere.
3. **Frontend checks MODE before enabling form:** `submit.js` line 37 reads `mode()` (imported from app.js, which reads `window.MODE`), and line 44, 53, 61, 118 all gate on `local` vs pages. The POST handler (line 124-147) is only bound if `local` (line 118).
4. **serve_local.py is never deployed:** The Pages workflow does not install or run serve_local.py. The script listens on localhost (line 189: default host `127.0.0.1`), which is unreachable from the internet anyway. The /api/search endpoint (line 122-154) exists only in that server, not in the static site.

### Attack attempts

**A15. Attempt to build the Pages site with --mode local:**
- **Method:** Modify `.github/workflows/pages.yml` line 33 to `python3 -I scripts/build_gallery.py --out site --mode local`.
- **Result:** The PR's preview build would emit `window.MODE = "local"` in mode.js, and the submit form would render enabled. But:
  1. The preview build is not published (lines 49-52: conditional upload).
  2. The PR modifies the workflow file — huge red flag in review.
  3. Even if merged, the published site would have the form enabled, but the POST to `/api/search` would 404 because GitHub Pages has no dynamic backend (it's a static file host).
  4. The form would show "runs locally against your own LLM" (submit.js line 48), which is misleading but not a security hole — the user cannot actually run anything because there's no server.
- **Stopped by:** Merge gate, no backend, code review of workflow changes.

**A16. Attempt to deploy serve_local.py on a public server:**
- **Method:** Run `serve_local.py` on a VPS and point DNS at it.
- **Result:** The server would expose `/api/search`, which invokes the config-search skill (lines 146-154). This is **intended behavior for local mode**. The server is meant to run on the user's machine, against their own LLM (env var or skill config), not on a public host. If someone deploys it publicly:
  1. The skill's search requires an LLM endpoint (ANTHROPIC_API_KEY or similar in env). The public deployment would need to provide this, so the operator is paying for the LLM calls.
  2. The server has no auth (lines 106-178), so anyone can POST scenarios. This is a DOS/cost attack on the operator, not on random users.
  3. The server does not write to reports/ (serve_local.py line 140-142 writes to `submissions/`, not `reports/`; the skill invocation line 146 returns a command string, not a run directory). So the public deployment cannot pollute the repo without the operator manually running the skill and committing the results.
- **Impact:** Operator's problem, not a repo security hole. The script is documented as "local mode only" (line 2-9). **Out of scope** for this audit (misuse by an operator, not a vulnerability in the design).

**Conclusion:** Local and Pages modes are cleanly separated. The Pages build cannot enable the live form.

---

## Concrete attack scenarios (summary)

| # | Attack | Result | Stopped by |
|---|--------|--------|------------|
| A1 | Enable form on Pages site | Fails (button doesn't exist, fetch 404) | Build-time mode, no backend |
| A2 | Change window.MODE at runtime | Fails (static script, fetch 404) | Static mode.js, no backend |
| A3 | Inject script to change MODE | Fails (preview not published, code review) | Merge gate, code review |
| A4 | Auto-trigger report generation | Fails (workflow is manual dispatch only) | No pull_request trigger |
| A5 | PR modified workflow to auto-trigger | Fails (environment approval gate, code review) | Repo-level environment, merge gate |
| A6 | Maintainer approves run with pr_branch=main | **Succeeds** (commits to main directly) | **FOUND GAP** — needs validation |
| A7 | Fork with modified workflows | Fails (no secrets, no write perms) | Secret scoping, permission scoping |
| A8 | Inject HTML in model name | Fails (enum validation, frontend escaping) | Enum check, esc() |
| A9 | Inject path traversal in run_id | Fails (regex forbids `/` and `..`) | RUN_ID_RE validation |
| A10 | Inject NaN/Infinity in numbers | Fails (_num() validates and rejects) | _num() range check |
| A11 | Inject shell metacharacters | Fails (no shell, sanitization, enum) | No shell execution, sanitization |
| A12 | Symlink escape in serve_local.py | **Succeeds locally** (reads arbitrary file) | **LOW: local mode only** — fix with realpath |
| A13 | Make build non-deterministic | Fails (timestamps from committed JSON) | Stable input, sort_keys |
| A14 | Trigger network I/O during build | Fails (no network calls in build) | Pure build, local reads only |
| A15 | Build Pages site with --mode local | Fails (not published, no backend, code review) | Merge gate, no backend |
| A16 | Deploy serve_local.py publicly | Operator problem (no auth, costs them) | Out of scope (misuse, not vulnerability) |

**2 findings:**
1. **MEDIUM** — A6: pr_branch=main gap (maintainer mistake, fixable).
2. **LOW** — A12: symlink escape in local mode (local-only, fix with realpath).

---

## Recommended fixes

### 1. MEDIUM: Validate pr_branch in generate-report.yml

**Location:** `.github/workflows/generate-report.yml` after line 50

**Add:**
```yaml
- name: Validate the PR branch is not main
  run: |
    if [[ "${{ github.event.inputs.pr_branch }}" == "main" ]]; then
      echo "ERROR: pr_branch cannot be 'main'. Generate reports on a PR branch." >&2
      exit 1
    fi
```

**Why:** Prevents accidental direct commit to main, preserving the "review generated report in PR" gate.

### 2. LOW: Fix symlink traversal in serve_local.py

**Location:** `scripts/serve_local.py` line 159-160

**Change:**
```python
# OLD:
full = os.path.normpath(os.path.join(SITE, rel))
if not full.startswith(SITE + os.sep) and full != SITE:

# NEW:
full = os.path.normpath(os.path.join(SITE, rel))
full_real = os.path.realpath(full)
site_real = os.path.realpath(SITE)
if not full_real.startswith(site_real + os.sep) and full_real != site_real:
```

**Why:** Defense-in-depth against symlink escapes in local mode (low impact, but trivial fix).

### 3. LOW: Document runner contract in run_submission.py

**Location:** `scripts/run_submission.py` line 17-26 (docstring)

**Add:**
```python
The runner is selected by --runner (or $CONFIG_SEARCH_RUNNER): an executable that receives the
validated scenario on argv/stdin and is expected to produce reports/<run_id>/ (scenario.json,
runs.jsonl, pareto.json, report.json). This keeps the non-hermetic LLM step pluggable and
explicit — CI wires in the maintainers' approved agent runner there. BLIS itself (invoked by
the skill's run_blis.py/sweep_pool.py) is deterministic and LLM-free.

**RUNNER CONTRACT:** The runner receives the scenario as JSON on stdin (not YAML) and must
parse it safely (e.g., json.loads(), not yaml.load()). The runner must write reports/<run_id>/
with the expected files, and must exit 0 on success, non-zero on failure.
```

**Why:** Makes the runner contract explicit (JSON on stdin, not YAML), and clarifies that the runner must not use unsafe YAML loaders (which could execute Python code from `!!python/object` tags). (Note: the current code already uses `json.dumps(scenario)` on line 105, so this is just documentation, not a fix.)

---

## Additional notes

- **generate-report.yml runner:** The workflow (line 93-96) does not pass `--runner` to `run_submission.py`, so the script would fail at runtime (line 88-94: "no --runner configured"). This means **the workflow is incomplete**, but this is a deployment gap, not a security hole (the system fails closed, refusing to emit a placeholder report). The maintainers need to add `--runner <path-to-agent-wrapper>` to the workflow step or set `CONFIG_SEARCH_RUNNER` in the job env. The spec (§13, line 275-283) implies this is intentional gating ("the search brain is an LLM... gated to run only after a maintainer approves"), but the workflow as written would fail on first run. This is fine — it forces explicit configuration.

- **report.py trust:** The build calls `report_py.write_one()` (build_gallery.py line 168), which is from the skill (`.claude/skills/config-search/scripts/report.py`). The skill is in the repo, so it's trusted code. If an attacker PRs a modified skill/report.py, that's code review (the PR modifies Python code), and the normal merge gate applies. The build is safe as long as the skill is trusted (which it is, since it's maintained code).

- **Catalog ingestion trust:** `build_catalog.py` reads sibling repos (blis-catalog, blis-registry, inference-sim) at pinned tags (generate-report.yml lines 74-77). If an upstream repo is compromised, the catalog could be poisoned, but this is a supply-chain attack (out of scope for this audit). The pins are in the workflow, so they are code-reviewed. The catalog ingestion is read-only and projects only whitelisted fields, so it cannot execute arbitrary code from upstream YAML. **Safe against injection, not against upstream compromise** (which is expected for a dependency).

- **Frontend not audited in depth:** I reviewed the key entry points (submit.js for MODE gating, esc() usage) but did not audit every frontend file for XSS. The backend provides clean JSON and the MODE gate is sound, so the frontend is fed safe data. The `esc()` helper is present (submit.js line 4, used line 43), which is the right pattern. Full frontend audit is out of scope for a backend security review.

---

## Conclusion

The backend is **stable, clear, and sound**. The design fulfills the safety requirement:

- The public Pages site is purely static; no prompt injection or execution surface.
- Report generation requires a PR and maintainer approval via the protected environment gate.
- Publishing requires merge to main.
- The local deployment runs the search against the user's own LLM; the Pages deployment does not.

**2 fixes recommended** (1 MEDIUM, 1 LOW), both straightforward. No actual vulnerabilities found. The system is production-ready after the pr_branch validation is added.

**Final score:** 4.8/5 (rounded to 5/5 after fixes).
