#!/bin/sh
# Start every check on a pull request the pin-bump automation pushed to with GITHUB_TOKEN.
#
#   sh scripts/dispatch_pr_checks.sh BRANCH PR BASE
#
# A push or a PR opened with GITHUB_TOKEN starts no workflow, but a workflow_dispatch by
# GITHUB_TOKEN does, and its runs report on BRANCH's head commit, which is what branch protection
# reads. A dispatched run is never held for approval ("action_required"): that gate applies to
# pull_request runs, never to workflow_dispatch. GH_TOKEN needs actions: write.
#   checks.yml        python, experiments, web (required on main)
#   pin-compat.yml    gate (required on main); on a pin branch it then dispatches pin-ready.yml
#   blis.yml          blis checks (description X14, commit policy, workflow lint, ...)
#   private-names.yml scan
#   pages.yml         the PR's preview
# GitHub dispatches only a workflow whose default-branch copy has a workflow_dispatch trigger, so a
# workflow that is not dispatchable there yet (the PR that adds the trigger) is skipped with a
# warning; any other failed dispatch fails this script.
set -eu
branch="$1"; pr="$2"; base="$3"
case "$pr" in ''|*[!0-9]*) echo "not a PR number: $pr" >&2; exit 1;; esac
repo="${GITHUB_REPOSITORY:?}"

dispatchable() {
  gh api -H "Accept: application/vnd.github.raw" \
    "repos/$repo/contents/.github/workflows/$1?ref=$base" 2>/dev/null | grep -q '^  workflow_dispatch:'
}

failed=""
run() {
  wf="$1"; shift
  if ! dispatchable "$wf"; then
    echo "::warning::$wf has no workflow_dispatch on $base yet, so it cannot be dispatched; it runs once this lands"
    return 0
  fi
  gh workflow run "$wf" --repo "$repo" --ref "$branch" "$@" || failed="$failed $wf"
}
run checks.yml -f base="$base"
run pin-compat.yml -f base="$base" -f pr="$pr"
run blis.yml -f pr="$pr"
run private-names.yml -f pr="$pr"
run pages.yml -f pr="$pr"
[ -z "$failed" ] || { echo "::error::could not dispatch:$failed" >&2; exit 1; }
echo "dispatched the checks of PR $pr on $branch"
