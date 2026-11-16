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
set -eu
branch="$1"; pr="$2"; base="$3"
case "$pr" in ''|*[!0-9]*) echo "not a PR number: $pr" >&2; exit 1;; esac

gh workflow run checks.yml --ref "$branch" -f base="$base"
gh workflow run pin-compat.yml --ref "$branch" -f base="$base" -f pr="$pr"
gh workflow run blis.yml --ref "$branch" -f pr="$pr"
gh workflow run private-names.yml --ref "$branch" -f pr="$pr"
gh workflow run pages.yml --ref "$branch" -f pr="$pr"
echo "dispatched every check of PR $pr on $branch"
