# Self-hosted GitHub runners (ARC)

`pin-compat.yml`'s smoke jobs and `regen-reports.yml`'s sweeps run on `vars.REGEN_RUNS_ON`. One
AgentX replay peaks at 15-19 GiB RSS, more than a 16 GiB hosted runner has, so those jobs run on
GitHub's actions-runner-controller (`gha-runner-scale-set`) in the `jchen` namespace.

```sh
# 1. A PAT with `repo` scope (or fine-grained "Administration: write" on this repo)
oc create secret generic blis-config-search-runner-token -n jchen --from-literal=github_token=<PAT>

# 2. The controller, watching jchen only. Its CRDs and ClusterRoles are cluster-scoped.
helm install arc oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller \
  --version 0.15.1 -n jchen -f deploy/openshift/arc/controller-values.yaml

# 3. The scale set; `runs-on` names it
helm install blis-bigmem oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set \
  --version 0.15.1 -n jchen -f deploy/openshift/arc/scale-set-values.yaml

# 4. Point the workflows at it, and run 8 BLIS processes per job (the pod is sized for 8)
gh variable set REGEN_RUNS_ON --body '"blis-bigmem"'
gh variable set SWEEP_MAX_PARALLEL --body 8
```

The scale set idles at zero pods and starts one per queued job, up to `maxRunners` (16). To change
the scale, edit `maxRunners` and run `helm upgrade blis-bigmem ... -f scale-set-values.yaml` with
the same arguments as step 3. Keep `REGEN_MAX_PARALLEL` at or below `maxRunners`, or the extra jobs
just wait in the queue.

Two knobs, both repository variables:

- `REGEN_MAX_PARALLEL`: how many specs (jobs, so pods) run at once. Default 16.
- `SWEEP_MAX_PARALLEL`: how many BLIS runs execute at once inside one spec. Default 1. A run is
  single-threaded, so this is what speeds up a single slow spec. Memory scales with it (budget about
  25 GiB per run on a 64-GPU replay), so if you raise it past 8, raise the pod's `memory` too. The
  report comes out the same at any setting: rows are re-sorted into expansion order.

The repo is public, and a fork's PR runs the fork's own workflow files, so any of them can target
`blis-bigmem`. What keeps fork code off these runners is that every fork PR needs a maintainer's
approval before its workflows run (Settings -> Actions -> General -> "Require approval for all
external contributors"). Keep that setting, and don't approve a fork PR that changes
`.github/workflows/`.
