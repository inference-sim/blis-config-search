#!/bin/sh
# Check out the pinned inference-sim's substrate/ (the catalog and registry the scripts read files
# from): sh scripts/clone_stack.sh DIR
#
# Since M3.2 blis-substrate is a directory of inference-sim, so there is no second repository to
# clone: the catalog and registry are DIR/substrate/catalog and DIR/substrate/registry at pins.env's
# BLIS_SHA, the same commit the binary is built from. Only those two directories are checked out
# (a sparse, blobless, one-commit fetch), and the commit is verified.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:?usage: clone_stack.sh DIR}
sha=$(sed -n 's/^BLIS_SHA=//p' "$root/pins.env")
printf '%s\n' "$sha" | grep -Eq '^[0-9a-f]{40}$' || { echo "pins.env BLIS_SHA=$sha is not a 40-hex commit" >&2; exit 1; }
git init -q "$dir"
git -C "$dir" remote add origin https://github.com/inference-sim/inference-sim.git
git -C "$dir" sparse-checkout set --no-cone /substrate/catalog /substrate/registry
git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$sha"
git -C "$dir" checkout -q FETCH_HEAD
test "$(git -C "$dir" rev-parse HEAD)" = "$sha" || { echo "$dir is not at $sha -- aborting" >&2; exit 1; }
