#!/bin/sh
# Build the pinned simulator: sh scripts/build_blis.sh [DIR]   (DIR defaults to upstream)
#
# DIR is a checkout of inference-sim at pins.env's BLIS_SHA; the binary is written to DIR/blis.
#
# A release is stamped with its tag. A Result records the build it came from, and the simulator
# calls a build "released" only when `cmd.buildRelease` carries a plain vMAJOR.MINOR.PATCH tag,
# set by the linker: Go cannot stamp it itself, because inference-sim's replace directives forbid
# `go install ...@version`. The tag is read from pins.env, never typed twice.
#
# While BLIS_REF is a commit (the M3.2 stack head, before the release is tagged) there is no tag to
# stamp and none is invented: the build is unreleased, every Result says data.build "unreleased" and
# released_data false, and run_experiment calls a report priced by it experimental and refuses to
# write it to reports/. The pin moves to the tag once M3.2 is tagged, and this script then stamps it.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-upstream}
ref=$(sed -n 's/^BLIS_REF=//p' "$root/pins.env")
cd "$dir"
case "$ref" in
  v[0-9]*.[0-9]*.[0-9]*)
    go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRelease=$ref" -o blis main.go ;;
  *)
    printf '%s\n' "$ref" | grep -Eq '^[0-9a-f]{40}$' || { echo "pins.env BLIS_REF=$ref is neither a plain release tag nor a 40-hex commit" >&2; exit 1; }
    echo "build_blis: BLIS_REF is the commit $ref, not a release tag: the build is unreleased and its reports experimental" >&2
    go build -o blis main.go ;;
esac
