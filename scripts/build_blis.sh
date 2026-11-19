#!/bin/sh
# Build the pinned simulator: sh scripts/build_blis.sh [DIR]   (DIR defaults to upstream)
#
# DIR is a checkout of inference-sim at pins.env's BLIS_REF; the binary is written to DIR/blis.
#
# The build is stamped with the release tag. A Result records the build it came from, and the
# simulator calls a build "released" only when `cmd.buildRelease` carries a plain vMAJOR.MINOR.PATCH
# tag, set by the linker: Go cannot stamp it itself, because inference-sim's replace directives
# forbid `go install ...@version`. An unstamped build writes data.build "unreleased" and
# released_data false in every Result, and run_experiment then calls the report experimental and
# refuses to write it to reports/. The tag is read from pins.env, never typed twice.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
dir=${1:-upstream}
ref=$(sed -n 's/^BLIS_REF=//p' "$root/pins.env")
case "$ref" in
  v[0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "pins.env BLIS_REF=$ref is not a plain release tag" >&2; exit 1 ;;
esac
cd "$dir"
go build -ldflags "-X github.com/inference-sim/inference-sim/cmd.buildRelease=$ref" -o blis main.go
