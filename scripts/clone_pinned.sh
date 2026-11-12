#!/bin/sh
# Clone a repository at a pin: sh scripts/clone_pinned.sh REF URL DIR
#
# REF is a full 40-hex commit (cloned without checkout, then checked out and verified), a release tag (cloned by name, shallow, and required to be exactly that tag) or a Go
# pseudo-version such as v0.13.1-0.20261111044143-f70c6c7503a4, which go.mod uses for an unreleased
# commit. A pseudo-version is no git ref: its last twelve hex digits name the commit, so the repository
# is cloned without checkout (blobless, full history) and that commit is checked out and verified.
set -eu
ref=$1; url=$2; dir=$3
if printf '%s\n' "$ref" | grep -Eq '^[0-9a-f]{40}$'; then
  # A full commit: no ref names it, so clone without checkout and check it out.
  git clone -q --filter=blob:none --no-checkout "$url" "$dir"
  git -C "$dir" checkout -q "$ref"
  test "$(git -C "$dir" rev-parse HEAD)" = "$ref" || {
    echo "$url is not at $ref -- aborting" >&2; exit 1; }
elif printf '%s\n' "$ref" | grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+-([0-9A-Za-z.]+\.)?[0-9]{14}-[0-9a-f]{12}$'; then
  sha=${ref##*-}
  git clone -q --filter=blob:none --no-checkout "$url" "$dir"
  git -C "$dir" checkout -q "$sha"
  case "$(git -C "$dir" rev-parse HEAD)" in
    "$sha"*) ;;
    *) echo "$url is not at $ref -- aborting" >&2; exit 1 ;;
  esac
else
  git clone -q --branch "$ref" --depth 1 "$url" "$dir"
  test "$(git -C "$dir" describe --tags --exact-match)" = "$ref" || {
    echo "$url is not $ref -- aborting" >&2; exit 1; }
fi
