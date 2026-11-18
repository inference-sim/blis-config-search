# Specs waiting for the pin

A spec here names an engine kind the pinned simulator does not register yet, so the validator and
the dry run cannot read its field list and `checks.yml` does not run it. `scripts/check_pending_pin.py`
holds the condition: it fails once the pinned build registers the kind, and the spec then moves to
`experiments/`. To try one now, build a simulator that registers the kind and run
`validate_experiment.py` and `run_experiment.py --dry-run` with `--fields` set to that engine's list
(`blis fields --scenario ...`).
