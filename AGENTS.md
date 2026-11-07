# Agent instructions for blis-config-search

These rules apply to every AI-assisted change to this repository. Human contributors follow the
same rules; they are written here so an agent reads them first. CLAUDE.md covers commands and
architecture.

<!-- BEGIN GENERATED: invariants (go run ./tools/invariants render; do not edit) -->
## Invariants checked in blis-config-search

Generated from the invariant registry, `blis-substrate/contracts/invariants`, at the
substrate version this repository pins (design §13.1). Each line is an invariant this
repository runs a check for: its identifier, the first sentence of its statement, its
status, and each of its checks here. `ci` means the check runs in CI; `new` means it does not
yet, and the target milestone says when it must. A pull request cites the identifiers it
keeps or introduces; CI refuses an identifier that does not exist (X14).

- **X7** (partial; target ci by M4) Contracts live in blis-substrate/contracts.
  - `new`, pull_request: A lower repository refers to accuracy-dashboard only through the harness pin file
  - `new`, pull_request: Every generated file matches its generator's output
  - `new`, pull_request: No hand-copied list of another repository's facts; field lists, knobs and plugin parameters are read from their owner at the pinned version
- **X8** (new; target ci by M1) Every number records the exact versions and data it came from, and can be regenerated from them.
  - `new`, pull_request: The dashboard and committed config-search reports accept only released-data results; an experimental result is refused
  - `new`, pull_request: Every published number carries its data units
  - `new`, pull_request: Published numbers regenerate: on redistributable fixtures on every pull request, on the full data nightly
- **X12** (partial; target ci by M1) Only derived values are committed, never third-party data, with one exception: vendor config.json files whose licence is on the allow-list.
  - `new`, pull_request: Name scan in every repository, which also reads the text of every open issue each night
- **X13** (partial; target ci by M1) Every behaviour change has a test that fails without it.
  - `new`, pull_request: The corpus comparison, run by the scoring harness at the pinned digest from each repository's CI, for pull requests declared as refactors
- **X14** (new; target ci by M1) A pull request's description is accurate: the commit hashes it lists are the branch heads, it merges cleanly, and every invariant it cites exists.
  - `new`, pull_request: The description check, a shared workflow run in every repository
- **X16** (new; target ci by M1) A skipped test in a required or gate job is a failure, not a pass, unless it is on the quarantine list of X25.
  - `new`, pull_request: Each repository's CI configuration
- **X17** (partial; target ci by M1) Every repository's protected branch requires its named checks, including formatting and lint.
  - `new`, settings: The protected branch requires the repository's named checks, including formatting and lint
- **X19** (new; target ci by M2) Scripts and generators, in every language, never write an engine default.
  - `new`, pull_request: The engine-name lint (X2), run on *.py and *.sh in every repository that has scripts
  - `new`, pull_request: In every scenario generator, a test that a generated scenario contains only stated or recorded values, and workload-dependent values with their citation. The name lint cannot see a bare number under a neutral key, so the second test is the one that catches it
- **X20** (new; target ci by M4) Every commit written for BLIS is signed.
  - `new`, pull_request: A commit-policy check, required on every pull request in every repository, over the commits the pull request adds
- **X21** (new; target ci by M1) Everything a build or a release runs is pinned and traceable.
  - `new`, pull_request: A workflow lint that refuses an action not pinned by commit
  - `new`, pull_request: Pip install --require-hashes in every Python CI job
  - `new`, pull_request: A check that LICENSE and any required NOTICE exist
- **X22** (partial; target ci by M4) Every verdict (a CI result, an agent's review, or a person's answer to a review question) names the git tree hash of the exact tree it judged.
  - `new`, pull_request: The shared merge check in blis-substrate/.github/workflows, called by every repository, which compares verdict tree hashes
  - `new`, pull_request: A workflow lint that refuses ignored exit codes
  - `new`, pull_request: Each job's item count
- **X23** (new; target ci by M1) A version is written in one place.
  - `new`, pull_request: A lint in every repository, over every file type, for version literals of BLIS modules or Go that are not generated
- **X24** (new; target ci by M4) Every review finding ends in one of three outcomes before merge: fixed, filed as a linked issue, or rejected with a reason.
  - `new`, pull_request: The shared merge check, which refuses a merge while a review thread has no outcome
- **X25** (new; target ci by M1) Tests read only their declared inputs, never sibling checkouts, ignored directories or the user's environment.
  - `new`, pull_request: CI in a clean container with no sibling checkouts
  - `new`, pull_request: The quarantine list, which fails CI when an entry expires
  - `new`, pull_request: A required check that fails while main is failing and the pull request does not name the failing test's issue
- **X26** (new; target ci by M1) In documentation, every source path, symbol, issue or pull-request reference, and every statement that a feature is supported, is either generated (from surface.json, stack.json or the code) or checked against the pinned tree.
  - `new`, pull_request: A documentation reference check in every repository
- **X28** (new; target ci by M1) An issue in a BLIS repository that is cited as the reason for a refusal, for a refused or not-modelled item in a field or surface list, or in a code comment that marks a gap or a divergence, stays open.
  - `new`, nightly: A nightly job in every repository that resolves each cited issue and fails on a closed one
  - `new`, pull_request: A merge check that refuses a change closing an issue still cited elsewhere
  - `new`, nightly: The nightly job also reports an issue closed outside a merge
- **SUB-19** (partial; target ci by M1) Every document that crosses a repository boundary has one definition, in contracts/spec: the scenario, the deployment, the Result, and the per-point record the scoring harness writes.
  - `new`, pull_request: Schema validation in the applications' CI
  - `new`, pull_request: A lint in the applications that refuses a hand-written schema for these documents
- **CFG-1** (ci; target reached) One pin file names the simulator release, and every report states it.
  - `ci`, pull_request: Scripts/test_build.py (`scripts/test_build.py`)
  - `ci`, pull_request: Workflows that stop if the cloned commit differs from the pinned one (`checks.yml: pin guard on the cloned commit`)
- **CFG-2** (partial; target ci by M3) Everything after the specification is deterministic and produced by a bot.
  - `ci`, pull_request: Scripts/test_experiment.py (`scripts/test_experiment.py`)
  - `new`, pull_request: Reports arrive only through the bot's pull request
- **CFG-3** (new; target ci by M1) Settings are the named engine's own settings, read from the simulator.
  - `new`, pull_request: A check that every settings list is read from inference-sim fields at the pinned release, and that the repository holds no copy
- **CFG-4** (partial; target ci by M3) A specification ranks and checks configurations on any metric of the metric registry, at any scope (§8.7).
  - `ci`, pull_request: Regression tests: worst-instance checks (`scripts/test_experiment.py`)
  - `new`, pull_request: Regression tests: the metric registry and the overload rule
- **CFG-5** (new; target ci by M3) A search varies only settings that the engine's field list marks as modelled at the pinned release.
  - `new`, pull_request: A specification check against inference-sim fields
  - `new`, pull_request: The report schema
- **DEV-8** (new; target ci by M2) Every reported metric names its definition and scope (§8.7), and a comparison is refused unless both sides have the same definition, scope and aggregate.
  - `new`, pull_request: A test that they compare only figures with matching definitions, scopes and aggregates
- **DEV-11** (new; target ci by M3) A figure that compares configurations, or that is published as a property of BLIS, carries an interval from replications or batch means and states which uncertainty that interval covers, or states that the run has no random input and which uncertainty it therefore leaves out.
  - `new`, pull_request: The report schemas of blis-config-search and accuracy-dashboard
  - `new`, pull_request: A coverage test in which the full procedure, repeated 1,000 times on a model with a known difference, covers that difference at its stated rate
- **DEV-12** (new; target ci by M1) Each change is a verification change or a validation change by what it does (§8.9), and is judged as such.
  - `new`, pull_request: In every repository, the change class in the pull-request template, and a check that the outputs that moved match the declared class
- **COV-1** (partial; target ci by M1) One scorer produces every number, and it lives only in accuracy-dashboard: no other repository compares a simulation with a measured end-to-end benchmark.
  - `new`, pull_request: A lint that refuses code outside accuracy-dashboard that reads end-to-end benchmark units to compute an error
- **LIF-1** (new; target ci by M1) A tag names one commit forever.
  - `new`, settings: Protected tags
  - `new`, nightly: A nightly check that every tag resolves to its recorded commit
- **LIF-3** (new; target ci by M1) Only the release bot cuts releases, and only from a commit where every ci invariant passed.
  - `new`, settings: Repository settings that let only the release bot and release owners create tags
  - `new`, nightly: A nightly check that every tag made by a person is signed and listed in its release notes

### Review questions

CI adds each question below to a pull request that changes one of its paths, as a
required checklist item. A reviewer with write access answers it yes or no with a one-line reason; an agent
may propose an answer but never gives one (design §13.7).

- **X7** Does this change give a second home to a fact another repository owns? Reviewer: expertise in the changed repository. When: a pull request adds a literal list or table, or a constant that cites another repository. Added by a reviewer with the label `review:X7`.
- **DEV-12** Does this change alter what is modelled? Reviewer: expertise in the changed directory. When: a pull request is declared a verification change but moves a Result outside the byte gate. Added by a reviewer with the label `review:DEV-12`.
<!-- END GENERATED: invariants -->
