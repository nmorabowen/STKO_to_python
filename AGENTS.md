# STKO_to_python — working rules for agents

This is a helper class to parse results obtained from opensees using MPCO recorders.

The class works with STKO outputs

## Versioning policy

We follow semver and tag releases on `main`:

- **MAJOR** (`vX.0.0`) — breaking changes to the public API.
- **MINOR** (`v1.X.0`) — new backward-compatible features (new methods, new result types, new plot helpers).
- **PATCH** (`v1.x.Y`) — bug fixes, docs, tests, internal refactors with no API change.

When merging a PR (or a batch of related PRs) that warrants a release:
1. Bump `version` in `pyproject.toml`.
2. After the PR merges, tag the merge commit on `main` (`git tag vX.Y.Z <sha> && git push origin vX.Y.Z`).
3. Tags are lightweight unless a real GitHub release with artifacts is being cut.

## Task guides — read the matching one before starting

Short checklists for the kinds of work this repo's history is made of. Each item
points into a lessons doc and quotes the heading to grep for; the guides do not
replace those docs.

| Doing this | Read first |
|---|---|
| Changing how `.mpco` / `.cdata` data is read: a new element class or result bucket, the META parser, the Gauss catalog, `CDataReader`, time series, managers / query engines, canonical names | [`.claude/skills/stko-mpco-reader/SKILL.md`](.claude/skills/stko-mpco-reader/SKILL.md) |
| Anything under `src/STKO_to_python/viewer/` or `tests/viewer/`, or a `ds.plot.*` entry point that routes through the viewer | [`.claude/skills/stko-viewer/SKILL.md`](.claude/skills/stko-viewer/SKILL.md) |
| Bumping the version, editing `CHANGELOG.md`, tagging a release | [`.claude/skills/stko-release/SKILL.md`](.claude/skills/stko-release/SKILL.md) |

The mechanical lessons are enforced by `python ci/check_quirk_patterns.py` (job
"Quirk-pattern lint" in `.github/workflows/test.yml`). When a lesson names a
greppable pattern AND has a documented incident, add a rule there — with its
pre-fix/fix mutation evidence in `docs/agent-surface.md` — instead of adding
another paragraph here.

**Out of scope for these guides:** the end-user skill (`stko-to-python-SKILL.md`,
its packaged copy `stko-to-python.skill`, and the maintained copy in the separate
`stko-to-python-skills` repo). That skill teaches how to *use* the library; the
guides above are about *changing* it. Keep them apart.

## Map

- `src/STKO_to_python/` — the package (src layout). Layers, back-compat contract,
  pickle contract, fixture table: `docs/architecture.md`. The design rules the
  refactor follows (composition over mixins, explicit constructors, logging not
  `print`, "format assumptions are named, not assumed"):
  `docs/architecture-refactor-proposal.md` §2.
- MPCO on-disk format lessons: `docs/mpco_format_conventions.md`.
- Viewer design: `docs/viewer-refactor-directives.md` and `docs/viewer/00-03`.
- Plans and design notes live in `docs/` (there is no ADR folder). This agent
  surface and its evidence: `docs/agent-surface.md`.
- `stko_results_examples/` — real STKO outputs used as test fixtures (see
  "Working rules").
- `tests/` (unit, integration, viewer), `bench/` (pytest-benchmark), `examples/`,
  `ci/` (the quirk lint and its self-test).

Don't list module contents here; read the module docstrings and `docs/architecture.md`.

## Install, test, docs — exact commands

CI (`.github/workflows/test.yml`) runs `pip install -e ".[test]"` then
`pytest tests/ -q` on Python 3.11 / 3.12 / 3.13 with `MPLBACKEND=Agg`; a second
job installs `".[viewer,test]"` and runs `pytest tests/viewer/ -q`.
`docs.yml` runs `mkdocs build --strict`; `bench.yml` runs `pytest bench/`.

Locally, from the checkout you are working in:

```bash
PYTHONPATH=src MPLBACKEND=Agg python -m pytest tests -q     # ~1 min
python -m pytest -q ci/test_check_quirk_patterns.py && python ci/check_quirk_patterns.py
mkdocs build --strict                                         # needs .[docs]
pytest bench/ -q                                              # needs .[bench]
```

Traps:

1. **Test the checkout you are in.** `python -c "import STKO_to_python; print(STKO_to_python.__file__)"`
   must print a path inside this checkout. An editable install made from another
   checkout or worktree silently tests other code (on 2026-09-25 the system
   Python's editable install still pointed at a deleted worktree of an older
   clone). Use `PYTHONPATH=src`, or a venv with `pip install -e .`.
2. **The heavy fixtures are gitignored** (`Test_NLShell`, `dispBeamCol`,
   `forceBeamCol`, `solid_partition_example`); their tests skip, and `-ra`
   prints why. In a worktree under `.claude/worktrees/`,
   `tests/conftest.py::_resolve_examples_dir` falls back to the main checkout's
   copy, but only when the main checkout has them. Read the skip lines: a
   skipped suite is not a green suite.
3. **A `DeprecationWarning` raised from library code is an error in tests**
   (`filterwarnings` in `pyproject.toml`). pandas' `Pandas4Warning` subclasses
   it (294b5fd).
4. **`mkdocs build --strict` fails on a relative link that leaves `docs/`** (into
   `src/` or `examples/`, #39) and on a link that a file move broke (#66). Point
   at code with an absolute GitHub URL or a backticked path.
5. **PyVista tests skip without the `[viewer-3d]` extra** (`pytest.importorskip`),
   so a local run without it has not exercised the 3-D backend. See the viewer
   guide.

## Working rules

Moved from the owner's agent memory into the repo on 2026-09-25.

- **Trust that the existing library worked.** The code looks rough in places,
  but it has been run productively on real analyses. When a change surfaces a
  failure on real data, look for the delta first (numpy / pandas version,
  STKO / MPCO writer version, single- vs multi-partition run, input-format
  variant) before "fixing" working code. Keep fixes minimal and additive — wrap
  the one line (`np.asarray(x).item()`), don't rewrite the function. Don't
  "improve" working code during a refactor: backward compatibility is
  non-negotiable (`docs/architecture.md` "Back-compat contract"). Unusual
  conventions (verbose OOP, `__slots__`, explicit classes instead of
  `@dataclass` in the refactored core) are intentional; preserve them. In the
  owner's words: "the library could be programmed badly, but it did work, that
  may point the direction."
- **`stko_results_examples/` is ground-truth input data, not scratch.** Never
  delete or regenerate files there without asking. Prefer pointing a test at a
  file there over synthesizing HDF5, and ask before copying one into
  `tests/fixtures/`. Which fixtures are committed and which are gitignored:
  `docs/architecture.md` "Test fixtures". The multi-partition case is
  `elasticFrame/QuadFrame_results/`. More cases get added — don't hard-code
  `elasticFrame` as the only example.

## Lessons from incidents

Each entry carries its evidence. The task guides point here by heading.

### Never merge on a red check

It has happened three times:

1. `mkdocs build --strict` failed on every completed push to `main` from #27
   through #38 (2026-04-28: 12 red runs, 3 cancelled). #39 (953aa65) fixed the
   links into `src/` and `examples/`.
2. #55's own Docs check failed on 2026-05-10; it merged on 2026-05-12 anyway, and
   `main` stayed red through #63–#65 until #66 (cf9b3b1) fixed the links that the
   file move had broken.
3. "Viewer extras resolve" (`test.yml`) has been red since #92 (2026-05-13). The
   PyVista off-screen `snapshot` segfaults on the headless Ubuntu runner (exit
   139 in `tests/viewer/backends/pyvista/test_backend.py::test_snapshot_returns_rgb_array`).
   #93–#98 and the `cut_curvature` merge all landed on red, and `main` was still
   red on 2026-09-25.

`main` has no branch protection, so nothing stops it. Before asking for a merge,
run `gh pr checks <n>`: every check must be green. If a check was already red on
`main`, fix it in its own PR first. A job that is always red hides every new
failure in that job.

### A merge resolution can silently drop lines

- PR #20 (`nmb_WIP` → `main`, merged 2026-02-21): the merge lost
  `compute_table`, `collect_interstory_drift_envelope_pd`,
  `collect_roof_drift_df` and `drift_df` from `MPCOResults`. 481a66e restored
  them verbatim on 2026-03-27.
- ab335f2 (2026-04-28, "Merge branch 'main' into feat/mpco-layered-shells"):
  both sides had appended to `.gitignore`, and the resolution kept only one side.
  It dropped `.claude/` and `solid_partition_example/`, which #26 (26570bd) had
  added. cecd3db re-added the fixture line ("had been silently untracked").
  `.claude/` stayed un-ignored until the agent-surface PR.

After resolving a conflict, run `git diff <merge>^1 <merge>` and
`git diff <merge>^2 <merge>`. Every line that one side added must survive,
unless you meant to drop it.

### Text-mode file I/O: always pass `encoding=`

`.cdata` sidecars were opened with the locale encoding, which is cp1252 on
Windows, so a non-ASCII byte raised `UnicodeDecodeError` on the owner's machine
(#57, 1f4c4c4; CHANGELOG 1.3.0 "Fixed"). CI runs on Ubuntu with a UTF-8 locale
and cannot see this class of bug. **[lint Q1]**

### STKO scalar attrs are 1-element arrays

STKO writes `STEP` / `TIME` (and other scalar attrs) as 1-element arrays, and
`int(np.array([5]))` raises `TypeError` on numpy 2.x. ModelInfo's time series
broke this way (294b5fd). Unwrap with `np.asarray(x).item()`; see the
`io/time_series_reader.py` docstring. **[lint Q2]**

### A version bump is not a release until it is tagged

The "Versioning policy" above says to tag after merging, and it has been
skipped. `pyproject.toml` went to 1.6.0 (f7e2b62), and that was never tagged
(1.9.0, bumped in 2a40ca9, was folded into 1.12.0 on purpose). `CHANGELOG.md` has `[1.6.0]` and `[1.7.0]` sections, but no
tag exists for either, and 1.7.0 never appeared in `pyproject.toml`. The link
references at the bottom of the changelog stop at 1.8.0. Tags on origin as of
2026-09-25: v1.0.0–v1.5.0, v1.8.0 and v1.12.0. See the release guide.

## PRs and branches

- Base every PR on `main`, the default branch. For a sequence of dependent PRs,
  use `--base main` on every one of them. A PR based on another feature branch
  merges into that branch, and its commits never reach `main`.
- Work in a worktree under `.claude/worktrees/<name>`. `.gitignore` covers it,
  and the conftest fixture fallback expects that path. Don't switch branches or
  commit in the shared main checkout.
- End commit messages with the `Co-Authored-By:` line your harness specifies.
  Update `CHANGELOG.md` `[Unreleased]` in the same PR for anything a user or
  maintainer would notice.
