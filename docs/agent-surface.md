# Agent surface — AGENTS.md, task guides, quirk-pattern lint

Revision 1. Not yet adversarially reviewed.

**Status:** built on branch `claude/agent-surface`, cut from `main` @ `ade3e09`
(2026-09-25), as a draft PR. No production (library) code changed. The lint reports
**0 findings** on `main`, so it has no merge-order dependency. The PR will still show
one red job, "Viewer extras resolve", which has been red on `main` since #92; this PR
does not cause it (see "Live incidents").

Method: the agent-surface playbook (a Ladruno OpenSees pilot of basecamp/omarchy's
`AGENTS.md` + task guides + CI lint). The method was ported; every rule below comes
from this repo's own incidents.

## Problem

The lessons exist but are scattered. They sit in `docs/mpco_format_conventions.md`
(17 format gotchas), `docs/architecture.md`, `CHANGELOG.md` "Fixed" sections, commit
messages, and two entries in the owner's agent memory, which lives outside the repo.
`CLAUDE.md` held only the versioning policy.

### Phase 0 baseline (measured 2026-09-25)

**Recurrences** are lessons that were written down and then bit again. There are 4.

| Lesson | Written down | Bit again | Evidence |
|---|---|---|---|
| Never merge on a red check | 953aa65 (#39, 2026-04-28): "failing on main since #36"; `docs/architecture.md` "Testing & CI" (#50) | (a) #55's own Docs check failed 2026-05-10; merged 2026-05-12; main red through #63–#65; fixed #66. (b) "Viewer extras resolve" red since #92 (2026-05-13); #93–#98 and `cut_curvature` (2026-07-28) merged on red; still red | `gh run list --workflow docs.yml / test.yml` (first episode: 12 red + 3 cancelled pushes, #27–#38) |
| A merge resolution can drop lines | 481a66e (2026-03-27): PR #20's merge lost 4 `MPCOResults` methods | ab335f2 (2026-04-28) dropped `.claude/` and `solid_partition_example/` from `.gitignore`; the fixture line came back in cecd3db, `.claude/` never did | `git show ab335f2:.gitignore`; `git status` in the main checkout lists `.claude/` as untracked |
| Tag every release | `CLAUDE.md` versioning policy (27d704d, 2026-04-28) | 1.6.0 bumped (f7e2b62) but never tagged; the `[1.7.0]` changelog section has neither a bump nor a tag | `git ls-remote --tags origin` |

**Viewer**, counted separately:

- 1 recurrence: episode (b) of "never merge on a red check".
- 0 viewer fix, revert or regression commits: `git log --no-merges` over
  `src/STKO_to_python/viewer`, `tests/viewer` and `docs/viewer*` returns 18 commits,
  all additive (Phases 0–3.0e).
- 0 memory entries about the STKO_to_python viewer; the viewer entries in memory are
  about apeGmsh's viewer.
- The viewer docs hold design rules (coupling table, no silent fallback, no actor
  recreation). No incident is recorded against any of them.

**Gate: pass.** Recurrences exist, so Phases 1–3 apply.

## Shape

1. **`AGENTS.md` is the single source.** The old `CLAUDE.md` moved verbatim to the
   top of `AGENTS.md`, and `CLAUDE.md` is now the one line `@AGENTS.md`. Additions: a
   task-guide table, a map, exact install/test/docs commands with their traps,
   working rules, "Lessons from incidents", and PR rules. The links that pointed at
   `CLAUDE.md` (`CHANGELOG.md`, `docs/architecture.md`) now point at `AGENTS.md`.
   *Accept:* the old text is a verbatim substring of `AGENTS.md`, and its only
   heading ("Versioning policy") is still present, so the `#versioning-policy` anchor
   still resolves.
2. **Memory lessons moved into the repo (2).** "Trust that the existing library
   worked" and "stko_results_examples folder" are now `AGENTS.md` "Working rules".
   The `tests/conftest.py` comment that cited the memory file now cites `AGENTS.md`.
   The memory files themselves were left untouched. Other memory directories mention
   STKO or MPCO only as STKO the software, as the OpenSees recorder, or as a tool
   that other projects call. None of them holds a lesson about this repo.
   *Accept:* no pointer into `memory/` remains in the tree.
3. **Three task guides**, chosen from the merged-PR mix. `main` has 85 PR merges;
   of the 76 on its first-parent line, 23 touched the reader packages, 19 the
   viewer, 10 bumped the version, 15 were docs-only, and 4 touched cuts. The guides are `.claude/skills/stko-mpco-reader`, `stko-viewer`
   and `stko-release`, each at most 100 lines and pointing into the archive by quoted
   heading. The viewer guide is the playbook's "a UI gets its own visual-verification
   guide", adapted to what this repo can actually run: off-screen Agg/PyVista,
   `scene.snapshot()`, a reference-vs-candidate PNG diff, programmatic input (there is
   no Qt UI yet), and cleanup by PID. `.gitignore` now has `.claude/*` +
   `!.claude/skills/`, which also restores the ignore rule ab335f2 dropped.
   *Accept:* every quoted heading exists (see Results), and the viewer recipe runs.
4. **End-user skill kept separate.** `stko-to-python-SKILL.md` gains a scope note
   ("changing the library is out of scope; see AGENTS.md"), and `stko-to-python.skill`
   was re-zipped so its `SKILL.md` stays byte-identical to the root file (LF line
   endings, `evals.json` unchanged). The guides and `AGENTS.md` say the reverse.
   *Accept:* the extracted zip entry equals the normalized root file.
5. **`ci/check_quirk_patterns.py`** (stdlib `ast`; about 2 s wall for 113 files on
   the owner's Windows machine) and its
   self-test `ci/test_check_quirk_patterns.py` (54 cases) run in a new job,
   "Quirk-pattern lint", in `test.yml`. The self-test runs first and the lint last.
   The repo had no static job to append to, and a separate job neither hides nor
   waits on the pytest matrix. The self-test lives in `ci/` deliberately:
   `testpaths = ["tests"]` keeps it out of the library suite, which would otherwise
   run it three times in the matrix. `main` has no branch protection, so no
   required-check name changes.
   - **Q1 encoding.** Flags text-mode `open()` / `io.open()` / `Path(...).open()` /
     `.read_text()` / `.write_text()` calls with no encoding. Incident: #57
     (1f4c4c4), CHANGELOG 1.3.0 "Fixed". The Ubuntu CI runs a UTF-8 locale, so no
     test can see this bug; only a static check can.
   - **Q2 attrs.** Flags `int()` / `float()` applied to `X.attrs[...]` or
     `X.attrs.get(...)`, or to a local name bound only to one of those. Incident:
     294b5fd. numpy 2.4.4 raises `TypeError` on `int(np.array([5]))` (verified
     locally).
   - Unreadable cases stay silent: a variable mode, `*args` / `**kwargs`, a foreign
     `.open` receiver, or a name that is rebound or bound in an enclosing scope.
     Waivers take the form `# stko-lint: encoding-ok|attrs-ok <reason ≥12 chars>`.
     A stale waiver is itself a finding.
   *Accept:* the mutation gate below.

## Rejected approaches

- **A lint for "logging, not `print`"** (proposal §2). There are 27 live `print()`
  calls in 10 library functions, some behind `verbose`. The only incident (#57) was
  a consistency fix, not a bug. The rule would flag production code this WP must not
  touch, and waiving it wholesale would be noise.
- **A lint for `pd.concat(copy=...)`** (294b5fd, part 2). The strict
  `DeprecationWarning` filter already fails the tests on it; that is how it was
  found.
- **A lint for broken doc links.** `mkdocs build --strict` in `docs.yml` already is
  that gate. The recurrence was merging on red, not a missing check.
- **A lint for viewer coupling** (`docs/viewer/01-architecture.md` §11). The tree
  has 0 violations, surveyed with `grep` on imports under `viewer/layers`,
  `viewer/core` and `cuts`, and the history records no violation. No incident, no
  rule.
- **An L3-style check that guide pointers still resolve.** This repo's guides have
  no rot incident yet, so it has no pre-fix commit to test against. The pointers
  were checked by hand instead (Results).
- **Q1 over `tests/`.** 7 `tcl.write_text(...)` calls in
  `tests/unit/cuts/test_per_layer_shell.py` write ASCII-only literals, so flagging
  them would be noise. The incident lived in `src/`. Scope: `src/STKO_to_python/`.
- **A `@dataclass` rule** (proposal §2). The tree now uses frozen dataclasses on
  purpose (`ElementInfo`, `BeamProfile`, `LayerInfo`, viewer styles), so the rule is
  no longer live.
- **A section-cuts guide.** 4 PRs plus 1 direct merge, and no incident or
  recurrence. Revisit when cuts work resumes.
- **Fixing the red viewer-extras job here.** It is a CI-environment change to the
  viewer and needs its own reviewed PR. The fix was not attempted and not verified.
- **`pip install -e .` into the system Python for local runs.** That would change
  the owner's environment. `PYTHONPATH=src` does the same job for one command.
- **Branch protection on `main`.** That is a repository setting and the owner's
  call (Open questions).

## Results (2026-09-25)

**Mutation acceptance.** The lint was run on `git archive` trees of the real commits.

| Run | Tree | Expected | Got |
|---|---|---|---|
| Q1 pre-fix | `d2eb7bb` (parent of 1f4c4c4, #57) | flags `cdata_reader.py` | `src/STKO_to_python/model/cdata_reader.py:45` flagged, exit 1 |
| Q1 fix | `1f4c4c4` | clean | 0 findings, exit 0 |
| Q2 pre-fix | `bc5b676` (parent of 294b5fd) | flags `model_info.py` | 4 findings: `model_info.py:471` and `:518`, `int()` and `float()` at each |
| Q2 fix | `294b5fd` | clean | 0 findings, exit 0 |
| Live | `main` @ `ade3e09` (113 files) | — | 0 findings |

**History sweep.** The lint ran on every one of the 321 commits reachable from `main`
that contain `src/STKO_to_python`. Across all of them it flagged only the two incident
sites. Q1 hit `model/cdata.py`, renamed `cdata_reader.py` in #49, from ab18114
(2025-05-02) to e586d57 (2026-05-09); every commit from 1f4c4c4 on is clean. Q2 hit
`model/model_info.py` from ab18114 to bc5b676. The sweep also found 3 WIP commits from
2025 whose files did not parse; the lint reports those as `PARSE`, it does not swallow
them. Historical noise: nil.

**The self-test fails when the lint is broken.** 8 one-line mutations of the lint,
each run against the 54-case self-test:

| Mutation | Failing cases |
|---|---|
| binary-mode exemption removed | 4 |
| Q2 name-binding removed | 2 |
| stale-waiver check removed | 2 |
| reason-length check removed | 1 |
| egg-info / `__pycache__` skip removed | 2 |
| `*args` / `**kwargs` silence removed | 2 |
| ambiguous-binding silence (`all` → `any`) | 1 |
| waiver on the line above ignored | 1 |

**Viewer visual recipe, validated on a real change.** `ds.plot.mesh()` and
`ds.plot.deformed_shape(step=5, scale=10)` were rendered on `elasticFrame/results`
from `38fdcc5~1` (before the #89 MeshLayer rewire) and from `19f2231` (after #90).
Both PNGs are pixel-identical (`np.array_equal`), which confirms the rewires'
"byte-identical" claim on this fixture. Step 0 vs step 5 differ by 84 px, so the
stale-state check has signal. `deformed_s5.png` was opened and inspected.

**Pointer check (scripted once, not in CI).** All 32 headings or bold lead-ins that
the three guides quote exist: 11 in `mpco_format_conventions.md`, 4 in
`architecture.md`, 3 in `viewer/01-architecture.md`, 9 in `AGENTS.md`, and 1 each in
`architecture-refactor-proposal.md`, `ElementResults.md`, `api/canonical-names.md`,
`viewer/00-roadmap.md` and `viewer/02-porting-from-apegmsh.md`.

**Repo gates on the new files.** Library suite: `PYTHONPATH=src MPLBACKEND=Agg python -m
pytest tests -q` gives 1565 passed, 105 skipped (no pyvista, heavy fixtures absent),
the same before and after. Self-test: 54 passed. The repo configures no ruff, pyright
or mypy. `ruff check` (default rules) and `ruff format --check` are clean on `ci/`
anyway. `mkdocs build --strict` passes.

## Live incidents — merge order

- **Lint:** none. Q1 and Q2 are clean on `main`, so nothing has to merge first.
- **CI (not a lint finding):** "Viewer extras resolve" (`test.yml`) has been red since
  #92. `PyVistaBackend.snapshot` → `pyvista.Plotter.screenshot` → `render` segfaults
  (exit 139) on the headless Ubuntu runner, in
  `tests/viewer/backends/pyvista/test_backend.py::test_snapshot_returns_rgb_array`
  (log of run 30335284188). The log of #92's own run is no longer retrievable; that
  run failed in the same step. As long as this job is red, CI validates no PyVista
  code. It should be fixed in its own PR, after adversarial review, before the next
  viewer PR.

## Phase 6 — measurement

Baseline: 4 recurrences (above). After about 10 work packages, count the review and
post-merge findings that match an entry already in `AGENTS.md` "Lessons from
incidents" or `docs/mpco_format_conventions.md`. If the count doesn't drop, stop
investing in guides here. Keep the lint regardless.

## Open questions

- Should `main` get branch protection with required checks (Test matrix, Docs,
  Quirk-pattern lint)? It would have stopped all three merged-on-red episodes.
- The maintained end-user skill (the separate `stko-to-python-skills` repo) was not
  touched, so it lacks the scope note. The in-repo copy is an older snapshot of it.
- Stale docs, left as found: `docs/architecture.md` says "Current release: v1.1.0";
  `docs/ElementResults.md` shows the deprecated `utilities.gauss_points` import; the
  `CHANGELOG.md` link references stop at 1.8.0; and
  `src/STKO_to_python/STKO_to_python.egg-info/` has been tracked since ab18114
  despite the `*.egg-info/` rule.
- The system Python's editable `STKO_to_python` install (1.6.0) points at a deleted
  worktree of the older clone. That is the owner's environment, so it was left as is
  (`AGENTS.md` trap 1).
- `docs/viewer/00-roadmap.md` asks for image-diff tests of the rewired plots. They
  don't exist; the viewer guide makes the diff a manual step instead.
