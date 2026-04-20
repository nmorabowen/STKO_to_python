# STKO_to_python refactor — implementation prompt

Copy the block below into a fresh Claude Code (or equivalent coding agent)
session rooted at `C:\Users\nmora\Github\STKO_to_python`, on branch
`refactor/oop-architecture-proposal`. The prompt is self-contained — it
tells the agent what to read, what to do, what not to do, and how to ship
the work in small, reviewable pieces.

---

## The prompt

You are implementing a staged refactor of the `STKO_to_python` library.
The canonical specification is `docs/architecture-refactor-proposal.md` on
this branch (`refactor/oop-architecture-proposal`). **Read that file in
full before writing any code.** It defines the target architecture, the
class hierarchy, the file layout, the performance strategy, the
compatibility rules, and the phased migration plan. This prompt is a
harness around that spec; the spec is authoritative.

### Non-negotiable constraints

1. **Verbose, explicit OOP.** Every class defines `__init__`, `__repr__`,
   `__slots__`, and public-method docstrings. No clever metaclass tricks,
   no implicit attribute creation, no `**kwargs` soup. Collaborators are
   passed through named constructor parameters.
2. **No `@dataclass`, no mixins.** Ever. Replace existing dataclasses
   with plain classes using `__slots__`. Share behavior through abstract
   base classes (single inheritance via `abc.ABC`) or through composed
   helper objects — never through multi-inheritance mixins.
3. **Performance-first.** Defaults in the refactor favor throughput.
   `Hdf5PartitionPool` is on by default (`pool_size = min(16, n_partitions)`),
   the query engine caches MultiIndex and ID arrays by default, the
   result LRU is on by default (`cache_size = 32`), and dimension
   columns use `pd.Categorical`. Users who need the old semantics opt
   out explicitly. No per-row Python loops in any hot path.
4. **Hard backward compatibility.** Every public import that works on
   `main` today must continue to work after every phase. Public class
   names (`MPCODataSet`, `Nodes`, `Elements`, `NodalResults`,
   `ElementResults`, `Plot`, `MPCOResults`, `MPCO_df`, `HDF5Utils`,
   `ModelInfo`, `CData`, `Aggregator`, `StrOp`, `H5RepairTool`,
   `AttrDict`, `NodalResultsPlotter`, `ModelPlotSettings`) stay
   importable from their current paths. Public method signatures do
   not change; new parameters are added only with safe defaults.
   Renamed classes keep their old names as aliases; old modules become
   thin compat shims that emit `DeprecationWarning` on import.
5. **Python 3.11+.** Target CPython 3.11 as the floor. Use
   `typing.Self`, `match`/`case`, `tomllib`, and
   `functools.cached_property` freely. 3.12-only features
   (`typing.override`, PEP 695) are used behind
   `sys.version_info >= (3, 12)` fences.
6. **Pickle stability.** `NodalResults` keeps its `__module__` and
   `__qualname__` across the refactor so existing pickles load cleanly.
   A tolerant `__setstate__` drops unknown fields with a debug log.

If you are ever tempted to violate one of these constraints, stop and
ask instead.

### Working discipline

- **One phase per PR.** Do not interleave phases. Finish phase N (code,
  tests, docs, green CI) before touching phase N+1. Commit messages name
  the phase (e.g., `Phase 1: introduce Hdf5PartitionPool`).
- **Baseline tests first.** Before any phase of refactor work, land the
  golden-fixture test suite described in §9 of the proposal. It pins
  current behavior so every subsequent commit can be verified against
  it. The fixture lives in `tests/fixtures/` and is a single small
  `.mpco` file checked in with LFS or plain-bytes depending on size.
- **No file deletions except the two dead files.** Phase 0 deletes
  `src/STKO_to_python/nodes/nodes copy.py` and
  `src/STKO_to_python/nodes/nodes.bak.py`. Every other existing module
  stays, possibly as a compat shim.
- **Logging, not `print`.** Every touched module gets
  `logger = logging.getLogger(__name__)` at the top. Existing `print`
  calls inside touched files convert to `logger.info` / `logger.debug`.
  `self.verbose=True` on `MPCODataSet` is routed to
  `logger.setLevel(logging.INFO)` on the dataset's own logger.
- **Every new class gets a unit test.** Placed under
  `tests/unit/<layer>/test_<class>.py`. Tests cover the constructor,
  the happy-path method, one edge case, and `__repr__`.
- **Every new class gets a docstring** that states: (1) the one-line
  responsibility, (2) the constructor arguments and their types, (3)
  thread-safety assumptions, (4) performance notes if non-obvious.
- **No new dependencies without a justification comment.** Everything
  in `pyproject.toml` must have a one-line reason next to it.
- **Benchmarks run on every phase.** `tests/bench/` runs under
  `pytest-benchmark` and is tracked. A regression >25% on any
  benchmark blocks the PR.

### The phases

Execute the phases in order. Do not start a phase until the previous
one is green on CI.

**Phase 0 — housekeeping.**

- Delete `src/STKO_to_python/nodes/nodes copy.py` and
  `src/STKO_to_python/nodes/nodes.bak.py`.
- Bump `pyproject.toml` `requires-python` from `>=3.8` to `>=3.11`,
  add 3.11/3.12/3.13 trove classifiers, and update the CI matrix.
- Add a module-level `logger` to every file under `src/STKO_to_python/`
  that calls `print()`. Replace `print` calls in
  `core/dataset.py`, `MPCOList/MPCOResults.py`, `model/model_info.py`,
  and `utilities/h5_repair_tool.py` with `logger.info`/`logger.debug`.
- Add `__enter__`/`__exit__` stubs on `MPCODataSet` that currently
  just return/close nothing (real cleanup arrives in Phase 1).
- Document the friend-method convention in `MPCODataSet`'s class
  docstring (which `_*` methods are intended for managers vs. truly
  internal).
- Land the golden-fixture tests, the notebook smoke test, the pickle
  round-trip test, and the format-policy test (§9 of the proposal).

**Phase 1 — Layer 1 (partition pool + format policy).**

- Add `src/STKO_to_python/io/partition_pool.py` with `Hdf5PartitionPool`
  (see §3.1 of the proposal). Default `pool_size = min(16, n_partitions)`.
  Expose `open()`, `with_partition()`, `close_all()`.
- Add `src/STKO_to_python/format/policy.py` with `MpcoFormatPolicy`
  and `src/STKO_to_python/format/gauss.py` with `GaussPointMapper`
  (opt-in via `coords="natural" | "global"`, default `"natural"`).
- Route every existing `h5py.File(...)` call in the library through
  the pool. Old behavior is preserved at `pool_size=0`.
- `MPCODataSet.__enter__`/`__exit__` now call `pool.close_all()` on
  exit.
- Benchmarks: wire up `tests/bench/` with single-fetch (cold/warm) and
  dataset-construction suites.

**Phase 2 — Layer 2 (query engines + selection resolver).**

- Add `src/STKO_to_python/selection/resolver.py` with
  `SelectionSetResolver` (composed by both managers — not a mixin).
- Add `src/STKO_to_python/query/base_query_engine.py`,
  `nodal_query_engine.py`, and `element_query_engine.py`. The base
  implements chunk-sorted fancy indexing, MultiIndex caching, and an
  LRU result cache (`cache_size=32` default). Subclasses implement
  exactly the two things that differ: component layering and the
  per-element path to the result dataset.
- Route `Nodes.get_nodal_results` and
  `Elements.get_element_results` through the engines. Public
  signatures stay identical. Add a unit test patching
  `pd.DataFrame.iterrows` to raise and running a representative
  fetch — enforces "no per-row Python loops".

**Phase 3 — Layer 3 (managers + readers).**

- Rename `Nodes` to `NodeManager` in its actual class definition;
  alias `Nodes = NodeManager` in `nodes/nodes.py` with a
  `DeprecationWarning`. Same pattern for `Elements` → `ElementManager`
  in `elements/elements.py`.
- Split `ModelInfo` into `ModelInfoReader` (model/model_info_reader.py)
  and `TimeSeriesReader` (model/time_series_reader.py). Keep
  `ModelInfo` as a compat shim.
- Split `CData` into `CDataReader` (model/cdata_reader.py). Keep
  `CData` as a compat shim.
- Introduce `BaseDomainManager(abc.ABC)` as a parent for
  `NodeManager` and `ElementManager`. Not a mixin — single inheritance,
  never instantiated.

**Phase 4 — results split + plotting consolidation.**

- Extract aggregations (drift, drift profile, envelope, residual
  drift, base rocking) from `NodalResults` into a new
  `AggregationEngine` under `dataprocess/aggregation_engine.py`.
  `NodalResults` keeps the old methods as forwarders.
- Rewrite `NodalResults` as a plain class with `__slots__`, manual
  `__init__`, manual `__repr__`, manual `__eq__`. No `@dataclass`.
- Introduce `BaseResults(abc.ABC)` under `results/base_results.py`.
  Both `NodalResults` and `ElementResults` extend it. Declare
  `fetch`, `list_results`, `list_components`, `save_pickle`,
  `load_pickle`, `plotter` as the contract.
- Consolidate plotting: `NodalResultsPlotter` becomes the canonical
  result-bound plotter. `PlotNodes` becomes a thin dataset-level
  wrapper that builds a `NodalResults` and delegates. The
  `__getattr__` forwarder on `Plot` is removed in favor of explicit
  methods.
- Add `MPCOResults.df` as a property that returns the DataFrame view
  that `MPCO_df` currently computes. Keep `MPCO_df` as a thin alias
  (deprecated).

**Phase 5 — benchmarks, docs, polish.**

- Fill out `tests/bench/` to cover all four measured-target rows in
  §6 of the proposal.
- Update `docs/` to describe the new architecture (one page per layer).
- Update `examples/` to show the new `with MPCODataSet(...)` pattern
  without breaking the old un-`with` pattern.
- Confirm every public import in the "hard compat" list still works
  and `__all__` in the package `__init__.py` is unchanged.

### Stop-work conditions

Pause and ask the user if any of these happen:

- A public method needs a signature change to make a phase work.
- A pickle round-trip breaks with no clear tolerant `__setstate__`
  fix.
- A benchmark regresses by more than 10% at the end of a phase (the
  25% CI gate is the hard limit; 10% is the point where you check
  in before continuing).
- You discover an existing public name that is not in the hard-compat
  list above but is imported by an example or notebook.
- `MPCO_df` turns out to be imported directly by a known downstream
  user — decide together whether to keep the thin alias or merge.

### Deliverables per phase

Each phase PR contains:

1. The code for that phase only.
2. New unit tests for new classes.
3. Updated benchmarks if relevant.
4. A one-paragraph PR description referencing the proposal section
   the PR implements.
5. `CHANGELOG.md` entry under `[Unreleased]`.
6. Green CI: notebook smoke test, golden-fixture test, pickle round-trip
   test, format-policy test, unit tests, benchmarks within threshold.

### Style

- `black` with default line length (88).
- `ruff` with `E`, `F`, `I`, `UP`, `B`, `SIM` rule sets. `pyupgrade`
  behavior (via `ruff`'s `UP`) is fine — we're 3.11+.
- Type hints on every public signature. Use `from __future__ import
  annotations` at the top of every module.
- Docstrings in NumPy style (consistent with the current codebase).

Do not take shortcuts. Verbose and explicit is the target; cleverness is
not.

---

## How to use this prompt

Feed the entire **## The prompt** section (from "You are implementing" to
"cleverness is not.") to a coding agent. The agent will read
`docs/architecture-refactor-proposal.md` first, then start Phase 0. If
using Claude Code, set the working directory to the repo root, checkout
`refactor/oop-architecture-proposal`, and paste the prompt. If using an
autonomous long-running agent, run it phase-by-phase with user review
between phases — this matches the "one phase per PR" discipline above.
