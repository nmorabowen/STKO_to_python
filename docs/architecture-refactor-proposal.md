# STKO_to_python — Architecture Refactor Proposal

**Branch:** `refactor/oop-architecture-proposal`
**Status:** Draft for review. No code changes land with this document.
**Author:** Nicolas Mora Bowen (with Claude's architectural review)
**Date:** 2026-04-19

---

## 1. Summary

This document proposes a layered, explicitly object-oriented refactor of
`STKO_to_python`. The goal is to pay down the architectural debt surfaced in
the recent review (tight coupling between `MPCODataSet` and its managers,
fragmented plotting, asymmetric nodal/element result APIs, silent MPCO format
assumptions, print-based verbosity, dead files) without breaking the existing
public API or the notebooks that drive the library today.

The refactor is constrained by four hard rules agreed up front:

1. **Verbose, explicit OOP.** Classes have real `__init__` methods, explicit
   attributes initialized in one place, docstrings on every public method,
   clear `__repr__`, and no clever metaclass tricks. "Guess what this does"
   does not appear anywhere.
2. **No `@dataclass`, no mixins.** Shared state is passed through constructors
   or held by composition. Shared behavior lives on abstract base classes
   (`abc.ABC`) that subclasses extend directly, not on multi-inheritance
   mixins. `MetaData`, `ModelPlotSettings`, and `NodalResults` — currently
   dataclass-like — become plain classes with `__slots__`.
3. **Performance-aware.** HDF5 partition handles are poolable; result fetches
   reuse sorted-index arrays; MultiIndex objects are built once per query, not
   per step; large result reads use chunk-aligned fancy indexing.
4. **Hard backward compatibility.** Every import that works today against
   `STKO_to_python` 0.1.0 continues to work. Public class names, public
   attributes, and public method signatures are unchanged. Old call sites
   see at most a `DeprecationWarning` pointing at the new name.

No module is deleted during this refactor. Old modules become thin adapters
over the new OOP core. Removal is deferred to a future release once users have
had time to migrate at their own pace.

## 2. Guiding principles

The refactor follows a small set of design rules. They are stated here
explicitly because several of them are non-obvious in Python and are easy to
walk back mid-implementation.

**Composition over inheritance, inheritance over mixins.** When two classes
need to share behavior (e.g. selection-set resolution between `Nodes` and
`Elements`), the shared behavior lives on a standalone object that each class
*holds*, not on a mixin class that each class *inherits*. When two classes
need a common interface (e.g. `NodalResults` and `ElementResults` both expose
`fetch`), they share an abstract base class with concrete, well-named
template methods, not a mixin.

**Explicit constructors, no `**kwargs` soup.** Every object takes its
collaborators through named parameters. No object pulls collaborators from a
service locator, global, or module-level singleton. This is verbose on
purpose — the wiring is the documentation.

**One owner per piece of state.** Each attribute has exactly one class that
writes it. Everything else reads. This removes the "who mutated the time
series?" class of bug that can appear when every manager has a reference to
the same dataset object and can poke its attributes.

**Protocols at the public seams, concrete classes inside.** Public
multi-implementation surfaces (the `Results` family, the `Plotter` family)
use `abc.ABC` so that static analyzers and IDEs understand them. Internal
helpers remain concrete classes.

**No `@dataclass`.** Every replacement for a dataclass uses
`__slots__` to give the same memory footprint and attribute discipline,
defines `__init__`, `__repr__`, and `__eq__` explicitly, and exposes fields
as read-only properties where it matters. `@dataclass` is convenient but it
hides the constructor and encourages the "fat dataclass with methods"
anti-pattern that `NodalResults` currently embodies.

**Logging, not `print`.** Every module gets a module-level
`logger = logging.getLogger(__name__)`. `self.verbose` continues to work
and becomes a shorthand for `logger.setLevel(logging.INFO)` on the dataset's
own logger. Users can redirect or suppress output through the standard
`logging` tree.

**Format assumptions are named, not assumed.** MPCO format quirks (0-based
Gauss-point indices, natural-coordinate `GP_X`, shell vs beam fiber keyword
swap, MODEL_STAGE staging semantics) live in a single module with a class
per concern, and every query that depends on them goes through that class.
Silent misinterpretation becomes a visible constructor choice.

## 3. Proposed class hierarchy

The refactor introduces four layers. Each layer is a concrete set of classes,
not a conceptual distinction. The layers, top to bottom:

```
          +---------------------------------+
  layer 4 |  Facade (backward-compat API)   |   MPCODataSet, MPCOResults,
          |                                 |   NodalResults, ElementResults,
          |                                 |   Plot  (unchanged public names)
          +---------------------------------+
                          |
                          v
          +---------------------------------+
  layer 3 |  Domain managers                |   NodeManager, ElementManager,
          |                                 |   ModelInfoReader, CDataReader,
          |                                 |   TimeSeriesReader
          +---------------------------------+
                          |
                          v
          +---------------------------------+
  layer 2 |  Query / aggregation engines    |   ResultsQueryEngine,
          |                                 |   SelectionSetResolver,
          |                                 |   AggregationEngine
          +---------------------------------+
                          |
                          v
          +---------------------------------+
  layer 1 |  HDF5 access + format           |   Hdf5PartitionPool,
          |                                 |   MpcoFormatPolicy,
          |                                 |   GaussPointMapper
          +---------------------------------+
```

Data flows down the stack on init (managers consult readers which consult
the partition pool) and back up on query (the facade calls a manager, which
delegates to a query engine, which pulls from the partition pool through the
format policy). No arrow goes the other direction. Nothing in layer 1 knows
layer 2 exists.

### 3.1 Layer 1 — HDF5 access and format

**`Hdf5PartitionPool`** wraps the set of `.mpco` partitions that make up one
recorder output. Today the library opens and closes `h5py.File` inside every
query; at a hundred partitions and a hundred queries per session that is ten
thousand opens. The pool keeps an optional LRU of `h5py.File` handles with an
explicit `close_all()` and context-manager support. A dataset with a small
partition count keeps the old behavior (open-per-query) by configuring the
pool size to zero; a dataset with many partitions configures a pool of, say,
sixteen handles and pays the open cost once per handle.

```python
class Hdf5PartitionPool:
    """Owns the set of .mpco HDF5 files for one recorder output.

    Thread-unsafe by design; h5py is not thread-safe without SWMR. The
    AggregationEngine uses process-level parallelism instead.
    """

    __slots__ = ("_paths", "_pool_size", "_open", "_lru")

    def __init__(self, partition_paths: dict[int, Path], pool_size: int = 0):
        ...

    def open(self, partition_idx: int) -> h5py.File: ...
    def with_partition(self, partition_idx: int) -> "PartitionHandle": ...
    def close_all(self) -> None: ...
```

**`MpcoFormatPolicy`** is the single place that knows MPCO conventions. It
answers questions like "is this element class tag a shell?", "which result
keyword maps to fiber stress for this element?", "is `GP_X` present on this
group?". Today those questions are spread across `Elements`, `ModelInfo`,
and inline string checks. Centralizing them makes format changes a one-file
edit and makes the "silent assumption" problem auditable.

**`GaussPointMapper`** converts natural Gauss-point coordinates to global
coordinates using the element's shape functions. Today the library does not
do this at all — GP coordinates stored as `GP_X` in `[-1, +1]` are passed
through as if they were global. The mapper is opt-in at the query level
(`fetch_gauss_results(..., coords="natural" | "global")`, default
`"natural"` to preserve current behavior) and is implemented per element
geometry (line, quad, hex).

### 3.2 Layer 2 — Query and aggregation engines

**`SelectionSetResolver`** is the class that fixes today's most visible
duplication. It owns the `{name -> ids, id -> name, id -> ids}` bidirectional
maps built from `.cdata` and exposes four methods:
`resolve_nodes(names=..., ids=...)`,
`resolve_elements(names=..., ids=...)`,
`list_node_sets()`, `list_element_sets()`. Both `NodeManager` and
`ElementManager` take one in their constructor; the duplicated
`_selection_set_name_for` / `_selection_set_ids_from_names` helpers move
here and the managers lose ~50 lines each.

```python
class SelectionSetResolver:
    __slots__ = ("_by_name", "_by_id", "_node_ids", "_element_ids")

    def __init__(self, cdata_reader: "CDataReader"):
        # reads once at construction; no HDF5 access after this
        ...

    def resolve_nodes(self, *, names=None, ids=None) -> np.ndarray:
        """Return a sorted, deduplicated numpy array of node IDs."""
```

**`ResultsQueryEngine`** is the class that fixes today's asymmetry between
nodal and element results. It has two subclasses, `NodalResultsQueryEngine`
and `ElementResultsQueryEngine`, each with a concrete `fetch` that returns a
`DataFrame` with a normalized MultiIndex. The subclasses share a parent that
handles shared concerns (MultiIndex construction, stage iteration, step
throttling), not through a mixin but through template-method inheritance.

**`AggregationEngine`** owns the engineering aggregations that
`NodalResults` is currently swollen with — interstory drift, drift profile,
envelope, residual drift, base rocking. Today these live as methods on
`NodalResults`, forcing the data object to know about seismic engineering.
In the refactor they become instance methods on `AggregationEngine` that
take a `NodalResults` (or `ElementResults`) as an argument. The data object
stops knowing anything about drift; the aggregation object knows nothing
about HDF5.

### 3.3 Layer 3 — Domain managers

`NodeManager` and `ElementManager` are the explicit OOP replacements for
today's `Nodes` and `Elements`. They are thicker than their predecessors
because they own the construction of their readers, resolvers, and query
engines, but they are not god objects — every piece of work delegates
downward. Their public methods (`get_nodal_results`, `get_element_results`,
`get_all_nodes_ids`, ...) preserve current signatures verbatim.

`ModelInfoReader`, `CDataReader`, `TimeSeriesReader` replace today's
`ModelInfo`, `CData`, and the ad-hoc time handling in `io/time_utils.py`.
They are "readers", not "managers" — they load once, expose read-only
views, and never mutate.

### 3.4 Layer 4 — Facade

`MPCODataSet`, `NodalResults`, `ElementResults`, `MPCOResults`, and `Plot`
retain their current import paths and public signatures exactly. Internally
they become thin adapters over the new layers. `MPCODataSet.__init__`
constructs the partition pool, format policy, readers, resolvers, managers,
and query engines in a deterministic order, and exposes the old attributes
(`self.nodes`, `self.elements`, `self.model_info`, `self.cdata`, `self.plot`,
`self.time`, `self.model_stages`, ...) as properties backed by the new
internals. The old protected-method pattern (`self.dataset._get_*`) continues
to work via compat shims that `warnings.warn(..., DeprecationWarning)` on
call.

## 4. Replacing dataclasses with explicit classes

Three of today's classes are effectively dataclasses: `MetaData`
(dataclass-flavored container), `ModelPlotSettings` (`@dataclass(slots=True)`),
and `NodalResults` (a fat dataclass-like container with engineering methods).
Each is replaced with a plain class following the same pattern.

### 4.1 `MetaData` → `ModelMetadata`

The current `MetaData` uses an `_extras` dict with `__getattr__`/
`__setattr__` to allow arbitrary attributes. The refactor keeps the flexible
bag semantics but replaces the dataclass frame with an explicit class:

```python
class ModelMetadata:
    """Free-form metadata bag attached to a dataset.

    Arbitrary keys are allowed and stored in an internal dict. This class
    intentionally does not use @dataclass so that the set of known fields
    can evolve without forcing schema migrations on pickle files.
    """

    __slots__ = ("_extras",)

    def __init__(self, **extras):
        object.__setattr__(self, "_extras", dict(extras))

    def __getattr__(self, name):
        try:
            return self._extras[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self._extras[name] = value

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._extras.items())
        return f"ModelMetadata({items})"

    def as_dict(self) -> dict:
        return dict(self._extras)
```

The public import path (`from STKO_to_python.core.dataclasses import MetaData`)
stays valid via a one-line re-export plus deprecation warning:

```python
# core/dataclasses.py (compat shim)
import warnings
from .metadata import ModelMetadata as MetaData  # noqa: F401
warnings.warn(
    "STKO_to_python.core.dataclasses.MetaData is deprecated; "
    "import ModelMetadata from STKO_to_python.core.metadata instead.",
    DeprecationWarning, stacklevel=2,
)
```

### 4.2 `ModelPlotSettings` → `PlotSettings`

The current version is already reasonable — slots, small surface, no
inheritance. The refactor re-expresses it as a plain class with explicit
properties for the settings that matter semantically (color palette, line
width, label template) and keeps `to_mpl_kwargs()` / `make_label()` as
methods. The `@dataclass` decorator drops; the behavior does not.

### 4.3 `NodalResults` → `NodalResults` (thinned) + `AggregationEngine`

The biggest structural change. `NodalResults` today is a fat object carrying
both result data and engineering operations (drift, envelope, base rocking,
pickle I/O, plotting delegation). The refactor splits it:

- **`NodalResults`** (kept under its current import path, same signature):
  a view over result data. Holds a reference to a `NodalResultsQueryEngine`
  and a handful of pre-resolved IDs. Exposes `fetch`, `list_results`,
  `list_components`, `save_pickle`, `load_pickle`, `plotter`. No engineering
  aggregations live here.
- **`AggregationEngine`** (new, in `dataprocess/aggregation.py`): owns the
  drift/envelope/residual methods. `NodalResults.drift_profile(...)` keeps
  working as a compat method that calls
  `self._aggregation_engine.drift_profile(self, ...)` under the hood.
- **`NodalResultsPlotter`** (kept, thinned): takes the `NodalResults` as a
  constructor argument, no longer reaches into its innards.

Neither class uses `@dataclass`. Both declare `__slots__` for attribute
discipline and memory. The split is invisible from outside because every
old method still exists on `NodalResults`; they become forwarders.

## 5. Replacing mixins with composition and ABCs

The review recommended a `SelectionSetResolver` mixin. On reflection, given
the "no mixins" rule, composition is the right call here anyway — a mixin
would have forced `Nodes` and `Elements` to inherit from a class whose only
job is to resolve IDs, muddying both class hierarchies. Composition reads
more cleanly:

```python
class NodeManager:
    __slots__ = ("_pool", "_reader", "_selection", "_query", "_dataset")

    def __init__(
        self,
        dataset: "MPCODataSet",
        pool: Hdf5PartitionPool,
        reader: ModelInfoReader,
        selection: SelectionSetResolver,
        query: NodalResultsQueryEngine,
    ):
        self._dataset = dataset
        self._pool = pool
        self._reader = reader
        self._selection = selection
        self._query = query
```

Shared behavior across `NodeManager` and `ElementManager` (ID parsing,
stage iteration, DataFrame formatting) lives on an abstract base class
`BaseDomainManager(abc.ABC)` with concrete helpers and abstract methods
for the things that actually differ. This is classical OOP inheritance,
not a mixin — `BaseDomainManager` is never instantiated, and the subclasses
extend it in exactly one direction.

Similarly, `NodalResults` and `ElementResults` both extend an abstract
`BaseResults(abc.ABC)` that declares the query contract. Today's `Results`
family has no shared parent at all; every extension forces duplicated
boilerplate.

## 6. Performance strategy

The current library's performance bottlenecks are the open-per-query HDF5
pattern, MultiIndex reconstruction on every `fetch`, and the absence of any
cache between the `.mpco` file and the returned DataFrame. The refactor
addresses each explicitly.

**HDF5 handle reuse.** `Hdf5PartitionPool` holds up to N open `h5py.File`
handles (N configurable, default 8). Each partition is opened once and kept
until the pool's LRU evicts it or `close_all()` is called. Benchmarks on a
100-partition file should see a 5–20× speedup on query-heavy workflows,
with zero change on single-query runs.

**Chunk-aware reads.** When fetching results for many nodes or elements, the
query engine sorts requested IDs by their on-disk position before the fancy
index, which aligns the read with HDF5's chunking and avoids seek thrash.
This is a numpy one-liner (`argsort` / `np.take`) but has to be applied
consistently; centralizing it in the query engine makes that easy.

**MultiIndex reuse.** Today every call to `fetch` rebuilds the
`(stage, node_id, step)` MultiIndex from scratch, which is expensive for
thousands of steps. The query engine caches the step axis per model stage
and the ID axis per selection set; the full MultiIndex is constructed in
one `from_product` call.

**Optional result caching.** `ResultsQueryEngine` accepts a `cache_size`
parameter (default 0 — no cache, current behavior). When set, it maintains
an LRU of `(stage, result_name, component, ids_hash) -> DataFrame`. A user
running a report that re-fetches the same displacement trace across five
plots sees four cache hits. Disabled by default because it trades memory
for compute.

**Optional parallel partition reads.** `AggregationEngine` gains a
`n_workers` parameter for the explicitly batch-ish operations (multi-case
drift profile, envelope-across-cases). Implemented with
`concurrent.futures.ProcessPoolExecutor` — not threads, because h5py is
not thread-safe. Default is serial; users opt in.

**No per-row Python loops.** The review did not find any such loops in the
hot path, but the refactor's query engine is written to make this a policy
rather than an accident. All result assembly is numpy/pandas vectorized.
This is documented in `BaseResultsQueryEngine`'s class docstring.

**Benchmarks become part of the repo.** `tests/bench/` gains three
benchmarks: single-result fetch, multi-case aggregation, pickle
round-trip. They run with `pytest-benchmark` and are tracked in CI so
regressions are visible.

## 7. Backward compatibility strategy

"Hard compat" means a user upgrading from 0.1.0 to the refactor release
changes exactly nothing in their code and sees exactly the same outputs.
The strategy has three layers.

**Import-path preservation.** Every module that exists today continues to
exist and exports the same names. Where a module's contents move to a new
location, the old module becomes a one-line re-export with a
`DeprecationWarning` at import time. Examples:

- `from STKO_to_python.core.dataclasses import MetaData` →
  compat shim in `core/dataclasses.py`, actual class at
  `core/metadata.py::ModelMetadata`, alias `MetaData = ModelMetadata`.
- `from STKO_to_python.nodes.nodes import Nodes` → compat shim,
  actual class at `nodes/node_manager.py::NodeManager`, alias
  `Nodes = NodeManager`.
- The top-level `from STKO_to_python import MPCODataSet, Nodes, Elements,
  NodalResults, ElementResults, Plot, ...` continues to work because
  `__init__.py` re-exports the same names from their new locations.

**Signature preservation.** Every public method keeps its current
signature. New parameters are added only with safe defaults (`cache_size=0`,
`coords="natural"`, `n_workers=1`) so that existing call sites continue to
produce identical results. Return types that are DataFrames stay
DataFrames; any `NodalResults` return stays a `NodalResults` (now thinned,
but with all old methods present as forwarders).

**Pickle compatibility.** `NodalResults` is commonly pickled today
(`save_pickle` / `load_pickle` and the `_LazyPickle` wrapper in
`MPCOList`). The refactor keeps `NodalResults.__module__` and
`__qualname__` unchanged — the class is defined in the same file and under
the same name, even though its internals are rewritten. Old pickles load
cleanly. A custom `__setstate__` handles the case where an old pickle has
fields (e.g. inline drift-profile caches) that the new class no longer
stores; they are silently dropped with a debug log message.

**Two CI jobs.** One runs the existing notebooks and scripts verbatim and
confirms their outputs match a recorded baseline. The other runs the new
tests. A PR cannot merge if the first job fails for any reason other than
an explicit, versioned change.

## 8. Phased migration plan

The refactor is large and should ship in five phases, each independently
valuable and independently merge-able. None of them break the public API.

**Phase 0 — housekeeping (≈2 hours).** Delete `nodes/nodes.bak.py` and
`nodes/nodes copy.py`. Replace `print` with `logging` in `core/dataset.py`,
`MPCOList/MPCOResults.py`, `model/model_info.py`, and
`utilities/h5_repair_tool.py`. Add module-level `logger =
logging.getLogger(__name__)` and route `self.verbose=True` to
`logger.setLevel(logging.INFO)`. Document the friend-method convention in
`MPCODataSet`'s class docstring. No structural changes; all tests keep
passing.

**Phase 1 — Layer 1 (partition pool + format policy) (≈1–2 days).**
Introduce `Hdf5PartitionPool` and `MpcoFormatPolicy`. Route every existing
`h5py.File(...)` call through the pool. Default `pool_size=0` preserves
current behavior. `MpcoFormatPolicy` centralizes the scattered format
checks; migrate one consumer at a time. No public API change.

**Phase 2 — Layer 2 (query engines + selection resolver) (≈2–3 days).**
Introduce `SelectionSetResolver`, `NodalResultsQueryEngine`,
`ElementResultsQueryEngine`, and the abstract base they share. Route
`Nodes.get_nodal_results` and `Elements.get_element_results` through the
engines. The managers lose duplicated code; the public API is unchanged.

**Phase 3 — Layer 3 (managers) (≈2 days).** Rename `Nodes` → `NodeManager`
and `Elements` → `ElementManager` in their actual class definitions; keep
the old names as aliases. Split `ModelInfo` into `ModelInfoReader` and
`TimeSeriesReader`. Add `BaseDomainManager` as an abstract parent.

**Phase 4 — `NodalResults` split + `AggregationEngine` + plotting consolidation
(≈3–4 days).** Extract aggregations to `AggregationEngine`. Thin
`NodalResults` to a view. Merge `PlotNodes` and `NodalResultsPlotter`
responsibilities into a single result-bound plotter with a dataset-level
convenience wrapper. Collapse `MPCO_df` into `MPCOResults` as a `.df`
accessor. Every old call site keeps working.

**Phase 5 — benchmarks + documentation (≈1–2 days).** Add
`pytest-benchmark` suite. Add Sphinx/MkDocs pages for the new layered
architecture. Deprecation warnings move from `DeprecationWarning` to
`PendingDeprecationWarning` or stay as-is depending on adoption. No API
removals this cycle.

A rough total of two working weeks of focused effort, split across five
PRs, each small enough to review in one sitting.

## 9. Testing strategy

The refactor cannot land safely without tests that pin down the current
behavior first. The proposal is to add, before any refactor phase:

1. A **golden-fixture test** using one representative `.mpco` file checked
   in under `tests/fixtures/`. Every public method of `MPCODataSet`,
   `Nodes`, `Elements`, `NodalResults`, `ElementResults`, `MPCOResults`,
   and `Plot` is exercised and its output compared against a stored
   snapshot (numpy arrays, DataFrames, matplotlib figure hashes).
2. A **pickle round-trip test** for `NodalResults` against a pickle file
   produced by the current release. This guards pickle compatibility
   across the refactor.
3. A **notebook smoke test** that executes every notebook under `examples/`
   headless and checks for zero exceptions. Not a correctness test, but a
   cheap signal.
4. A **format-policy test** that covers the cases the MPCO skill flags:
   standard vs custom integration rule, shell vs beam fiber keyword,
   staged construction with two MODEL_STAGE groups, selection sets by ID
   and by name.

Each refactor phase is gated on all four of these green. Internal
unit tests for the new layers come with each phase's PR.

## 10. Open questions

These are items that need a decision before or during implementation:

- **Default partition-pool size.** Candidate is 8. Too small for very
  large-partition files, too large for small ones. A better default might
  be `min(16, n_partitions)` set automatically on dataset construction.
- **`MPCO_df` merge vs preserve.** Collapsing it into
  `MPCOResults.df` is cleaner but breaks the "hard compat" rule if any
  user imports `MPCO_df` directly. The conservative answer is to keep it
  as a thin alias and mark it deprecated. Needs confirmation.
- **Global vs natural Gauss-point coordinates default.** The safe default
  is `"natural"` (matches current silent behavior). A case could be made
  for `"global"` in a future release with a big deprecation sign attached.
  Out of scope for this refactor but worth deciding.
- **Parallel aggregation default.** `n_workers=1` (serial) preserves
  current behavior. `n_workers=os.cpu_count()` would be a surprise.
  Leave at 1 for this refactor.
- **Python version floor.** Current `pyproject.toml` says 3.8. `__slots__`
  and `typing.Protocol` work; the proposal does not require 3.10+. If
  3.10+ is acceptable we can use `match`/`case` in the format policy,
  which reads cleanly.

## 11. What this proposal does not do

To keep scope honest, the refactor explicitly does not:

- Remove any public API.
- Change any return shape (DataFrames keep their columns and MultiIndex).
- Add new features (no new result types, no new plot kinds).
- Introduce async I/O or typing-heavy libraries (pydantic, attrs).
- Port to a different HDF5 binding (h5py stays).

Those are future releases. This one pays down debt and makes the next
feature easier to add.

---

**Appendix A — file layout after refactor**

```
src/STKO_to_python/
├── __init__.py                     # unchanged public re-exports
├── core/
│   ├── __init__.py
│   ├── dataset.py                  # MPCODataSet (facade, unchanged signature)
│   ├── metadata.py                 # ModelMetadata (new)
│   └── dataclasses.py              # compat shim re-exporting MetaData
├── io/
│   ├── __init__.py
│   ├── partition_pool.py           # Hdf5PartitionPool (new)
│   ├── hdf5_utils.py               # kept, thinned
│   ├── info.py                     # kept
│   ├── time_utils.py               # kept, may fold into time_reader
│   └── utilities.py                # kept
├── format/
│   ├── __init__.py                 # new package
│   ├── policy.py                   # MpcoFormatPolicy
│   └── gauss.py                    # GaussPointMapper
├── model/
│   ├── __init__.py
│   ├── model_info.py               # ModelInfo compat shim
│   ├── model_info_reader.py        # ModelInfoReader (new)
│   ├── time_series_reader.py       # TimeSeriesReader (new)
│   ├── cdata.py                    # CData compat shim
│   └── cdata_reader.py             # CDataReader (new)
├── selection/
│   ├── __init__.py                 # new package
│   └── resolver.py                 # SelectionSetResolver
├── nodes/
│   ├── __init__.py
│   ├── nodes.py                    # Nodes compat shim (alias of NodeManager)
│   └── node_manager.py             # NodeManager (new)
├── elements/
│   ├── __init__.py
│   ├── elements.py                 # Elements compat shim
│   ├── element_manager.py          # ElementManager (new)
│   └── element_results.py          # ElementResults (thinned)
├── results/
│   ├── __init__.py
│   ├── base_results.py             # BaseResults abstract class (new)
│   ├── nodal_results_dataclass.py  # compat shim (name preserved)
│   ├── nodal_results.py            # NodalResults (thinned, no @dataclass)
│   ├── nodal_results_info.py       # kept
│   └── nodal_results_plotting.py   # NodalResultsPlotter (thinned)
├── query/
│   ├── __init__.py                 # new package
│   ├── base_query_engine.py        # BaseResultsQueryEngine (abstract)
│   ├── nodal_query_engine.py       # NodalResultsQueryEngine
│   └── element_query_engine.py     # ElementResultsQueryEngine
├── dataprocess/
│   ├── __init__.py
│   ├── aggregator.py               # kept
│   └── aggregation_engine.py       # AggregationEngine (new)
├── plotting/
│   ├── __init__.py
│   ├── plot.py                     # Plot (facade, unchanged)
│   ├── plot_nodes.py               # kept, thinned
│   └── plot_settings.py            # PlotSettings (no @dataclass)
├── MPCOList/
│   ├── __init__.py
│   ├── MPCOList.py                 # kept
│   ├── MPCOResults.py              # thinned; .df accessor added
│   └── MPCOdf.py                   # compat shim
└── utilities/
    ├── __init__.py
    ├── attribute_dictionary_class.py   # kept (AttrDict)
    └── h5_repair_tool.py               # kept, print→logging
```

Files deleted: `nodes/nodes copy.py`, `nodes/nodes.bak.py`, and any
`__pycache__` debris. Nothing else is removed; every old path remains
importable.

**Appendix B — example of the compat shim pattern**

```python
# nodes/nodes.py  (compat shim after refactor)
"""Compatibility shim.

The class previously named Nodes is now NodeManager and lives in
node_manager.py. This module re-exports the old name so that existing
imports (from STKO_to_python.nodes.nodes import Nodes) keep working.
A DeprecationWarning is emitted on import.
"""
from __future__ import annotations
import warnings

from .node_manager import NodeManager

Nodes = NodeManager  # public alias; unchanged behavior

warnings.warn(
    "STKO_to_python.nodes.nodes.Nodes is now NodeManager and lives in "
    "STKO_to_python.nodes.node_manager. The old name will continue to "
    "work for the foreseeable future, but new code should use NodeManager.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Nodes", "NodeManager"]
```

The same pattern applies to `core/dataclasses.py`, `model/model_info.py`,
`model/cdata.py`, `elements/elements.py`, `results/nodal_results_dataclass.py`,
and `MPCOList/MPCOdf.py`.
