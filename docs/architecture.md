# STKO_to_python — Architecture

**Status:** living document. Updated as the refactor lands.
**Source:** `docs/architecture-refactor-proposal.md` is the original
design spec; this page summarises what is actually in the tree today
and where it diverged from the proposal.

---

## Layered design

Five concrete groupings, top to bottom. Data flows **down** the stack
on init (managers read through readers, readers read through the
partition pool) and **up** on query (the facade calls a manager, which
delegates to a query engine, which pulls from the partition pool through
the format policy). No upward reference ever crosses a layer boundary.

```
          +---------------------------------+
  layer 4 |  Facade (backward-compat API)   |   MPCODataSet, MPCOResults,
          |                                 |   NodalResults, ElementResults,
          |                                 |   Plot  (unchanged public names)
          +---------------------------------+
                          |
                          v
          +---------------------------------+
  layer 3 |  Domain managers + readers      |   NodeManager, ElementManager,
          |                                 |   ModelInfoReader, CDataReader,
          |                                 |   TimeSeriesReader
          +---------------------------------+
                          |
                          v
          +---------------------------------+
  layer 2 |  Query / aggregation engines    |   NodalResultsQueryEngine,
          |                                 |   ElementResultsQueryEngine,
          |                                 |   SelectionSetResolver,
          |                                 |   AggregationEngine
          +---------------------------------+
                          |
                          v
          +---------------------------------+
  layer 1 |  HDF5 access + MPCO format      |   Hdf5PartitionPool,
          |                                 |   MpcoFormatPolicy,
          |                                 |   format/  (Gauss catalog,
          |                                 |    shape functions, IP layout)
          +---------------------------------+
```

### Layer 1 — HDF5 access + MPCO format

- **`Hdf5PartitionPool`** ([`io/partition_pool.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/src/STKO_to_python/io/partition_pool.py))
  wraps the set of `.mpco` partitions. LRU of `h5py.File` handles with
  default size `min(16, n_partitions)`; explicit `close_all()` and
  context-manager support. Thread-unsafe by design (h5py is not
  thread-safe without SWMR).
- **`MpcoFormatPolicy`** ([`io/format_policy.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/src/STKO_to_python/io/format_policy.py))
  centralises MPCO quirks — shell vs beam fiber keyword, `GP_X`
  detection, staged-construction semantics.
- **`format/` package** — the Gauss-point catalog, integration
  primitives, shape functions, and the natural→physical coordinate
  mapping. Added in PR #48 (relocated from `utilities/`). The package
  exposes registries (`ELEMENT_IP_CATALOG`, `SHAPE_FUNCTIONS`) plus
  free functions (`get_ip_layout`, `compute_physical_coords`,
  `compute_jacobian_dets`); there is no wrapper class.

### Layer 2 — Query / aggregation

- **`SelectionSetResolver`** ([`selection/resolver.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/src/STKO_to_python/selection/resolver.py))
  owns the `{name → ids, id → name, id → ids}` maps built from `.cdata`.
  Shared by `NodeManager` and `ElementManager` via composition, not
  mixin.
- **`NodalResultsQueryEngine`** / **`ElementResultsQueryEngine`**
  ([`query/`](https://github.com/nmorabowen/STKO_to_python/tree/main/src/STKO_to_python/query))
  handle `DataFrame` assembly with a normalised MultiIndex. Caching
  shim around `manager._fetch_*_uncached`; LRU on by default
  (size 32). The actual read logic still lives on the manager — the
  engine layer is a caching + selection-resolution wrapper.
- **`AggregationEngine`** ([`dataprocess/aggregation.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/src/STKO_to_python/dataprocess/aggregation.py))
  holds every engineering aggregation that `NodalResults` previously
  carried (`drift`, `residual_drift`, `interstory_drift_envelope`,
  `story_pga_envelope`, `roof_torsion`, `base_rocking`,
  `asce_torsional_irregularity`, `orbit`, and the private
  `_resolve_story_nodes_by_z_tol` helper). Stateless — attached as
  a class-level singleton on `NodalResults`.

### Layer 3 — Managers & readers

- **`NodeManager`** / **`ElementManager`** — same public surface as
  the legacy `Nodes` / `Elements` (which remain importable as
  aliases). Take their collaborators through named constructor
  params; no globals, no service locators.
- **`ModelInfoReader`**, **`CDataReader`**, **`TimeSeriesReader`** —
  pure readers. Load once, expose read-only views.

### Layer 4 — Facade

- **`MPCODataSet`** — unchanged signature. Constructs the partition
  pool, format policy, readers, resolvers, managers, and query
  engines in a deterministic order. Exposes the old attributes
  (`self.nodes`, `self.elements`, `self.model_info`, `self.cdata`,
  `self.plot`, `self.time`, `self.model_stages`, ...) as properties
  backed by the new internals. Context-manager support via
  `__enter__` / `__exit__` closes the partition pool deterministically.
- **`NodalResults`** — slotted view since PR #47. Holds a reference
  to a `NodalResultsQueryEngine` (for future re-fetches) and
  pre-resolved IDs. Exposes `fetch`, `list_results`, `list_components`,
  `save_pickle`, `load_pickle`, `plot`, and forwarders for every
  engineering method that now lives on `AggregationEngine`. Pickle
  stable (`__module__` / `__qualname__` unchanged) with a tolerant
  `__setstate__` that drops unknown keys at DEBUG level and respects
  the `__slots__` discipline.
- **`ElementResults`** — analogous result view; carries Gauss-point
  layout (`gp_natural`, `gp_weights`, `gp_xi`, `element_node_coords`)
  so that `physical_coords()` and `jacobian_dets()` resolve through
  the `format/` package without the dataset round-trip.
- **`Plot`** — minimal dataset-level facade. `ds.plot.xy(...)` fetches
  a `NodalResults` and delegates rendering to `NodalResultsPlotter.xy`.
  Use `nr.plot.*` for repeated plotting off one fetched result; use
  `ds.elements.get_element_results(...).plot.*` for elements.
- **`MPCOResults.df`** — Phase 4.5 accessor. Same instance as the older
  `create_df` attribute; `.df` is the preferred spelling. The
  `MPCO_df` *class* itself is still the canonical implementation —
  only the access pattern was unified.

---

## Reality checks vs the proposal

Four decisions where what landed differs from `architecture-refactor-proposal.md`,
each made deliberately during implementation review:

### 1. No `GaussPointMapper` class

The proposal called for a `GaussPointMapper` class that wraps
per-element shape functions and exposes a `coords="natural"|"global"`
flag at the query level. What landed instead:

- Two registry dicts (`ELEMENT_IP_CATALOG`, `SHAPE_FUNCTIONS`) keyed
  by element class tag.
- Free functions `compute_physical_coords()` / `compute_jacobian_dets()`
  that dispatch on `geom_kind` ∈ `{"line", "shell", "solid"}`.
- A separate-method API on `ElementResults` (`physical_coords()`,
  `jacobian_dets()`) instead of a flag that mutates fetch return shape.

The class wrapper would have added construction ceremony with no
shared state to carry; the registries already give us per-tag
dispatch. The "coords flag changes return shape" idea was an
anti-pattern. Both calls were left out intentionally.

### 2. Plotter merge already done

The proposal's Phase 4.4 was "merge `PlotNodes` and
`NodalResultsPlotter`." By the time the refactor reached this step,
`PlotNodes` had already been removed and the result-bound plotter
pattern (`NodalResults.plot`, `ElementResults.plot` returning a
plotter instance) was in place. The proposal's target shape was
reached through the natural evolution of plotting PRs (#32, #37, #38,
#39) rather than a dedicated merge PR.

### 3. `MPCO_df` collapse already done

The proposal's "collapse `MPCO_df` into `MPCOResults.df` accessor
with a deprecation alias" was implemented as Phase 4.5: the `.df`
property on `MPCOResults` returns the same instance as the older
`.create_df` attribute. The `MPCO_df` class stays — the collapse
was about the access pattern, not the class. No further deprecation
work was needed.

### 4. No `BaseDomainManager(abc.ABC)`

The proposal envisioned an abstract base class collapsing shared
behaviour between `NodeManager` and `ElementManager`. A measured
audit in Phase 5 found ~25 lines of real duplication, mostly
`_sort_step_keys` (12 lines, byte-identical) and the time-stitching
pattern (~12 lines). Everything else either diverges in semantics
(`_normalize_stages` defaults differ) or is genuinely
manager-specific (~95% of `ElementManager` handles heterogeneous
integration rules, bucket validation, GP metadata, Z-level filtering
that have no node analogue). A base class to capture 25 lines of
boilerplate would force readers to chase three places (base + two
subclasses) instead of two — net negative for clarity. The item was
dropped.

If the duplicated helpers grow over time, the simpler refactor is
to lift `_sort_step_keys` (and any other byte-identical helpers)
into a small shared module rather than an abstract base class.

---

## Back-compat contract

Every import that worked at v1.0.0 continues to work at v1.1.0+.
Public names preserved at the package surface (no warning):
`MPCODataSet`, `Nodes`, `Elements`, `ModelInfo`, `CData`,
`NodalResults`, `ElementResults`, `MPCOResults`, `MPCO_df`, `Plot`,
`ModelPlotSettings`, `MetaData`, `NodeManager`, `ElementManager`,
`ModelInfoReader`, `CDataReader`, `ModelMetadata`, `PlotSettings`.

Deprecation policy uses the **PEP 562 module-level `__getattr__`**
pattern: only the *deep* import path emits a warning, and only when
the deprecated name is actually accessed. Plain `import` of the shim
module stays silent.

| Import | After v1.1.0 |
|---|---|
| `from STKO_to_python import Nodes` | quiet (top-level alias of `NodeManager`) |
| `from STKO_to_python.nodes import NodeManager, Nodes` | quiet |
| `from STKO_to_python.nodes.node_manager import NodeManager` | quiet (canonical) |
| `from STKO_to_python.nodes.nodes import NodeManager` | quiet (re-export through shim) |
| `from STKO_to_python.nodes.nodes import Nodes` | **DeprecationWarning** |

Same shape for `Elements` / `ElementManager`, `ModelInfo` /
`ModelInfoReader`, `CData` / `CDataReader` (Group B, PR #49) and for
`MetaData` / `ModelMetadata`, `ModelPlotSettings` / `PlotSettings`
(Group A, PR #46), and for the relocated `format/` modules (PR #48):

| Deprecated path | Canonical path |
|---|---|
| `STKO_to_python.core.dataclasses.MetaData` | `STKO_to_python.core.metadata.ModelMetadata` |
| `STKO_to_python.plotting.plot_dataclasses.ModelPlotSettings` | `STKO_to_python.plotting.plot_settings.PlotSettings` |
| `STKO_to_python.utilities.gauss_points.*` | `STKO_to_python.format.gauss_points.*` |
| `STKO_to_python.utilities.shape_functions.*` | `STKO_to_python.format.shape_functions.*` |
| `STKO_to_python.nodes.nodes.Nodes` | `STKO_to_python.nodes.node_manager.NodeManager` |
| `STKO_to_python.elements.elements.Elements` | `STKO_to_python.elements.element_manager.ElementManager` |
| `STKO_to_python.model.model_info.ModelInfo` | `STKO_to_python.model.model_info_reader.ModelInfoReader` |
| `STKO_to_python.model.cdata.CData` | `STKO_to_python.model.cdata_reader.CDataReader` |

The legacy class name resolves to the same canonical class object
in every case — `isinstance` and pickle compatibility preserved.

---

## Pickle compatibility

Since PR #47, `NodalResults` is slotted; `__setstate__` tolerates:

- **Unknown keys** (from an older class layout). Dropped silently
  with a DEBUG record at logger
  `STKO_to_python.results.nodal_results_dataclass`. See
  `_PICKLE_FIELDS` for the current set of persisted slots.
- **Missing optional keys** (e.g. `time`, `name`, `plot_settings`).
  The slot stays unset; accessing it later raises `AttributeError`
  rather than a cryptic unpickling failure.
- **A state without `df`**. View rebuild is skipped; `_views` stays
  empty.

`_aggregation_engine` is a class-level singleton and is always
resolved after unpickle, regardless of state-dict contents.
`__module__` and `__qualname__` are pinned by
`tests/unit/test_public_api.py::test_pickle_module_qualname_pins`.

---

## Testing & CI

- **`pytest tests/`** — 628 unit + integration tests. Runs on every
  PR via `.github/workflows/test.yml` across Python 3.11 / 3.12 /
  3.13 (PR #45).
- **`pytest bench/`** — 10 benchmarks via `pytest-benchmark`. Opt-in
  through the `[bench]` extra. Runs on every PR via
  `.github/workflows/bench.yml` and uploads `benchmark.json` as an
  artifact for later regression analysis (PR #44).
- **mkdocs strict build** — every PR builds the docs site under
  `--strict`; broken cross-references fail the build.

The strict-warning filter
(`pyproject.toml [tool.pytest] filterwarnings`) elevates every
`DeprecationWarning` from the library itself to an error during
testing. Tests that exercise legacy paths wrap the import in
`pytest.warns(DeprecationWarning)`; everything else stays silent
under the filter.

---

## Versioning policy

Tags on `main` follow semver:

- **MAJOR** (`vX.0.0`) — breaking changes to the public API.
- **MINOR** (`v1.X.0`) — new backward-compatible features.
- **PATCH** (`v1.x.Y`) — bug fixes, docs, tests, internal refactors.

The release flow is documented in
[`CLAUDE.md`](https://github.com/nmorabowen/STKO_to_python/blob/main/CLAUDE.md):
bump `pyproject.toml` in the release-bearing PR, then tag the merge
commit on `main` and push the tag. Tags are lightweight unless a
real GitHub release with artifacts is being cut.

Current release: **v1.1.0** ([release page](https://github.com/nmorabowen/STKO_to_python/releases/tag/v1.1.0)).

---

## Running the benchmarks

```bash
pip install -e ".[bench]"
pytest bench/ -q
```

Benchmarks live under `bench/` (outside `tests/`). Numbers on the
tiny `elasticFrame` fixture are regression guards, not the
100-partition targets in the refactor proposal §6. When a larger
fixture is checked in, the same benches scale up without code change.

Bench summary (elasticFrame, for reference):

| Bench | Mean |
|---|---|
| `setstate_with_unknown_keys` | ~5 μs |
| `fetch_warm_cache_hit` | ~10 μs |
| `fetch_warm_different_component` | ~10 μs |
| `resolve_story_nodes` | ~70 μs |
| `pickle_loads_only` | ~330 μs |
| `pickle_roundtrip` | ~500 μs |
| `orbit` | ~970 μs |
| `drift_pairwise` | ~1.0 ms |
| `interstory_drift_envelope` | ~3.2 ms |
| `fetch_cold` | ~30 ms |
