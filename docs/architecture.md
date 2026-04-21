# STKO_to_python — Architecture

**Status:** living document. Updated as the refactor lands.
**Source:** `docs/architecture-refactor-proposal.md` is the design
spec; this page summarises what is actually in the tree after Phases
0 through 5.

---

## Layered design

Four layers, top to bottom. Data flows **down** the stack on init
(managers read through readers, readers read through the partition
pool) and **up** on query (the facade calls a manager, which delegates
to a query engine, which pulls from the partition pool through the
format policy). No upward reference ever crosses a layer boundary.

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
  layer 2 |  Query / aggregation engines    |   NodalResultsQueryEngine,
          |                                 |   ElementResultsQueryEngine,
          |                                 |   SelectionSetResolver,
          |                                 |   AggregationEngine
          +---------------------------------+
                          |
                          v
          +---------------------------------+
  layer 1 |  HDF5 access + format           |   Hdf5PartitionPool,
          |                                 |   MpcoFormatPolicy
          +---------------------------------+
```

### Layer 1 — HDF5 access

- **`Hdf5PartitionPool`** wraps the set of `.mpco` partitions. Holds an
  LRU of `h5py.File` handles; `close_all()` + context-manager support.
  Thread-unsafe by design (h5py is not thread-safe without SWMR).
- **`MpcoFormatPolicy`** centralises MPCO quirks — shell vs beam fiber
  keyword, `GP_X` detection, staged-construction semantics.

### Layer 2 — Query / aggregation

- **`SelectionSetResolver`** owns the `{name → ids, id → name, id → ids}`
  maps built from `.cdata`. Shared by `NodeManager` and
  `ElementManager` via composition, not mixin.
- **`NodalResultsQueryEngine`** / **`ElementResultsQueryEngine`**
  handle `DataFrame` assembly with a normalised MultiIndex. LRU cache
  is on by default (size 32). Chunk-sorted fancy indexing.
- **`AggregationEngine`** holds every engineering aggregation that
  `NodalResults` previously carried (`drift`, `residual_drift`,
  `interstory_drift_envelope`, `story_pga_envelope`, `roof_torsion`,
  `base_rocking`, `asce_torsional_irregularity`, `orbit`, and the
  private `_resolve_story_nodes_by_z_tol` helper). Stateless —
  attached as a class-level singleton on `NodalResults`.

### Layer 3 — Managers & readers

- **`NodeManager`** / **`ElementManager`** — former `Nodes` / `Elements`,
  same public surface. Take their collaborators through named
  constructor params; no globals, no service locators.
- **`ModelInfoReader`**, **`CDataReader`**, **`TimeSeriesReader`** — pure
  readers. Load once, expose read-only views.

### Layer 4 — Facade

- **`MPCODataSet`** — unchanged signature. Constructs the partition
  pool, format policy, readers, resolvers, managers, and query
  engines in a deterministic order. Exposes old attributes
  (`self.nodes`, `self.elements`, `self.model_info`, `self.cdata`,
  `self.plot`, `self.time`, `self.model_stages`, ...) as properties
  backed by the new internals.
- **`NodalResults`** — thin view since Phase 4.3. Holds a reference
  to a `NodalResultsQueryEngine` (for future re-fetches) and
  pre-resolved IDs. Exposes `fetch`, `list_results`, `list_components`,
  `save_pickle`, `load_pickle`, `plotter`, and forwarders for every
  engineering method that now lives on `AggregationEngine`. Pickle
  stable (`__module__`/`__qualname__` unchanged) with a tolerant
  `__setstate__` that drops unknown keys with a DEBUG log.
- **`Plot`** — minimal dataset-level facade. `ds.plot.xy(...)` fetches
  a `NodalResults` and delegates rendering to `NodalResultsPlotter.xy`.
  Use `nr.plot.*` for repeated plotting off one fetched result.
- **`MPCOResults.df`** — Phase 4.5 accessor. Same instance as the older
  `create_df` attribute; `.df` is the preferred spelling.

---

## Phase history

Every phase kept the full test suite green and shipped as a series of
small commits. Public API unchanged throughout.

| Phase | Theme | Commits | Tests | Notes |
|---|---|---:|---:|---|
| 0 | Housekeeping — dead-file delete, `print → logging`, Python 3.11 floor | — | — | Pre-session |
| 1 | `Hdf5PartitionPool` + `MpcoFormatPolicy` | — | — | Pre-session |
| 2 | Query engines + `SelectionSetResolver` | — | — | Pre-session |
| 3 | `NodeManager` / `ElementManager` / `ModelInfoReader` renames | — | — | Pre-session |
| 4.1 | `MetaData` → `ModelMetadata` | — | — | Pre-session |
| 4.2 | `ModelPlotSettings` → `PlotSettings` | — | — | Pre-session |
| **4.3** | `NodalResults` split + `AggregationEngine` | 15 | +104 | This session |
| **4.4** | Plotting consolidation — dead `PlotNodes` removed + `ds.plot.xy()` added | 4 | +4 | This session |
| **4.5** | `MPCOResults.df` accessor | 1 | +6 | This session |
| **5.1** | Bench suite (pytest-benchmark, opt-in under `bench/`) | 3 | — | This session |
| 5.2 | Documentation + README | — | — | In progress |

Total test count after Phase 5: **349** (from 235 at start of Phase 4.3).

---

## Pickle compatibility

Since Phase 4.3, `NodalResults.__setstate__` tolerates:

- **Unknown keys** (from an older class layout). Dropped silently with
  a DEBUG record at logger `STKO_to_python.results.nodal_results_dataclass`.
  See `_PICKLE_FIELDS` on `NodalResults` for the current set of
  persisted fields.
- **Missing optional keys** (e.g. `time`, `name`, `plot_settings`).
  The attribute stays unset; accessing it later raises
  `AttributeError` rather than a cryptic unpickling failure.
- **A state without `df`**. View rebuild is skipped; `_views` stays
  empty.

`_aggregation_engine` is a class-level singleton, so it is always
resolved after unpickle regardless of what's in the state dict.

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

Current bench summary (elasticFrame, for reference):

| Bench | Mean |
|---|---|
| `fetch_warm_cache_hit` | ~7 μs |
| `resolve_story_nodes` | ~37 μs |
| `orbit` | ~445 μs |
| `drift_pairwise` | ~479 μs |
| `interstory_drift_envelope` | ~1.4 ms |
| `pickle_roundtrip` | ~1.1 ms |
| `fetch_cold` | ~16 ms |

---

## Back-compat contract

Every import that works on `main` before the refactor continues to
work. Public names preserved: `MPCODataSet`, `Nodes` (alias of
`NodeManager`), `Elements`, `NodalResults`, `ElementResults`,
`MPCOResults`, `MPCO_df`, `Plot`, `ModelPlotSettings`, `MetaData`
(alias of `ModelMetadata`). Old call sites see at most a
`DeprecationWarning` pointing at the new name — and for most renames
no warning at all, per spec §7.
