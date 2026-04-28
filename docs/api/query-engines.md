# Query engines

Internal engines that assemble `DataFrame`s with a normalised
`(id, step)` MultiIndex from the HDF5 partition pool. Each dataset
holds one instance per kind under `ds._nodal_query_engine` /
`ds._element_query_engine`.

!!! note "Most users never touch these directly"
    `NodalResults` and `ElementResults` are the public API.
    Query engines are documented here for contributors and advanced
    users who want to understand caching or extend the library.

---

## Architecture

```
MPCODataSet
  ├── NodeManager.get_nodal_results()
  │       └── NodalResultsQueryEngine.query()  ──► NodalResults
  └── ElementManager.get_element_results()
          └── ElementResultsQueryEngine.query() ──► ElementResults
```

Both engines inherit from `BaseResultsQueryEngine`, which provides:

- **MultiIndex construction** — `(node_id, step)` or `(element_id, step)`
- **Stage iteration** — concatenates results across multiple model stages
- **LRU result cache** — default size 32; key is `(stage, result_name, ids_hash, step_slice)`. Hit/miss logged at DEBUG level.
- **Chunk-sorted fancy-index path** — reads HDF5 rows in sorted order for efficient I/O then reorders to match the requested IDs.

---

## Cache behaviour

The LRU cache is on by default. To clear it or adjust its size:

```python
# Clear the nodal cache
ds._nodal_query_engine.cache.clear()

# Inspect cache info
print(ds._nodal_query_engine.cache.cache_info())
```

If you modify the underlying HDF5 file between calls (unusual outside
of testing), clear the cache explicitly to avoid stale data.

---

## BaseResultsQueryEngine

Template-method parent. Holds the MultiIndex / ID-axis caches,
handles stage iteration, and implements the chunk-sorted fancy-index
path used by both subclasses.

::: STKO_to_python.query.base_query_engine.BaseResultsQueryEngine

---

## NodalResultsQueryEngine

Specialises the base engine for nodal results. Resolves node IDs from
the partition pool's node index and maps HDF5 column positions to the
normalised `(node_id, step)` MultiIndex.

::: STKO_to_python.query.nodal_query_engine.NodalResultsQueryEngine

---

## ElementResultsQueryEngine

Specialises the base engine for element results. Resolves element IDs
and their partition assignments, reads the appropriate bucket from each
partition, and merges results into a single `(element_id, step)`
MultiIndex DataFrame.

Also responsible for:

- Determining the bucket shape (closed-form vs. line-station vs.
  Gauss-level continuum vs. compressed fiber) from the partition's
  META dataset.
- Fetching `element_node_coords` and `element_node_ids` from the
  node index (needed by `physical_coords()` and `jacobian_dets()`).
- Resolving `gp_xi` from the connectivity `@GP_X` attribute (line
  elements) or `gp_natural` / `gp_weights` from the static catalog
  (`utilities/gauss_points.py`) for catalogued element classes.

::: STKO_to_python.query.element_query_engine.ElementResultsQueryEngine
