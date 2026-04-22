# API reference

Class-by-class documentation generated from the package's docstrings
via [mkdocstrings](https://mkdocstrings.github.io/). Every page below
resolves its signatures and docstrings directly from the sources under
`src/STKO_to_python/`.

## Layer 4 — facade

- **[MPCODataSet](mpco-dataset.md)** — top-level dataset, one per
  recorder output. Holds managers, readers, and query engines.
- **[NodalResults](nodal-results.md)** — view over result data with
  forwarders for every engineering aggregation.
- **[MPCOResults](mpco-results.md)** — multi-case container with the
  `.df` accessor for MPCO-specific DataFrame extractors.

## Layer 3 — plotting

- **[Plotting](plotting.md)** — `Plot` dataset-level facade and
  `NodalResultsPlotter` result-bound plotter.

## Layer 2 — engines

- **[Query engines](query-engines.md)** — `NodalResultsQueryEngine`,
  `ElementResultsQueryEngine`, shared base class, LRU cache.
- **[AggregationEngine](aggregation-engine.md)** — stateless
  engineering-aggregation engine. Every `NodalResults.drift(...)` /
  `.interstory_drift_envelope(...)` / etc. forwards here.

## Layers not exposed here

The HDF5 layer (`Hdf5PartitionPool`, `MpcoFormatPolicy`) is internal;
see [`architecture.md`](../architecture.md) for the design notes.
