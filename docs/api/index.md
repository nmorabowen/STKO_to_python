# API reference

Class-by-class documentation generated from the package's docstrings
via [mkdocstrings](https://mkdocstrings.github.io/). Every page below
resolves its signatures and docstrings directly from the sources under
`src/STKO_to_python/`.

## Layer 4 — facade

- **[MPCODataSet](mpco-dataset.md)** — top-level dataset, one per
  recorder output. Holds managers, readers, and query engines.
- **[NodalResults](nodal-results.md)** — nodal result view with
  forwarders for every engineering aggregation.
- **[ElementResults](element-results.md)** — element result container
  with Gauss-point integration, canonical names, plotting, and
  time-series statistics. Pickle-portable.
- **[MPCOResults](mpco-results.md)** — multi-case container with the
  `.df` accessor for MPCO-specific DataFrame extractors.

## Layer 3 — plotting

- **[Plotting](plotting.md)** — `Plot` dataset-level facade and
  `NodalResultsPlotter` for nodal result time histories.
- **[ElementResults → plot](element-results.md#elementresultsplotter)**
  — `ElementResultsPlotter` with `history()`, `diagram()` (beam
  moment/shear/axial), and `scatter()` (shell/solid contour-style).

## Layer 2 — engines

- **[Query engines](query-engines.md)** — `NodalResultsQueryEngine`,
  `ElementResultsQueryEngine`, shared base class, LRU cache.
- **[AggregationEngine](aggregation-engine.md)** — stateless
  engineering-aggregation engine. Every `NodalResults.drift(...)` /
  `.interstory_drift_envelope(...)` / etc. forwards here.

## Canonical names

- **[Canonical names](canonical-names.md)** — engineering-friendly
  quantity aliases (`axial_force`, `stress_11`, `membrane_xx`, …) and
  the regex rules that strip column suffixes.

## Layers not exposed here

The HDF5 layer (`Hdf5PartitionPool`, `MpcoFormatPolicy`) is internal;
see [`architecture.md`](../architecture.md) for the design notes.
