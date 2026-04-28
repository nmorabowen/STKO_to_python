# STKO_to_python

Python tools for parsing and analysing OpenSees **MPCO** (HDF5)
recorder outputs produced by **STKO**. Loads `.mpco` files into lazy
views over pandas DataFrames, computes engineering aggregations
(interstory drift, envelopes, residuals, base rocking, orbits), and
produces publication figures via `NodalResultsPlotter`.

---

## Install

The library ships as a wheel attached to each
[GitHub Release](https://github.com/nmorabowen/STKO_to_python/releases).
No clone, no `git` required:

```bash
# Replace vX.Y.Z with the version from the Releases page.
pip install https://github.com/nmorabowen/STKO_to_python/releases/download/vX.Y.Z/stko_to_python-X.Y.Z-py3-none-any.whl
```

### From a clone (contributors)

```bash
pip install -e .                 # core
pip install -e ".[test]"         # pytest
pip install -e ".[bench]"        # pytest-benchmark
pip install -e ".[docs]"         # MkDocs + Material + mkdocstrings
pip install -e ".[notebook]"     # Jupyter stack for examples
```

Python 3.11+.

## Minimum working example

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet("path/to/output", "results")

# Fetch a result — returns a NodalResults view with a cached DataFrame.
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    model_stage="MODEL_STAGE[1]",
    node_ids=[1, 2, 3, 4],
)

# Engineering aggregations (forwarders to AggregationEngine).
ts    = nr.drift(top=4, bottom=1, component=1)
env   = nr.interstory_drift_envelope(component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-3)
sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)

# Plotting — per-result:
nr.plot.xy(
    y_results_name="DISPLACEMENT", y_direction=1, y_operation="Max",
    x_results_name="TIME",
)

# Plotting — dataset-level (one-shot fetch + plot):
ds.plot.xy(
    model_stage="MODEL_STAGE[1]",
    results_name="DISPLACEMENT",
    node_ids=[1, 2, 3, 4],
    y_direction=1, y_operation="Max",
    x_results_name="TIME",
)
```

## Where to read next

- **[Elements workflow guide](elements_guide.md)** — end-to-end
  walkthrough of element results: discovery, selection sets,
  integration-point access, canonical names, plotting, and pickling.
- **[ElementResults reference](api/element-results.md)** — full API
  for the element result container, including `physical_coords()`,
  `integrate_canonical()`, time-series statistics, and the new
  `plot.history()` / `plot.diagram()` / `plot.scatter()` plotters.
- **[Architecture](architecture.md)** — the layered design, what lives
  where, pickle compatibility, and the per-phase refactor history.
- **[API reference](api/index.md)** — class-by-class docs generated
  from the package's docstrings.

## Executable examples

Three runnable scripts — one per fixture — each focused on capabilities
unique to that element family:

| Script | Fixture | Highlights |
|---|---|---|
| [Elastic frame](examples/elastic_frame.md) | `elasticFrame` (single partition) | Nodal aggregations, closed-form beam forces, canonical names, plotting |
| [Quad-frame shells](examples/quad_frame_shell.md) | `QuadFrame_results` (2 partitions) | 4-IP shells, `physical_coords()`, `integrate_canonical()`, scatter plot |
| [Solid + fiber beam](examples/solid_mixed.md) | `solid_partition_example` (2 partitions) | 8-IP Brick volume integration, fiber `at_ip()`, beam moment diagram |

The [Usage tour](examples/usage_tour.md) covers all major API
surfaces in one script (nodal + element + plotting + pickle).

## Building these docs locally

```bash
pip install -e ".[docs]"
mkdocs serve               # hot-reload at http://127.0.0.1:8000
mkdocs build               # static site in ./site
```
