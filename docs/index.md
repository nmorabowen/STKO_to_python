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
- **[Architecture](architecture.md)** — the layered design, what lives
  where, pickle compatibility, and the per-phase refactor history.
- **[API reference](api/index.md)** — class-by-class docs generated
  from the package's docstrings.
- **[Design spec](architecture-refactor-proposal.md)** — the proposal
  document that drove the 2026 refactor. Kept for historical context.

## Building these docs locally

```bash
pip install -e ".[docs]"
mkdocs serve               # hot-reload at http://127.0.0.1:8000
mkdocs build               # static site in ./site
```
