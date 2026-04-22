# STKO_to_python

Python tools for parsing and analysing OpenSees MPCO (HDF5) recorder
outputs produced by **STKO**. Loads `.mpco` files into lazy views over
pandas DataFrames, computes engineering aggregations (interstory drift,
envelopes, residuals, base rocking, orbits), and produces publication
figures via `NodalResultsPlotter`.

## Install

Grab the wheel from the [latest GitHub Release](https://github.com/nmorabowen/STKO_to_python/releases/latest)
and install it with `pip`. No clone, no `git` required:

```bash
# Replace vX.Y.Z with the version from the Releases page.
pip install https://github.com/nmorabowen/STKO_to_python/releases/download/vX.Y.Z/stko_to_python-X.Y.Z-py3-none-any.whl
```

Upgrading is the same command pointed at a newer release URL — each
release is a distinct version, so `pip install --upgrade <url>` works
naturally.

### From a clone (contributors)

```bash
pip install -e .
pip install -e ".[test]"       # pytest
pip install -e ".[bench]"      # pytest-benchmark for bench/
pip install -e ".[docs]"       # MkDocs + Material + mkdocstrings
pip install -e ".[notebook]"   # Jupyter stack for examples/notebooks
```

Python 3.11+.

## Minimum working example

```python
from STKO_to_python import MPCODataSet

# Load a recorder output (single-partition or multi-partition MP).
ds = MPCODataSet("path/to/output", "results")

# Fetch a result — returns a NodalResults view with a cached DataFrame.
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    model_stage="MODEL_STAGE[1]",
    node_ids=[1, 2, 3, 4],
)

# Engineering aggregations live on NodalResults (forwarders to a
# shared AggregationEngine — see docs/architecture.md).
ts   = nr.drift(top=4, bottom=1, component=1)        # pandas.Series
env  = nr.interstory_drift_envelope(component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-3)
sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)

# Plotting — two flavours:
#   1. Per-result (preferred for repeated plots off the same fetch):
nr.plot.xy(
    y_results_name="DISPLACEMENT", y_direction=1, y_operation="Max",
    x_results_name="TIME",
)

#   2. Dataset-level convenience (fetch + plot in one call):
ds.plot.xy(
    model_stage="MODEL_STAGE[1]",
    results_name="DISPLACEMENT",
    node_ids=[1, 2, 3, 4],
    y_direction=1, y_operation="Max",
    x_results_name="TIME",
)
```

## Pickle stability

`NodalResults` pickles from older releases load cleanly into the
current class. Unknown fields are dropped with a DEBUG log at
`STKO_to_python.results.nodal_results_dataclass`; missing optional
fields leave attributes unset; `_aggregation_engine` is a class-level
singleton so it is always resolved after unpickle. See
`docs/architecture.md` → "Pickle compatibility" for details.

## Architecture

Layered: HDF5 access → query/aggregation engines → domain managers
→ facade. No upward reference crosses a layer boundary. See
[`docs/architecture.md`](docs/architecture.md) for the diagram, the
per-phase refactor history, and the back-compat name preservation
contract.

## Tests and benchmarks

```bash
pytest tests/ -q                 # full test suite (unit + integration)
pytest bench/ -q                 # benchmarks (requires the `bench` extra)
```

Test suite is **349 green** as of Phase 5. Benchmarks are opt-in and
scoped to `bench/` so the default `pytest tests/` command doesn't pay
their collection overhead.
