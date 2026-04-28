# MPCODataSet

Top-level entry point. Construct one per recorder output; the dataset
wires together the partition pool, readers, resolvers, managers, and
query engines.

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(
    r"C:\path\to\results",  # directory containing .mpco / .cdata files
    "results",              # recorder base name (without extension)
    name="RC_Building",     # optional label
    verbose=False,
)
```

---

## Key attributes

| Attribute | Type | Description |
|---|---|---|
| `ds.model_stages` | `list[str]` | Model stage names, e.g. `['MODEL_STAGE[1]']` |
| `ds.node_results_names` | `list[str]` | Available nodal result names |
| `ds.element_results_names` | `list[str]` | Available element result names |
| `ds.unique_element_types` | `list[str]` | Decorated element type names |
| `ds.element_types` | `dict` | Full breakdown: `{element_types_dict, unique_element_types}` |
| `ds.nodes_info` | `dict` | Node index `{array, dataframe}` |
| `ds.elements_info` | `dict` | Element index `{array, dataframe}` |
| `ds.selection_set` | `dict` | Named selection sets from `.cdata` |
| `ds.number_of_steps` | `int` | Total recorded steps |
| `ds.time` | `pd.DataFrame` | Time array per stage |
| `ds.nodes` | `NodeManager` | Nodal result manager |
| `ds.elements` | `ElementManager` | Element result manager |
| `ds.plot` | plot facade | Dataset-level XY plotter |

---

## Logging / print helpers

All `print_*` methods emit at `logging.INFO` level on the module logger.
Enable with `verbose=True` on construction, or configure logging
beforehand:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### `print_summary`

Emits a full overview: partitions, stages, nodal results, element
results, element types, node/element counts, selection sets.

```python
ds.print_summary()
```

### `print_model_stages`

```python
ds.print_model_stages()
# INFO  Number of model stages: 1
# INFO    - MODEL_STAGE[1]
```

### `print_nodal_results`

```python
ds.print_nodal_results()
# INFO  Number of nodal results: 3
# INFO    - ACCELERATION
# INFO    - DISPLACEMENT
# INFO    - REACTION
```

### `print_element_results`

```python
ds.print_element_results()
# INFO  Number of element results: 2
# INFO    - force
# INFO    - section.force
```

### `print_element_types`

Emits each result name and the decorated element types that carry it:

```python
ds.print_element_types()
# INFO  Number of unique element types: 1
# INFO    - force
# INFO      - 5-ElasticBeam3d[1000:1]
```

### `print_unique_element_types`

```python
ds.print_unique_element_types()
# INFO  Number of unique element types: 1
# INFO    - 5-ElasticBeam3d[1000:1]
```

### `print_selection_set_info`

```python
ds.print_selection_set_info()
# INFO  Selection set: 1
# INFO  Selection Set name: roof_diaphragm
```

---

## Context manager

`MPCODataSet` supports the context-manager protocol to ensure the
partition pool is released:

```python
with MPCODataSet(r"C:\results", "Recorder") as ds:
    nr = ds.nodes.get_nodal_results(...)
```

---

## API reference

::: STKO_to_python.core.dataset.MPCODataSet
