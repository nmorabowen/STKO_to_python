---
name: stko-to-python
description: >
  Helper for the STKO_to_python library — a Python package for parsing and analyzing
  OpenSees MPCO/STKO HDF5 recorder outputs. Use this skill whenever the user wants to:
  load MPCO datasets, fetch nodal or element results, use selection sets, plot
  time histories or drift profiles, aggregate results, serialize/deserialize with pickle,
  do multi-case analysis with MPCOResults, or write any code that imports from
  STKO_to_python. Also trigger when the user mentions: STKO, MPCO, mpco files,
  HDF5 recorder results, NodalResults, ElementResults, MPCODataSet, selection sets
  in an OpenSees post-processing context, drift profiles, roof drift, story drifts,
  or any post-processing of OpenSees simulation outputs stored in .mpco files.
  Even if the user just says "plot my results" or "get displacements" in a context
  where STKO/MPCO data is involved, use this skill.
---

# STKO_to_python — Usage & API Reference

This skill helps you write correct, idiomatic code using the `STKO_to_python` library.
The library parses OpenSees simulation results stored in STKO's MPCO HDF5 format and
provides high-level abstractions for accessing nodal/element data, plotting, and
statistical aggregation.

## Dependencies

```
numpy, pandas, h5py, matplotlib, scipy, tables
```

Install the library from the user's local source (it's not on PyPI):
```bash
pip install -e /path/to/STKO_to_python
```

---

## Architecture Overview

The library follows a **composite pattern** with a central coordinator:

```
MPCODataSet (main entry point)
 |- .nodes        -> Nodes         (nodal data reader)
 |- .elements     -> Elements      (element data reader)
 |- .model_info   -> ModelInfo     (metadata extraction)
 |- .cdata        -> CData         (selection set parsing)
 |- .plot         -> Plot          (visualization facade)
 |   |- .nodes    -> PlotNodes     (nodal plot helpers)
 |- .info         -> Info          (dataset statistics)
```

Users interact primarily with `MPCODataSet` and the result containers (`NodalResults`, `ElementResults`).

---

## 1. Loading a Dataset

```python
from STKO_to_python import MPCODataSet

dataset = MPCODataSet(
    hdf5_directory='/path/to/hdf5/files',   # folder with .mpco partition files
    recorder_name='results',                 # base name of the recorder
    name='MyModel',                          # optional display name
    file_extension='*.mpco',                 # default; rarely changed
    verbose=False,                           # print loading details
    plot_settings=None,                      # optional ModelPlotSettings
)
```

**What happens on init:** The class automatically discovers partition files (`results.part-0.mpco`, `results.part-1.mpco`, ...), reads model stages, result names, element types, time series, node/element indices, and selection sets. All of this is cached on the dataset object.

**Important attributes after loading:**

| Attribute               | Type             | Description                                    |
|-------------------------|------------------|------------------------------------------------|
| `dataset.model_stages`  | `list[str]`      | e.g. `['MODEL_STAGE[1]', 'MODEL_STAGE[2]']`   |
| `dataset.node_results_names` | `list[str]` | e.g. `['DISPLACEMENT', 'VELOCITY', 'ACCELERATION']` |
| `dataset.element_results_names` | `list[str]` | e.g. `['force', 'section.force', 'material.stress']` |
| `dataset.unique_element_types` | `list[str]` | e.g. `['203-ASDShellQ4', '64-DispBeamColumn3d']` |
| `dataset.time`          | `pd.DataFrame`   | MultiIndex `(MODEL_STAGE, STEP)` with `TIME` column |
| `dataset.nodes_info`    | `dict`           | Keys: `'array'` (structured ndarray), `'dataframe'` |
| `dataset.elements_info` | `dict`           | Keys: `'array'` (structured ndarray), `'dataframe'` |
| `dataset.number_of_steps` | `dict`         | `{stage_name: int}` |
| `dataset.selection_set` | `dict`           | `{set_id: {'SET_NAME': str, 'NODES': [...], 'ELEMENTS': [...]}}` |
| `dataset.results_partitions` | `dict`      | `{partition_id: file_path}` |

### Exploration Methods

```python
dataset.print_summary()              # full overview
dataset.print_model_stages()         # list stages
dataset.print_nodal_results()        # list nodal result names
dataset.print_element_results()      # list element result names
dataset.print_element_types()        # element types per result
dataset.print_unique_element_types() # unique element types
dataset.print_selection_set_info()   # selection set names and IDs
```

---

## 2. Fetching Nodal Results

```python
nr = dataset.nodes.get_nodal_results(
    results_name='DISPLACEMENT',          # str or list[str] for multiple results
    model_stage='MODEL_STAGE[3]',         # which analysis stage
    node_ids=[1, 2, 3],                   # explicit node IDs (optional)
    selection_set_id=1,                   # OR use a selection set ID
    selection_set_name='TopNode',         # OR use a selection set name (case-insensitive)
)
```

**Selection rules** — provide exactly ONE of:
- `node_ids` — explicit list of node IDs
- `selection_set_id` — integer ID from `dataset.selection_set`
- `selection_set_name` — string name (case-insensitive matching)
- None of them — returns results for ALL nodes

**Return type:** `NodalResults` object (see Section 4).

**`results_name`** can be a single string or a list of strings. When a list is provided, all results are fetched in a single pass over the HDF5 files (more efficient than separate calls).

---

## 3. Fetching Element Results

```python
er = dataset.elements.get_element_results(
    results_name='globalForces',
    model_stage='MODEL_STAGE[1]',
    element_ids=[10, 20, 30],             # optional; omit for all elements
)
```

Additional element query methods:

```python
# Filter elements by Z coordinate levels
elems = dataset.elements.get_elements_at_z_levels(z_levels=[0.0, 3.5, 7.0])

# Combined selection set + Z filtering
elems = dataset.elements.get_elements_in_selection_at_z_levels(
    selection_set_id=2, z_levels=[0.0, 3.5]
)

# Results filtered by selection and Z
er = dataset.elements.get_element_results_by_selection_and_z(
    results_name='force', selection_set_id=2, z_levels=[0.0]
)

# List available element results
dataset.elements.get_available_element_results()
```

**Return type:** `ElementResults` object (see Section 5).

---

## 4. NodalResults Container

The `NodalResults` object is the primary result container. It wraps a pandas DataFrame with a rich access API.

### DataFrame Structure
- **Index:** `(node_id, step)` or `(stage, node_id, step)` for multi-stage
- **Columns:** `MultiIndex (result_name, component)` when multiple results, or single-level strings

### Attribute-Style Access (ResultView)

```python
# Get component 1 of DISPLACEMENT for all nodes
disp_x = nr.DISPLACEMENT[1]                    # -> pd.Series

# Get component 1 for specific nodes
disp_x = nr.DISPLACEMENT[1, [14, 25]]          # -> pd.DataFrame

# Get all components for all nodes
disp_all = nr.DISPLACEMENT[:]                   # -> pd.DataFrame

# Get all components for specific nodes
disp_all = nr.DISPLACEMENT[:, [14, 25]]         # -> pd.DataFrame
```

Components are **1-indexed** (matching OpenSees convention): 1=X, 2=Y, 3=Z for displacement-like results.

### fetch() Method

```python
# Explicit fetch (equivalent to attribute access)
data = nr.fetch(result_name='DISPLACEMENT', component=1, node_ids=[14, 25])
```

### Introspection

```python
nr.list_results()                    # -> ('ACCELERATION', 'DISPLACEMENT', ...)
nr.list_components('DISPLACEMENT')   # -> ('1', '2', '3')
```

### The .info Sub-Object

```python
nr.info.nodes_ids          # tuple of node IDs in this result
nr.info.model_stages       # tuple of stage names
nr.info.results_components # tuple of component names
nr.info.analysis_time      # float, seconds
nr.info.size               # int, bytes

# Find nearest node to a point
nr.info.nearest_node_id(points=[(0.5, 0.0, 2.0)])
```

### Serialization

```python
# Save (auto-detects .gz for compression)
nr.save_pickle('/path/to/results.pkl.gz', compress=True)

# Load
from STKO_to_python import NodalResults
nr = NodalResults.load_pickle('/path/to/results.pkl.gz')
```

---

## 5. ElementResults Container

Similar pattern to NodalResults but for element-level data.

### DataFrame Structure
- **Index:** `(element_id, step)`
- **Columns:** `val_1`, `val_2`, ..., `val_N`

### Access

```python
# Attribute-style access
forces = er.val_1                        # all elements
forces = er.val_1[[10, 20]]              # specific element IDs

# Explicit fetch
data = er.fetch(col_name='val_1', element_ids=[10, 20])
```

### Serialization

```python
er.save_pickle('/path/to/elem_results.pkl.gz', compress=True)
er = ElementResults.load_pickle('/path/to/elem_results.pkl.gz')
```

---

## 6. Plotting

The `dataset.plot.nodes` object (a `PlotNodes` instance) provides several high-level plotting methods. All return `(axes_or_figure, metadata_dict)`.

### 6.1 Generic X-Y Plot

```python
ax, meta = dataset.plot.nodes.plot_nodal_results(
    model_stage='MODEL_STAGE[3]',
    # Vertical axis
    results_name_verticalAxis='REACTION_FORCE',
    selection_set_id_verticalAxis=2,
    direction_verticalAxis=1,              # 1=X, 2=Y, 3=Z or 'x','y','z'
    values_operation_verticalAxis='Sum',   # aggregation across nodes
    scaling_factor_verticalAxis=1.0,
    # Horizontal axis
    results_name_horizontalAxis='DISPLACEMENT',
    selection_set_id_horizontallAxis=1,    # note: typo in API, double 'l'
    direction_horizontalAxis=1,
    values_operation_horizontalAxis='Sum',
    scaling_factor_horizontalAxis=1.0,
    # Cosmetics
    figsize=(10, 6),
    linewidth=1.2,
    label='My Curve',
    marker=None,
)
```

**Special horizontal axis values:** If `results_name_horizontalAxis` is `None`, it defaults to `'TIME'` (or `'STEP'` if TIME is unavailable). You can explicitly pass `'TIME'` or `'STEP'`.

**Aggregation operations** (`values_operation_*`): `'Sum'`, `'Mean'`, `'Max'`, `'Min'`, `'Std'`, `'Percentile'`, `'Envelope'`, `'Cumulative'`, `'SignedCumulative'`, `'RunningEnvelope'`.

For envelope-type operations, pass a list: `['RunningEnvelope', 'min', 'max']`.

### 6.2 Time History

```python
fig, meta = dataset.plot.nodes.plot_time_history(
    model_stage='MODEL_STAGE[5]',
    results_name='DISPLACEMENT',
    selection_set_id=4,
    direction=2,                    # Y direction
    split_subplots=True,            # one subplot per node
    scaling_factor=1.0,
    sort_by='z',                    # sort nodes by z-coordinate
    figsize=(8, 3),                 # per-subplot size
    sharey=True,
)
```

### 6.3 Roof Drift

```python
ax, meta = dataset.plot.nodes.plot_roof_drift(
    model_stage='MODEL_STAGE[5]',
    selection_set_id=4,
    direction=2,
    normalize=True,                 # drift ratio (divide by height)
    aggregate='Mean',               # how to combine nodes at same Z
)
```

### 6.4 Story Drifts

```python
fig, meta = dataset.plot.nodes.plot_story_drifts(
    model_stage='MODEL_STAGE[5]',
    selection_set_id=4,
    direction=2,
    split_subplots=True,
)
```

### 6.5 Drift Profile

```python
ax, meta = dataset.plot.nodes.plot_drift_profile(
    model_stage='MODEL_STAGE[5]',
    selection_set_id=4,
    direction=1,
)
```

### 6.6 Orbit Plot

```python
fig, meta = dataset.plot.nodes.plot_orbit(
    model_stage='MODEL_STAGE[5]',
    results_name='DISPLACEMENT',
    selection_set_id=4,
    split_subplots=True,
)
```

---

## 7. Aggregator

The `Aggregator` class provides fast 1-D aggregation over step-grouped data. It's used internally by the plotting methods but can also be used directly.

```python
from STKO_to_python import Aggregator

# Create from a NodalResults DataFrame
agg = Aggregator(nr.df, direction=1)       # direction: column name or int index

# Available operations
result = agg.compute('Sum')
result = agg.compute('Mean')
result = agg.compute('Max')
result = agg.compute('Min')
result = agg.compute('Std')
result = agg.compute('Percentile', percentile=95)
result = agg.compute('Envelope')
result = agg.compute('Cumulative')
result = agg.compute('SignedCumulative')
result = agg.compute('RunningEnvelope')
```

The aggregator groups the DataFrame by `step` and applies the operation across nodes at each step. Results are memoized for efficiency.

---

## 8. Multi-Case Analysis (MPCOResults)

For parametric studies or multiple ground motions, `MPCOResults` manages a collection of `NodalResults` objects keyed by `(model, station, rupture)` tuples.

```python
from STKO_to_python import MPCOResults

# Load from a directory of pickle files
results = MPCOResults.load_dir(
    out_dir='/path/to/pkl/files',
    style=None,      # optional plot style dict
    name='MyStudy',
)

# Iterate
for (model, station, rupture), nr in results.items():
    print(f"{model} / {station} / {rupture}")

# Filter with glob patterns
subset = results.select(
    model='*Building*',
    station='*Station1*',
    rupture=None,          # None = all
)

# Length, indexing
len(results)
nr = results['ModelA', 'Station1', 'Rupture1']

# Create analysis DataFrames
df = results.create_df.some_method(...)
```

---

## 9. Utility Classes

### HDF5Utils
```python
from STKO_to_python import HDF5Utils

f = HDF5Utils.open_file('/path/to/file.mpco', mode='r')
group = HDF5Utils.get_group(f, '/MODEL_STAGE[1]/MODEL/NODES', required=False)
data = HDF5Utils.read_dataset_as_numpy(group, 'ID')
keys = HDF5Utils.list_keys(group)
```

### H5RepairTool
```python
from STKO_to_python import H5RepairTool
# Repair corrupted HDF5 files
H5RepairTool.repair('/path/to/corrupted.mpco')
```

### AttrDict
```python
from STKO_to_python import AttrDict
ad = AttrDict({'key': 'value'})
ad.key         # -> 'value'
ad.new_key = 42
```

---

## 10. Common Workflows

### Workflow A: Load and Plot Force-Displacement

```python
from STKO_to_python import MPCODataSet
import matplotlib.pyplot as plt

model = MPCODataSet(hdf5_directory='/path/to/results', recorder_name='results')
model.print_summary()
model.print_selection_set_info()

ax, meta = model.plot.nodes.plot_nodal_results(
    model_stage='MODEL_STAGE[3]',
    results_name_verticalAxis='REACTION_FORCE',
    selection_set_id_verticalAxis=2,         # base nodes
    direction_verticalAxis=1,
    values_operation_verticalAxis='Sum',
    results_name_horizontalAxis='DISPLACEMENT',
    selection_set_id_horizontallAxis=1,      # top node
    direction_horizontalAxis=1,
)
plt.title('Force vs Displacement')
plt.show()
```

### Workflow B: Extract and Process Nodal Data

```python
model = MPCODataSet(hdf5_directory='/path', recorder_name='results')

# Fetch displacements for a selection set
nr = model.nodes.get_nodal_results(
    model_stage='MODEL_STAGE[3]',
    results_name='DISPLACEMENT',
    selection_set_name='TopNode',
)

# Access X-displacement
disp_x = nr.DISPLACEMENT[1]
print(disp_x.describe())

# Save for later
nr.save_pickle('/path/to/disp.pkl.gz')
```

### Workflow C: Multi-Stage Time History

```python
model = MPCODataSet(hdf5_directory='/path', recorder_name='results_nodes')

fig, meta = model.plot.nodes.plot_time_history(
    model_stage='MODEL_STAGE[5]',
    results_name='DISPLACEMENT',
    selection_set_id=4,
    direction=2,
    split_subplots=True,
)
plt.savefig('time_history.png', dpi=150)
```

### Workflow D: Drift Analysis

```python
model = MPCODataSet(hdf5_directory='/path', recorder_name='results_nodes')

# Roof drift
ax, _ = model.plot.nodes.plot_roof_drift(
    model_stage='MODEL_STAGE[5]',
    selection_set_id=4,
    direction=2,
)

# Drift profile (envelope across time)
ax, _ = model.plot.nodes.plot_drift_profile(
    model_stage='MODEL_STAGE[5]',
    selection_set_id=4,
    direction=2,
)
plt.show()
```

---

## 11. Known Quirks and Gotchas

1. **`selection_set_id_horizontallAxis`** — Note the double `l` in `horizontall`. This is a known typo in the API. Use it as-is.

2. **Components are 1-indexed** — DISPLACEMENT component 1 is X, 2 is Y, 3 is Z. This matches OpenSees convention, not Python's 0-based indexing.

3. **`hd5f_directory`** — The attribute on the dataset is `hd5f_directory` (transposed letters), not `hdf5_directory`. The constructor parameter is `hdf5_directory`.

4. **"No elements found"** — If the recorder only captured nodal results, element info will be empty. This is normal, not an error.

5. **Model stages are 1-indexed** — The first stage is `'MODEL_STAGE[1]'`, not `'MODEL_STAGE[0]'`.

6. **Selection sets** — Always call `dataset.print_selection_set_info()` to discover available set IDs and names before querying by selection set.

7. **Multi-partition datasets** — The library handles partitioned files automatically. Just point `hdf5_directory` to the folder containing all `.mpco` files.

8. **Time DataFrame** — `dataset.time` has a MultiIndex `(MODEL_STAGE, STEP)`. To get time for a specific stage: `dataset.time.loc['MODEL_STAGE[3]']`.

---

## Public API Summary

```python
from STKO_to_python import (
    MPCODataSet,       # Main entry point
    Nodes,             # Nodal data reader (accessed via dataset.nodes)
    Elements,          # Element data reader (accessed via dataset.elements)
    ElementResults,    # Element results container
    ModelInfo,         # Metadata extractor (accessed via dataset.model_info)
    CData,             # Selection set parser (accessed via dataset.cdata)
    Plot,              # Plot facade (accessed via dataset.plot)
    NodalResults,      # Nodal results container
    NodalResultsPlotter,  # Plotting helpers for NodalResults
    MPCOResults,       # Multi-case analysis
    MPCO_df,           # DataFrame creation for multi-case
    Aggregator,        # Statistical aggregation
    StrOp,             # Type alias for aggregation operation strings
    HDF5Utils,         # Low-level HDF5 helpers
    H5RepairTool,      # HDF5 repair utility
    AttrDict,          # Dictionary with attribute access
    ModelPlotSettings, # Plot configuration (from plotting.plot_dataclasses)
)
```
