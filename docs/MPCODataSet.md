# MPCODataSet — Entry Point

`MPCODataSet` is the main entry point to your STKO/MPCO simulation data. It reads the HDF5 files produced by OpenSees MPCO recorders, indexes all nodes and elements, and exposes methods to extract results into self-contained result objects (`NodalResults`, `ElementResults`) that you can work with independently — or pickle for later use.

## Quick Start

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(
    hdf5_directory=r"C:\path\to\your\results",
    recorder_name="mpco",
    name="MyModel",          # optional display name
    verbose=True,             # print summary on load
)
```

On construction the dataset automatically:

1. Discovers all `.mpco` partition files and companion `.mpco.cdata` files.
2. Reads model stages, time series, node/element indices, selection sets, and available result names.
3. Prints a welcome banner and (if `verbose=True`) a full summary.

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hdf5_directory` | `str` | — | Path to the folder containing `.mpco` files |
| `recorder_name` | `str` | — | Base name of the recorder (e.g. `"mpco"`) |
| `name` | `str` or `None` | `None` | Display name; if `None`, derived from folder |
| `file_extension` | `str` | `"*.mpco"` | Glob pattern for partition files |
| `verbose` | `bool` | `False` | Print summary after loading |
| `plot_settings` | `ModelPlotSettings` or `None` | `None` | Custom plot settings forwarded to results |

## Key Attributes

After construction, the following attributes are populated and available:

```python
ds.model_stages            # list[str] — e.g. ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]
ds.node_results_names      # list[str] — e.g. ["DISPLACEMENT", "ACCELERATION", ...]
ds.element_results_names   # list[str] — e.g. ["globalForces", "section_deformation"]
ds.unique_element_types    # list[str] — e.g. ["203-ASDShellQ4", "2-truss"]
ds.element_types           # dict with 'unique_element_types' and 'element_types_dict'
ds.time                    # pd.DataFrame indexed by stage with TIME column
ds.number_of_steps         # dict — steps per stage
ds.selection_set           # dict — {id: {'SET_NAME': ..., 'NODES': [...], 'ELEMENTS': [...]}}
ds.results_partitions      # dict — {file_id: filepath}
ds.nodes_info              # dict — {'array': np.ndarray, 'dataframe': pd.DataFrame}
ds.elements_info           # dict — {'array': np.ndarray, 'dataframe': pd.DataFrame}
```

## Exploring the Dataset

```python
ds.print_summary()              # Full overview
ds.print_model_stages()         # List model stages
ds.print_nodal_results()        # List available nodal result types
ds.print_element_results()      # List available element result types
ds.print_element_types()        # Element types grouped by result
ds.print_unique_element_types() # Flat list of unique element types
ds.print_selection_set_info()   # Selection set names and IDs
```

## Extracting Results

The dataset has two component objects — `ds.nodes` and `ds.elements` — that read HDF5 data and return self-contained result containers. Once extracted, these containers carry everything they need (data, time, metadata, geometry) and can be used completely independently of the dataset.

### Nodal Results

```python
# All results, all stages, specific nodes
nr = ds.nodes.get_nodal_results(
    node_ids=[1, 2, 3, 4, 5],
)
# results_name=None -> fetches ALL available results
# model_stage=None  -> fetches ALL model stages

# Specific results + selection set
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    model_stage="MODEL_STAGE[2]",
    selection_set_name="ControlPoints",
)

# Multiple results + explicit node IDs
nr = ds.nodes.get_nodal_results(
    results_name=["DISPLACEMENT", "ACCELERATION"],
    node_ids=[10, 20, 30],
)
```

**You must provide at least one of** `node_ids`, `selection_set_id`, or `selection_set_name`. If multiple are given, they are unioned. `results_name` and `model_stage` default to ALL if omitted.

`get_nodal_results()` returns a `NodalResults` object. See [NodalResults documentation](NodalResults.md).

### Element Results

```python
# By element type + selection set
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
)

# By explicit element IDs
er = ds.elements.get_element_results(
    results_name="section_deformation",
    element_type="2-truss",
    element_ids=[100, 101, 102],
)
```

`get_element_results()` returns an `ElementResults` object. See [ElementResults documentation](ElementResults.md).

### Element Spatial Queries

Building and bridge models are often post-processed **story by story** — you want forces in all columns at floor level 3.0 m, or the membrane stresses in every shear wall panel at a particular elevation. The spatial query helpers implement this horizontal-slice pattern so you don't have to filter the element DataFrame by hand.

#### What is a "Z-level"?

A Z-level is a specific elevation in the global Z-axis (vertical axis in most OpenSees models). The helper matches elements whose **centroid Z-coordinate** falls within a configurable tolerance of the requested value. A single call with `list_z=[0.0, 3.0, 6.0, 9.0]` partitions the entire element set into per-story buckets in one pass.

#### `get_elements_at_z_levels`

Returns a filtered `DataFrame` — a subset of `ds.elements_info["dataframe"]` — keeping only those elements of the requested type whose centroids are at the given Z-levels.

Use this when you want to **inspect** which elements sit at each floor before fetching results.

```python
df_at_z = ds.elements.get_elements_at_z_levels(
    list_z=[0.0, 3.0, 6.0],
    element_type="203-ASDShellQ4",
    tol=0.1,          # Z-coordinate tolerance (default 0.1 m)
)
# Returns a DataFrame with columns: element_id, element_type, z_level, …
```

#### `get_elements_in_selection_at_z_levels`

Same as above but intersects with a named selection set first. Useful for filtering a *specific structural system* (e.g. "only interior shear walls") at each floor.

```python
df_sel_z = ds.elements.get_elements_in_selection_at_z_levels(
    list_z=[0.0, 3.0],
    selection_set_name="ShearWall",
    element_type="203-ASDShellQ4",
)
```

#### `get_element_results_by_selection_and_z`

The most powerful variant. It applies the selection + Z-level filter and then **fetches `ElementResults` for each distinct element type found**. The return value is a dictionary keyed by the decorated element-type string (as it appears in the HDF5 file):

```python
results_by_type = ds.elements.get_element_results_by_selection_and_z(
    results_name="globalForces",
    list_z=[0.0, 3.0],
    selection_set_name="ShearWall",
    model_stage="MODEL_STAGE[1]",
)
# {
#   '203-ASDShellQ4[...]': ElementResults,   # shells at z=0 and z=3
#   '5-ElasticBeam3d[...]': ElementResults,  # columns at those floors
# }
```

Iterate the returned dict to process each type independently:

```python
for type_key, er in results_by_type.items():
    s = er.integrate_canonical("membrane_xx")
    print(f"{type_key}: total Fxx = {s.unstack().sum(axis=1).values}")
```

This pattern is the recommended way to extract **inter-story shear** or **story-level membrane forces** for post-processing workflows.

### Introspecting Available Element Results

```python
avail = ds.elements.get_available_element_results(
    element_type="203-ASDShellQ4",
)
# returns: {partition_id: {result_name: [decorated_type_names]}}
```

## Typical Workflow

```python
from STKO_to_python import MPCODataSet

# 1. Load dataset
ds = MPCODataSet(r"C:\results\building_model", "mpco", name="Building")

# 2. Explore
ds.print_summary()
ds.print_selection_set_info()

# 3. Extract nodal results into a self-contained object
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selection_set_name="ControlPoints",
)

# 4. Work with the results object (independent of ds)
drift_env = nr.interstory_drift_envelope(component=1)
print(drift_env)

# 5. Save for later
nr.save_pickle("displacement_results.pkl")

# 6. Reload without needing the HDF5 files
from STKO_to_python import NodalResults
nr = NodalResults.load_pickle("displacement_results.pkl")
```

## Selection Sets

Selection sets come from the `.mpco.cdata` companion files. Each set has:

- An integer ID (the dict key in `ds.selection_set`)
- A name (`SET_NAME`)
- Lists of `NODES` and `ELEMENTS`

You can reference them by name or ID when extracting results:

```python
# See what's available
ds.print_selection_set_info()

# Use by name
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selection_set_name="FloorNodes",
)

# Use by ID
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    selection_set_id=3,
)
```

Name resolution is case-insensitive. If a name is ambiguous (matches multiple IDs), an error is raised suggesting you use the ID instead.

## Notes

- The dataset uses a composition pattern: `ds.nodes`, `ds.elements`, `ds.model_info`, `ds.cdata`, `ds.plot`, and `ds.info` are internal components. The public API is through `ds.nodes.get_nodal_results()` and `ds.elements.get_element_results()`.
- Underscore-prefixed methods (e.g. `_get_all_nodes_ids`) are internal and called automatically during construction.
- HDF5 reads are optimized with sorted fancy indexing and single-pass multi-result reads.
- The dataset supports partitioned MPCO files (multiple `.mpco` files for parallel recorders).
