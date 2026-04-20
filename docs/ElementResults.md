# ElementResults — Element Results Container

`ElementResults` is a self-contained container for element-level results extracted from MPCO HDF5 files. Once created (via `ds.elements.get_element_results()`), it carries the data, time array, and metadata needed for post-processing — independent of the original dataset. It can be pickled and reloaded later without needing the HDF5 files.

## Creating an ElementResults Object

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"C:\path\to\results", "mpco")

# Extract element results
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
)

# From this point on, 'er' is fully independent of 'ds'
```

### Selection Options

You must provide **at least one** of `element_ids`, `selection_set_id`, or `selection_set_name`. If multiple are given, they are unioned. The `element_type` parameter is always required. Element selection supports three modes:

```python
# By selection set name (case-insensitive)
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
)

# By selection set ID
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_id=7,
)

# By explicit element IDs
er = ds.elements.get_element_results(
    results_name="section_deformation",
    element_type="2-truss",
    element_ids=[100, 101, 102, 103],
)
```

### Specifying Model Stage

By default the first model stage is used. To specify:

```python
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
    model_stage="MODEL_STAGE[2]",
)
```

## Internal Structure

The core data is in `er.df` — a pandas DataFrame with:

- **Index**: MultiIndex `(element_id, step)`
- **Columns**: `val_1`, `val_2`, ..., `val_N` (one per result component)

Additional attributes:

```python
er.time            # np.ndarray — time values per step
er.name            # str — model name
er.element_ids     # tuple[int, ...] — element IDs in this result set
er.element_type    # str — base element type (e.g. "203-ASDShellQ4")
er.results_name    # str — HDF5 result group name (e.g. "globalForces")
er.model_stage     # str — model stage this data comes from
```

## Introspection

```python
er.list_components()
# ('val_1', 'val_2', 'val_3', 'val_4', 'val_5', 'val_6')

er.n_components    # 6
er.n_elements      # 150
er.n_steps         # 2000
er.empty           # False
```

## Fetching Data

### Basic fetch

```python
# All components, all elements
df = er.fetch()

# Single component
s = er.fetch("val_1")

# Filter by element IDs
s = er.fetch("val_1", element_ids=[100, 101])

# All components, filtered elements
df = er.fetch(element_ids=[100, 101])
```

### Attribute-style access (ResultView)

Each column is available as an attribute via `_ElementResultView`:

```python
# Equivalent to er.fetch("val_1")
s = er.val_1.series

# Filter by element IDs
s = er.val_1[100]          # single element
s = er.val_1[[100, 101]]   # multiple elements
s = er.val_1[:]            # all elements
```

## Envelope

Compute min/max over all time steps for each element:

```python
# All components
env = er.envelope()
# Returns DataFrame indexed by element_id:
#   val_1_min, val_1_max, val_2_min, val_2_max, ...

# Single component
env = er.envelope(component="val_1")
# Returns DataFrame indexed by element_id:
#   val_1_min, val_1_max
```

## Time Snapshots

### By step index

```python
df_step = er.at_step(100)
# Returns DataFrame indexed by element_id with columns val_1, val_2, ...
```

### By time value

```python
df_time = er.at_time(5.0)
# Finds step closest to t=5.0, returns same as at_step()
```

## Exporting to DataFrame

```python
# Flat DataFrame with time column attached
df_flat = er.to_dataframe(include_time=True)
# Adds a 'time' column mapped from step index to time value
```

## Pickle Serialization

Save and reload without needing the original HDF5 files:

```python
# Save
er.save_pickle("shell_forces.pkl")

# Save compressed
er.save_pickle("shell_forces.pkl.gz")

# Reload
from STKO_to_python import ElementResults
er = ElementResults.load_pickle("shell_forces.pkl")
er = ElementResults.load_pickle("shell_forces.pkl.gz")
```

Compression is auto-detected from the `.gz` extension, or forced with `compress=True/False`.

## Spatial Queries (via ds.elements)

Before extracting results, the Elements class provides spatial filtering that can narrow down which elements you query:

```python
# All elements at certain Z-levels
df = ds.elements.get_elements_at_z_levels(
    list_z=[0.0, 3.0, 6.0],
    element_type="203-ASDShellQ4",
)

# Elements from a selection set at Z-levels
df = ds.elements.get_elements_in_selection_at_z_levels(
    list_z=[0.0, 3.0],
    selection_set_name="ShearWall",
    element_type="203-ASDShellQ4",
)

# Combined: selection + Z filter, then fetch, grouped by decorated type
results_by_type = ds.elements.get_element_results_by_selection_and_z(
    results_name="globalForces",
    list_z=[0.0, 3.0],
    selection_set_name="ShearWall",
)
# Returns: {'203-ASDShellQ4[...details...]': ElementResults, ...}
```

### Discovering Available Results

```python
avail = ds.elements.get_available_element_results(
    element_type="203-ASDShellQ4",
)
# Returns: {partition_id: {result_name: [decorated_type_names]}}
```

## Full Example: Shell Wall Force Envelopes

```python
from STKO_to_python import MPCODataSet, ElementResults

# --- Load ---
ds = MPCODataSet(r"C:\results\building", "mpco", name="RC_Building")

# --- Explore available element results ---
avail = ds.elements.get_available_element_results(
    element_type="203-ASDShellQ4",
)
print(avail)

# --- Extract shell forces ---
er = ds.elements.get_element_results(
    results_name="globalForces",
    element_type="203-ASDShellQ4",
    selection_set_name="ShearWall",
    model_stage="MODEL_STAGE[2]",
)

# --- Save for reuse ---
er.save_pickle("wall_forces.pkl.gz")

# --- Later: reload ---
er = ElementResults.load_pickle("wall_forces.pkl.gz")

# --- Inspect ---
print(er)
# ElementResults: globalForces on 203-ASDShellQ4
#   Elements: 150, Steps: 2000, Components: 6

print(er.list_components())
# ('val_1', 'val_2', 'val_3', 'val_4', 'val_5', 'val_6')

# --- Force envelope ---
env = er.envelope()
print(env.head())
#             val_1_min  val_1_max  val_2_min  val_2_max  ...

# --- Snapshot at t=5s ---
snap = er.at_time(5.0)
print(snap.head())

# --- Time history for specific element ---
s = er.val_1[42]
print(s)

# --- Export with time column ---
df = er.to_dataframe(include_time=True)
df.to_csv("wall_forces_flat.csv")
```

## Full Example: Z-Level Filtering

```python
from STKO_to_python import MPCODataSet

ds = MPCODataSet(r"C:\results\building", "mpco")

# Get shell elements at base and first floor
results_by_type = ds.elements.get_element_results_by_selection_and_z(
    results_name="globalForces",
    list_z=[0.0, 3.5],
    selection_set_name="ShearWall",
)

for etype, er in results_by_type.items():
    print(f"\n{etype}: {er.n_elements} elements")
    env = er.envelope(component="val_1")
    print(f"  Max val_1: {env['val_1_max'].max():.2f}")
    print(f"  Min val_1: {env['val_1_min'].min():.2f}")

    # Save each type separately
    er.save_pickle(f"forces_{etype.replace('/', '_')}.pkl.gz")
```

## Notes

- Column names (`val_1`, `val_2`, ...) are generic. The physical meaning depends on the element type and result name in your OpenSees model (e.g., for `globalForces` on shell elements, these might correspond to Nx, Ny, Nxy, Mx, My, Mxy).
- The `element_type` parameter uses the base type (before the bracket), e.g. `"203-ASDShellQ4"` not `"203-ASDShellQ4[some_detail]"`.
- `at_time()` uses the closest step to the requested time. If the exact time is not found, no error is raised — it picks the nearest available step.
