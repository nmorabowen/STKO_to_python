# MPCOResults

Multi-case orchestration wrapper for a collection of `NodalResults`
objects keyed by `(model, station, rupture)`. Designed for ensembles of
ground-motion analyses — e.g., 11 records per model, multiple structural
configurations.

---

## Construction

### From a dict

```python
from STKO_to_python.MPCOList.MPCOResults import MPCOResults

results = MPCOResults(
    {
        ("Model1A", "CHY101", "GM01"): nr_1,
        ("Model1A", "TCU052", "GM02"): nr_2,
        ("Model2B", "CHY101", "GM01"): nr_3,
    },
    name="RC_Building_Suite",
)
```

Each key is a 3-tuple `(model, station, rupture)` and each value is a
`NodalResults` instance (or a lazy proxy — see `load_dir`).

### From a directory of pickles — `load_dir`

```python
results = MPCOResults.load_dir(
    out_dir=Path("./outputs"),
    lazy=True,   # defer pickle loading until first access (default)
    name="Suite",
)
```

Scans `out_dir` for `.pkl` and `.pkl.gz` files whose names match the
pattern `<model>__<station>__<rupture>.pkl[.gz]`. Lazy mode wraps each
file in a proxy that loads on first attribute access — useful for large
suites.

---

## Collection protocol

`MPCOResults` behaves like a read-only mapping:

```python
len(results)          # number of cases
list(results)         # list of (model, station, rupture) keys
results.keys()        # key iterable
results.values()      # NodalResults iterable
results.items()       # (key, NodalResults) iterable
results[("Model1A", "CHY101", "GM01")]  # direct lookup
```

---

## Filtering — `select`

```python
pairs = results.select(
    model="Model1A",           # exact match (case-insensitive)
    station="*CHY*",           # glob: contains "CHY"
    rupture=None,              # None / "all" → match everything
)
# Returns list of ((model, station, rupture), NodalResults) tuples
```

Filter patterns follow shell-glob rules:
- `"*val*"` — contains
- `"*val"` — ends with
- `"val*"` — starts with
- `"exact"` — exact match (case-insensitive)
- `None` / `"all"` — match everything

Multiple values can be passed as a list (OR logic):

```python
pairs = results.select(station=["CHY101", "TCU052"])
```

---

## `.df` accessor (MPCO_df)

`results.df` gives access to the DataFrame-extractor family. The legacy
alias `results.create_df` points to the same object.

### `drift_df` — wide drift table

```python
df = results.df.drift_df(
    top=(10.0, 0.0, 6.0),       # (x, y, z) of the top point
    bottom=(10.0, 0.0, 3.0),    # (x, y, z) of the bottom point
    components=(1, 2),           # list of displacement components
    result_name="DISPLACEMENT",
    relative_drift=True,         # True → drift ratio, False → delta_u
    reduce_time="abs_max",       # "abs_max" | "max" | "min" | "rms"
    stage=None,
    combine="srss",              # "srss" | "maxabs" | "none"
    op="raw",                    # "raw" | "log"
    model=None,
    station=None,
    rupture=None,
)
```

Returns a **wide** `pd.DataFrame` with columns
`Tier | Case | sta | rup | EDP` — one row per `(model, station, rupture)`
case. `combine` controls how multi-component drifts are merged:
- `"srss"` — SRSS of the components' peak values
- `"maxabs"` — max of the absolute component peaks
- `"none"` — single component (requires `len(components) == 1`)

### `drift_df_long` — long/tidy drift table

```python
df_long = results.df.drift_df_long(
    top=..., bottom=...,
    components=(1, 2),
    reduce_time="abs_max",
    combine="srss",
    # ... same kwargs as drift_df
)
```

Converts the wide table to a tidy long format with columns
`Tier | Case | sta | rup | runkey | component | result_name |
reduce_time | relative_drift | op | edp` — ready for seaborn / plotnine
strip plots and statistical summaries.

### `pga_df` — wide PGA table

```python
df = results.df.pga_df(
    component=1,
    selection_set_name="ControlPoints",
    result_name="ACCELERATION",
    dz_tol=1e-3,
    to_g=True,
    g_value=9810,
    reduce_nodes="max_abs",
    model=None, station=None, rupture=None,
)
```

Returns a wide table with one column per story elevation and one row per
case. Also available as `pga_df_long`, `pga_df_mod`, `pga_df_long_mod`.

### `torsion_df` — wide roof torsion table

```python
df = results.df.torsion_df(
    node_a_coord=(0.0, 0.0),
    node_b_coord=(20.0, 0.0),
    ux_component=1,
    uy_component=2,
    reduce="abs_max",
    model=None, station=None, rupture=None,
)
```

Also available as `torsion_df_long`.

### `base_rocking_df` — wide base rocking table

```python
df = results.df.base_rocking_df(
    node_coords_xy=[(0, 0), (10, 0), (5, 8)],
    z_coord=0.0,
    uz_component=3,
    reduce="abs_max",
    model=None, station=None, rupture=None,
)
```

Returns a wide table with columns `theta_x_abs_max, theta_y_abs_max,
theta_mag_abs_max` per case. Also available as `base_rocking_df_long`.

### `wide_to_long` — general wide→long conversion

```python
df_long = results.df.wide_to_long(
    df_wide,
    id_cols=("Tier", "Case", "sta", "rup"),
    value_col="EDP",
    runkey_col="runkey",
    runkey_from=("sta", "rup"),
    runkey_sep=":",
    result_name=None,
    component=None,
    reduce_time=None,
    relative_drift=None,
    op="raw",
)
```

General-purpose WIDE → LONG normalizer. Adds metadata columns
(`result_name`, `component`, `reduce_time`, `relative_drift`, `op`) and
a `runkey` composite column from the specified source columns.

---

## Plotting

### `plot_drift`

```python
results.plot_drift(
    top=(10.0, 0.0, 6.0),     # (x, y, z) of the top point
    bottom=(10.0, 0.0, 3.0),  # (x, y, z) of the bottom point
    component=1,
    relative_drift=True,       # True → drift ratio, False → delta_u
    model=None, station=None, rupture=None,
    overlay=True,              # True → one figure; False → one per case
    figsize=(10, 6),
    running_envelope=None,     # None | "abs" | "signed"
    envelope_alpha=0.35,
    envelope_only=False,
    xlim=None, ylim=None,
    legend_fontsize=7,
)
```

Plots drift time histories for all (filtered) cases. When `overlay=True`
all records appear on a single axes. `running_envelope="abs"` overlays a
running-maximum envelope band.

### `plot_drift_envelope`

```python
results.plot_drift_envelope(
    component=1,
    selection_set_name="ControlPoints",
    dz_tol=1e-3,
    model=None, station=None, rupture=None,
    overlay=True,
    figsize=(6, 10),
    xlim=None, ylim=None,
)
```

Plots the interstory drift envelope profile (drift vs. story height) as
a step plot for each case.

---

## Static helpers

### `parse_tier_letter`

```python
tier, letter = MPCOResults.parse_tier_letter("2B_tall")
# tier=2, letter="B"
```

Extracts the tier number (int) and letter (str) from a model label
string using a regex `(\d+)([A-Z])`.

### `compute_table`

```python
df = results.compute_table(
    fn,           # Callable[[NodalResults], float | dict]
    model=None, station=None, rupture=None,
)
```

Applies an arbitrary function to every selected `NodalResults` and
assembles the results into a wide table. `fn` may return a `float` (one
column `value`) or a `dict` (one column per key).

```python
df = results.compute_table(
    lambda nr: nr.drift(top=42, bottom=10, component=1, reduce="abs_max"),
)
```

---

## Full example

```python
from pathlib import Path
from STKO_to_python.MPCOList.MPCOResults import MPCOResults

# Load all pickled NodalResults from a directory
results = MPCOResults.load_dir(out_dir=Path("./ground_motion_outputs"), lazy=True)

print(f"Loaded {len(results)} cases")

# Filter to a specific model
pairs = results.select(model="Model1A")
for (model, sta, rup), nr in pairs:
    print(f"  {model} | {sta} | {rup}  -> {nr.n_steps} steps")

# Drift EDP table (wide)
drift_wide = results.df.drift_df(
    top=(10.0, 0.0, 6.0),
    bottom=(10.0, 0.0, 3.0),
    components=(1, 2),
    reduce_time="abs_max",
    combine="srss",
)
print(drift_wide.head())

# Convert to long for plotting
drift_long = results.df.wide_to_long(drift_wide, result_name="DISPLACEMENT", op="raw")

# PGA profile
pga = results.df.pga_df(
    component=1,
    selection_set_name="ControlPoints",
    result_name="ACCELERATION",
    to_g=True,
)

# Overlay drift histories for one station
results.plot_drift(
    top=(10.0, 0.0, 6.0),
    bottom=(10.0, 0.0, 3.0),
    component=1,
    station="CHY101",
    running_envelope="abs",
)

# ASCE torsional irregularity across all cases
torsion = results.compute_table(
    lambda nr: nr.asce_torsional_irregularity(
        component=1,
        side_a_top=(0.0, 0.0, 6.0),
        side_a_bottom=(0.0, 0.0, 3.0),
        side_b_top=(20.0, 0.0, 6.0),
        side_b_bottom=(20.0, 0.0, 3.0),
    )["ratio"]
)
print(torsion.describe())
```

---

## API reference

::: STKO_to_python.MPCOList.MPCOResults.MPCOResults

## MPCO_df

The DataFrame-extractor accessor. Reach it via `mpco_results.df`
(preferred) or `mpco_results.create_df` (legacy).

::: STKO_to_python.MPCOList.MPCOdf.MPCO_df
