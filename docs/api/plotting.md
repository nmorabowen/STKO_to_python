# Plotting

Two flavours of XY plots:

- **Per-result** (`nr.plot.*`) — use when you'll produce multiple plots
  off the same `NodalResults`; reuses the cached DataFrame.
- **Dataset-level** (`ds.plot.*`) — one-shot "fetch and plot"
  convenience. Internally fetches a `NodalResults` and delegates.

For element result plots (`er.plot.*`) see
[ElementResults — plotting](element-results.md#elementresultsplotter).

---

## `xy` — shared parameter reference

Both `nr.plot.xy()` and `ds.plot.xy()` share the same parameter
structure. The dataset-level version prepends the fetch arguments.

### Y-axis

| Parameter | Default | Description |
|---|---|---|
| `y_results_name` | required | Result name, e.g. `"DISPLACEMENT"` |
| `y_direction` | `None` | Component index (int) or name. If `None`, all components are aggregated by `y_operation`. |
| `y_operation` | `"Sum"` | Aggregation over nodes/components: `"Sum"`, `"Max"`, `"Min"`, `"Mean"`, `"Percentile"` |
| `y_scale` | `1.0` | Scalar multiplier applied to the y values |

### X-axis

| Parameter | Default | Description |
|---|---|---|
| `x_results_name` | `"TIME"` | `"TIME"`, `"STEP"`, or another result name for an XY scatter |
| `x_direction` | `None` | Component index / name for the x result |
| `x_operation` | `"Sum"` | Aggregation for the x result |
| `x_scale` | `1.0` | Scalar multiplier for x values |

### Aggregation extras

| Parameter | Default | Description |
|---|---|---|
| `operation_kwargs` | `None` | Extra kwargs forwarded to the aggregator. Currently only `{"percentile": 95.0}` is accepted. |

### Cosmetics

| Parameter | Default | Description |
|---|---|---|
| `ax` | `None` | Existing `matplotlib.Axes` to draw on. If `None`, a new figure is created. |
| `figsize` | `(10, 6)` | Figure size in inches (ignored when `ax` is supplied) |
| `linewidth` | `None` | Overrides `PlotSettings.linewidth` |
| `marker` | `None` | Overrides `PlotSettings.marker` |
| `label` | `None` | Legend label. Auto-generated when `None`. |
| `**line_kwargs` | | Forwarded to `ax.plot` |

All methods return `(ax, meta)` where `meta` carries `{"x": ..., "y": ...}`.

---

## Examples

### Time history of maximum displacement

```python
nr = ds.nodes.get_nodal_results(
    results_name="DISPLACEMENT",
    model_stage="MODEL_STAGE[1]",
    node_ids=[1, 2, 3, 4],
)

ax, meta = nr.plot.xy(
    y_results_name="DISPLACEMENT",
    y_direction=1,
    y_operation="Max",
    x_results_name="TIME",
)
```

### One-shot dataset-level plot

```python
ax, meta = ds.plot.xy(
    model_stage="MODEL_STAGE[1]",
    results_name="DISPLACEMENT",
    node_ids=[1, 2, 3, 4],
    y_direction=1,
    y_operation="Max",
    x_results_name="TIME",
)
```

### XY scatter — roof displacement vs. base shear

```python
ax, meta = ds.plot.xy(
    model_stage="MODEL_STAGE[1]",
    results_name="DISPLACEMENT",
    node_ids=[4],            # roof node
    y_direction=1,
    y_operation="Sum",
    x_results_name="REACTION_FORCE",
    x_direction=1,
    x_operation="Sum",
)
```

### Per-node time-history subplots

`plot_TH` is a convenience wrapper that shows one curve per node on
either a shared axes or one subplot per node:

```python
fig, meta = nr.plot.plot_TH(
    result_name="DISPLACEMENT",
    component=1,
    node_ids=[1, 2, 3, 4],
    split_subplots=True,
    figsize=(8, 10),
    sharey=True,
)
```

---

## Style precedence

Style is resolved in three layers (highest wins):

1. **Model-level `PlotSettings`** — set via `ds.plot_settings` or attached
   to a `NodalResults` directly.
2. **Per-call keyword arguments** — `linewidth=`, `marker=`, `label=`.
3. **`**line_kwargs`** — any extra kwarg forwarded verbatim to `ax.plot`.

```python
from STKO_to_python.plotting.plot_settings import PlotSettings

nr.plot_settings = PlotSettings(linewidth=1.5, marker="o", color="steelblue")
ax, _ = nr.plot.xy(y_results_name="DISPLACEMENT", y_direction=1,
                   y_operation="Max", x_results_name="TIME")
```

---

## API reference

## Plot (dataset-level facade)

::: STKO_to_python.plotting.plot.Plot

## NodalResultsPlotter

Bound to a `NodalResults` instance as `nr.plot`. Handles aggregation
dispatch (via `Aggregator`) and style precedence.

::: STKO_to_python.results.nodal_results_plotting.NodalResultsPlotter

## PlotSettings

Dataclass holding per-result style overrides. Attach to `nr.plot_settings`
or pass to the dataset for model-wide defaults.

::: STKO_to_python.plotting.plot_settings.PlotSettings
