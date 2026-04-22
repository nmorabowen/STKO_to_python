# Plotting

Two flavours:

- **Per-result** (`nr.plot.*`) — use when you'll produce multiple
  plots off the same `NodalResults`; reuses the cached DataFrame.
- **Dataset-level** (`ds.plot.*`) — one-shot "fetch and plot"
  convenience. Internally fetches a `NodalResults` and delegates.

## Plot (dataset-level facade)

::: STKO_to_python.plotting.plot.Plot

## NodalResultsPlotter

Bound to a `NodalResults` instance. Handles the aggregation dispatch
(via [Aggregator](https://github.com/nmorabowen/STKO_to_python/blob/main/src/STKO_to_python/dataprocess/aggregator.py))
and style precedence (model-level `PlotSettings` → explicit kwargs
→ `**line_kwargs`).

::: STKO_to_python.results.nodal_results_plotting.NodalResultsPlotter

## PlotSettings

::: STKO_to_python.plotting.plot_settings.PlotSettings
