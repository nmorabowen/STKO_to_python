# NodalResults

Thin view over result data. Holds a reference to a
[NodalResultsQueryEngine](query-engines.md) and pre-resolved IDs.
Exposes `fetch`, `list_results`, `list_components`, `save_pickle`,
`load_pickle`, and a `.plot` accessor; every engineering aggregation
(`drift`, `interstory_drift_envelope`, `orbit`, `base_rocking`, …) is
a forwarder to [AggregationEngine](aggregation-engine.md).

Pickle-stable since Phase 4.3.3: the class's `(__module__, __qualname__)`
pair does not change, and `__setstate__` tolerates unknown keys (dropped
with a DEBUG log) and missing optional fields.

::: STKO_to_python.results.nodal_results_dataclass.NodalResults

## The dynamic-view proxy

`NodalResults` builds an internal `_ResultView` per result name so
you can write `nr.ACCELERATION[1, [14, 25]]` instead of
`nr.fetch(result_name="ACCELERATION", component=1, node_ids=[14, 25])`.

::: STKO_to_python.results.nodal_results_dataclass._ResultView
    options:
      filters: []
      show_root_heading: true
