# Query engines

Assemble `DataFrame`s with a normalised MultiIndex. Each dataset
holds one instance per kind (nodal, element) under
`ds._nodal_query_engine` / `ds._element_query_engine`. The LRU cache
(default size 32) is on by default; cache keys are
`(stage, result_name, component, ids_hash, step_slice)`.

## BaseResultsQueryEngine

Template-method parent. Holds the MultiIndex / ID-axis caches,
handles stage iteration, and implements the chunk-sorted fancy-index
path.

::: STKO_to_python.query.base_query_engine.BaseResultsQueryEngine

## NodalResultsQueryEngine

::: STKO_to_python.query.nodal_query_engine.NodalResultsQueryEngine

## ElementResultsQueryEngine

::: STKO_to_python.query.element_query_engine.ElementResultsQueryEngine
