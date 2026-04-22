# AggregationEngine

Stateless engineering-aggregation engine. Holds every drift /
envelope / rocking / torsion / orbit method that `NodalResults`
exposes. `NodalResults._aggregation_engine` is a class-level
singleton — one instance shared across every `NodalResults` in the
process, pickle-safe (not persisted in state).

Every method takes the `NodalResults` as its first positional
argument. Callers usually invoke the forwarders on `NodalResults`
instead (e.g. `nr.drift(...)` rather than
`engine.drift(nr, ...)`).

::: STKO_to_python.dataprocess.aggregation.AggregationEngine
