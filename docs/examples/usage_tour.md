# Usage tour

End-to-end tour of the complete STKO_to_python API, run against the
checked-in `elasticFrame` fixture. Every call is real and executable.

**Script:** [`examples/usage_tour.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/examples/usage_tour.py)

```bash
python examples/usage_tour.py
python examples/usage_tour.py --dataset-dir path/to/other/output --recorder results
```

---

## Sections covered

| # | Section | Key calls |
|---|---|---|
| 1 | Dataset introspection | `ds.model_stages`, `ds.node_results_names`, `ds.elements.get_available_element_results()` |
| 2 | Fetch `NodalResults` | `ds.nodes.get_nodal_results()` |
| 3 | Three ways to pull data | `nr.fetch()`, `nr.DISPLACEMENT[1]`, `nr.list_results()` |
| 4 | Engineering aggregations | `nr.drift()`, `nr.interstory_drift_envelope()`, `nr.residual_drift()`, `nr.orbit()` |
| 5 | Plotting | `nr.plot.xy()`, `ds.plot.xy()` |
| 6 | `NodalResults` pickle | `nr.save_pickle()`, `NodalResults.load_pickle()` |
| 7 | Fetch `ElementResults` | `ds.elements.get_element_results()` |
| 8 | ElementResults broker API | `er.fetch()`, `er.envelope()`, `er.at_step()`, `er.at_time()`, `er.to_dataframe()` |
| 9 | `ElementResults` pickle | `er.save_pickle()`, `ElementResults.load_pickle()` |
| 10 | Selection sets | `selection_set_name=`, `selection_set_id=` |
| 11 | Element discovery | `ds.elements.get_available_element_results(element_type=...)` |
| 12 | Canonical names | `er.list_canonicals()`, `er.canonical_columns()`, `er.canonical()` |
| 13 | Integration points | `er.gp_xi`, `er.n_ip`, `er.at_ip()`, `er.physical_x()` |
| 14 | Dataset print helpers | `ds.print_summary()`, `ds.print_nodal_results()`, … |

---

## Fixture used

`stko_results_examples/elasticFrame/results` — single-partition
`5-ElasticBeam3d` model, 4 nodes, 3 elements, 2 model stages.
