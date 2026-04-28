"""End-to-end usage tour of STKO_to_python.

Runs against the checked-in elasticFrame example fixture
(``stko_results_examples/elasticFrame/results``) so every line here
is a real, executable call. Pass the path to a different ``.mpco``
output directory via ``--dataset-dir`` to run the tour against your
own data.

Covers:
    1. Dataset construction + introspection.
    2. Fetching a ``NodalResults`` view.
    3. Reading it three ways: fetch(), dynamic views, introspection.
    4. Engineering aggregations (forwarders to AggregationEngine).
    5. Plotting (per-result and dataset-level facade).
    6. Pickle round-trip.
    7. Fetching an ``ElementResults`` view.
    8. The ElementResults broker API (envelope, at_step, to_dataframe).
    9. Selection sets as an alternative to explicit IDs.
   10. Element discovery (get_available_element_results).
   11. Canonical engineering names on ElementResults.
   12. Integration-point extraction (at_ip, gp_xi, physical_x).
   13. Dataset introspection print helpers.
   14. Element result discovery workflow (full pattern).

Run with::

    python examples/usage_tour.py
    python examples/usage_tour.py --dataset-dir path/to/other/output --recorder results
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the tour runs headlessly in CI and
# smoke tests. Swap to "TkAgg" / "Qt5Agg" interactively if you want to
# actually see the figures.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from STKO_to_python import MPCODataSet
from STKO_to_python.elements.element_results import ElementResults
from STKO_to_python.results.nodal_results_dataclass import NodalResults


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "results"
DEFAULT_RECORDER = "results"


def section(title: str) -> None:
    """Print a visual separator between tour sections."""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ---------------------------------------------------------------------- #
# Sections
# ---------------------------------------------------------------------- #
def section_1_introspect(ds: MPCODataSet) -> None:
    section("1. Dataset introspection")
    print(f"stages:              {ds.model_stages}")
    print(f"# node result names: {len(ds.node_results_names)}")
    print(f"first few:           {ds.node_results_names[:4]}")

    avail = ds.elements.get_available_element_results()
    for part_id, mapping in avail.items():
        for rname, types in mapping.items():
            print(f"element result:      partition {part_id} / {rname!r} -> {types}")


def section_2_fetch_nodal(ds: MPCODataSet) -> NodalResults:
    section("2. Fetch a NodalResults")
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    print(repr(nr))
    print(f"df.shape:       {nr.df.shape}")
    print(f"df.index.names: {list(nr.df.index.names)}")
    print(f"time (first 3): {nr.time[:3]}")
    return nr


def section_3_three_ways(nr: NodalResults) -> None:
    section("3. Three ways to pull data")

    # (a) fetch() — full API
    sub = nr.fetch(result_name="DISPLACEMENT", component=1, node_ids=[1, 4])
    print(f"(a) fetch(comp=1, node_ids=[1,4]) -> Series length {len(sub)}")

    # (b) dynamic attribute views
    view_all = nr.DISPLACEMENT[1]
    view_narrow = nr.DISPLACEMENT[1, [1, 4]]
    print(f"(b) nr.DISPLACEMENT[1]         -> Series length {len(view_all)}")
    print(f"    nr.DISPLACEMENT[1, [1,4]]  -> Series length {len(view_narrow)}")

    # (c) introspection
    print(f"(c) list_results():                 {nr.list_results()}")
    print(f"    list_components('DISPLACEMENT'): {nr.list_components('DISPLACEMENT')}")


def section_4_aggregations(nr: NodalResults) -> None:
    section("4. Engineering aggregations")

    drift_ts = nr.drift(top=4, bottom=1, component=1)
    print(f"drift series:           length {drift_ts.size}, max|.|={abs(drift_ts).max():.3e}")

    env = nr.interstory_drift_envelope(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-3,
    )
    print(f"interstory envelope:    {len(env)} story pair(s)")
    print(f"  columns:              {list(env.columns)}")

    resid = nr.residual_drift(top=4, bottom=1, component=1, tail=3, agg="mean")
    print(f"residual_drift (mean of last 3 steps): {resid:.3e}")

    sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)
    print(f"orbit for node 1:       x length {len(sx)}, y length {len(sy)}")


def section_5_plotting(ds: MPCODataSet, nr: NodalResults) -> None:
    section("5. Plotting")

    # (a) per-result
    ax, meta = nr.plot.xy(
        y_results_name="DISPLACEMENT", y_direction=1, y_operation="Max",
        x_results_name="TIME",
    )
    print(f"(a) nr.plot.xy          -> ax={ax}, meta keys={list(meta.keys())}")
    plt.close("all")

    # (b) dataset-level
    ax, meta = ds.plot.xy(
        model_stage="MODEL_STAGE[1]",
        results_name="DISPLACEMENT",
        node_ids=[1, 2, 3, 4],
        y_direction=1, y_operation="Max",
        x_results_name="TIME",
    )
    print(f"(b) ds.plot.xy          -> ax={ax}, meta keys={list(meta.keys())}")
    plt.close("all")


def section_6_pickle_nodal(nr: NodalResults) -> None:
    section("6. NodalResults pickle round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nr.pkl"
        nr.save_pickle(path)
        loaded = NodalResults.load_pickle(path)
        print(f"saved to:   {path.name}  ({path.stat().st_size} bytes)")
        print(f"loaded:     {loaded!r}")
        print(f"df match:   {loaded.df.shape == nr.df.shape}")


def section_7_fetch_element(ds: MPCODataSet) -> ElementResults:
    section("7. Fetch an ElementResults")
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )
    print(repr(er))
    print(f"df.shape:          {er.df.shape}")
    print(f"df.index.names:    {list(er.df.index.names)}")
    print(f"list_components(): {er.list_components()}")
    print(f"n_elements={er.n_elements}, n_steps={er.n_steps}, n_components={er.n_components}")
    return er


def section_8_element_broker(er: ElementResults) -> None:
    section("8. ElementResults broker API")

    sub = er.fetch(component="Pz_1", element_ids=[1, 2])
    print(f"fetch('Pz_1', [1,2]) -> Series length {len(sub)}")

    view = er.Pz_1[[1, 2]]
    print(f"er.Pz_1[[1,2]]        -> type {type(view).__name__}")

    env = er.envelope(component="Pz_1")
    print(f"envelope('Pz_1'):     {list(env.columns)}")
    print(env)

    snap_step = er.at_step(5)
    print(f"\nat_step(5) shape:      {snap_step.shape}")

    snap_time = er.at_time(0.5)
    print(f"at_time(0.5) shape:    {snap_time.shape}")

    flat = er.to_dataframe(include_time=True)
    print(f"to_dataframe() columns: {list(flat.columns)[:5]} ... ({len(flat.columns)} total)")


def section_9_pickle_element(er: ElementResults) -> None:
    section("9. ElementResults pickle round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "er.pkl"
        er.save_pickle(path)
        loaded = ElementResults.load_pickle(path)
        print(f"saved to:   {path.name}  ({path.stat().st_size} bytes)")
        print(f"loaded:     {loaded!r}")
        print(f"df match:   {loaded.df.shape == er.df.shape}")


def section_10_selection_sets(ds: MPCODataSet) -> None:
    section("10. Selection sets (alternative to explicit IDs)")
    print("If the .cdata defines named selection sets, pass the name or ID")
    print("instead of a list. Explicit IDs still work, and the resolver")
    print("takes the UNION when both are supplied.")
    print()
    print("    ds.nodes.get_nodal_results(")
    print("        results_name='DISPLACEMENT',")
    print("        model_stage='MODEL_STAGE[1]',")
    print("        selection_set_name='roof_diaphragm',")
    print("    )")
    print()
    print("    ds.elements.get_element_results(")
    print("        results_name='force',")
    print("        element_type='5-ElasticBeam3d',")
    print("        model_stage='MODEL_STAGE[1]',")
    print("        selection_set_id=3,")
    print("    )")


def section_11_element_discovery(ds: MPCODataSet) -> None:
    section("11. Element discovery — get_available_element_results")

    avail = ds.elements.get_available_element_results()
    for part_id, mapping in avail.items():
        for rname, decorated_types in mapping.items():
            print(
                f"partition {part_id!r}: result={rname!r}  "
                f"types={decorated_types}"
            )

    # Filter to a specific base type
    avail_beam = ds.elements.get_available_element_results(
        element_type="5-ElasticBeam3d"
    )
    print(f"\nfiltered to 5-ElasticBeam3d: {avail_beam}")


def section_12_canonical_names(ds: MPCODataSet) -> None:
    section("12. Canonical engineering names")

    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )

    print(f"list_canonicals():  {er.list_canonicals()}")

    for canon in er.list_canonicals():
        cols = er.canonical_columns(canon)
        print(f"  {canon!r:30s} -> {cols}")

    # Full DataFrame for one canonical name
    first_canon = er.list_canonicals()[0]
    df = er.canonical(first_canon)
    print(f"\ncanonical({first_canon!r}) shape: {df.shape}")

    # Module-level helpers
    from STKO_to_python.elements.canonical import (
        available_canonicals,
        shortname_of,
    )
    print(f"\navailable_canonicals() count: {len(available_canonicals())}")
    print(f"shortname_of('Pz_1') = {shortname_of('Pz_1')!r}")
    print(f"shortname_of('P_ip3') = {shortname_of('P_ip3')!r}")


def section_13_integration_points(ds: MPCODataSet) -> None:
    section("13. Integration-point access (gp_xi / at_ip)")

    # Closed-form bucket — gp_xi is None
    er_cf = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )
    print(f"Closed-form:  gp_xi={er_cf.gp_xi}  n_ip={er_cf.n_ip}")
    print("  (at_ip() raises ValueError for closed-form buckets)")

    # Line-station bucket — gp_xi is populated if section.force exists
    avail = ds.elements.get_available_element_results(
        element_type="5-ElasticBeam3d"
    )
    all_results = [
        r for mapping in avail.values() for r in mapping
    ]
    line_station_name = next(
        (r for r in all_results if "section" in r), None
    )

    if line_station_name:
        er_ls = ds.elements.get_element_results(
            results_name=line_station_name,
            element_type="5-ElasticBeam3d",
            model_stage="MODEL_STAGE[1]",
            element_ids=[1, 2, 3],
        )
        print(f"\nLine-station ({line_station_name!r}):")
        print(f"  gp_xi={er_ls.gp_xi}  n_ip={er_ls.n_ip}")
        if er_ls.n_ip:
            sub = er_ls.at_ip(0)
            print(f"  at_ip(0) columns: {list(sub.columns)}")
            phys = er_ls.physical_x(length=3.0)
            print(f"  physical_x(L=3.0): {phys}")
    else:
        print(
            "\n(No line-station result in this fixture; "
            "see test_gp_xi_and_at_ip.py for a full demo.)"
        )


def section_14_print_helpers(ds: MPCODataSet) -> None:
    section("14. Dataset print helpers")
    print("All print_* methods log at INFO level — enable with verbose=True")
    print("or logging.basicConfig(level=logging.INFO).\n")
    print("ds.print_summary()              -> stages + nodal + element overview")
    print("ds.print_model_stages()         -> list of model stages")
    print("ds.print_nodal_results()        -> available nodal result names")
    print("ds.print_element_results()      -> available element result names")
    print("ds.print_element_types()        -> result → decorated type mapping")
    print("ds.print_unique_element_types() -> flat list of all decorated types")
    print("ds.print_selection_set_info()   -> named selection sets from .cdata")


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def run(dataset_dir: Path, recorder: str) -> None:
    """Execute the full tour against the given dataset directory."""
    ds = MPCODataSet(str(dataset_dir), recorder, verbose=False)

    section_1_introspect(ds)
    nr = section_2_fetch_nodal(ds)
    section_3_three_ways(nr)
    section_4_aggregations(nr)
    section_5_plotting(ds, nr)
    section_6_pickle_nodal(nr)
    er = section_7_fetch_element(ds)
    section_8_element_broker(er)
    section_9_pickle_element(er)
    section_10_selection_sets(ds)
    section_11_element_discovery(ds)
    section_12_canonical_names(ds)
    section_13_integration_points(ds)
    section_14_print_helpers(ds)

    print()
    print("Tour complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to the .mpco output directory (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--recorder",
        type=str,
        default=DEFAULT_RECORDER,
        help=f"Recorder base name (default: {DEFAULT_RECORDER!r}).",
    )
    args = parser.parse_args()

    if not args.dataset_dir.exists():
        parser.error(
            f"Dataset directory not found: {args.dataset_dir}\n"
            f"Pass --dataset-dir pointing at your own .mpco output folder."
        )

    run(args.dataset_dir, args.recorder)


if __name__ == "__main__":
    main()
