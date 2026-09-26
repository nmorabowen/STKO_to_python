---
name: stko-mpco-reader
description: >
  Checklist for CHANGING how the STKO_to_python library reads OpenSees MPCO output:
  adding an element class or a result bucket, touching the META parser
  (io/meta_parser.py), the Gauss-point catalog or shape functions (format/),
  CDataReader / .cdata sections, ModelInfo / time series, the node/element managers,
  the query engines, canonical names, or the MPCO format policy. Use before editing
  anything under src/STKO_to_python/{io,format,model,elements,nodes,query,results,selection}.
  Not for USING the library to post-process results -- that is the end-user
  stko-to-python skill.
---

# Changing the MPCO / .cdata reader — checklist

Read this before changing how the library reads a `.mpco` or `.cdata` file. Each
item names the doc and heading to grep for; read that entry when the item applies.
Items marked **[lint Q1/Q2]** are enforced by `python ci/check_quirk_patterns.py`.

Out of scope: how to *use* the library (the end-user `stko-to-python` skill,
`stko-to-python-SKILL.md`). This guide is for changing it.

## Before writing code

- [ ] Open a real fixture and dump the HDF5 tree with `h5py` before you trust any
      description of the format. `docs/mpco_format_conventions.md`: "Inspect a real
      fixture before writing reader code".
- [ ] Pick the fixture: `docs/architecture.md` "Test fixtures" (committed vs gitignored).
      Point at `stko_results_examples/`, never regenerate it. `AGENTS.md`:
      "`stko_results_examples/` is ground-truth input data".
- [ ] A failure on real data is a delta (numpy, pandas, writer version, MP vs single
      partition) before it is a bug. Keep the fix to the one line. `AGENTS.md`:
      "Trust that the existing library worked".

## Format facts that have already cost time

- [ ] META is the source of truth, and the catalog is only a hint. Validate
      `NUM_COLUMNS` against the layout and raise on a mismatch; never silently
      rename columns. `docs/mpco_format_conventions.md`: "META is the source of truth",
      "`NUM_COLUMNS` should equal the META block sum".
- [ ] Section components come in `getType()` order, not in a canonical order. Read
      the codes from META/COMPONENTS. Same doc: "Section response order matches
      `getType()` order".
- [ ] `GP_X` is an attribute on the connectivity dataset, in natural [-1, +1],
      and it uses the 2-field bracket; results buckets use the 3-field one. Same
      doc: "`GP_X` lives on connectivity", "returns physical `pts[i] * L`".
- [ ] A single element class can be split across several buckets. Group by
      bucket, not by class. Same doc: "The `customRuleIdx` axis groups elements".
- [ ] `GAUSS_IDS=[[-1]]` is the closed-form sentinel, and layered shells repeat
      GAUSS_IDS and carry empty COMPONENTS segments. Same doc: "Closed-form META
      has a `GAUSS_IDS=[[-1]]` sentinel", "Layered-shell META — additional notes".
- [ ] Path ints in META/COMPONENTS are descriptor types, not indices. Same doc:
      "Path ints in META/COMPONENTS are descriptor types".
- [ ] `globalForce` and `localForce` name their components differently: `My`/`Mz`
      appear in both frames with different meanings. Same doc: "`globalForce` /
      `localForce` use different component-name conventions".
- [ ] **[lint Q2]** Scalar attrs (`STEP`, `TIME`, ...) are 1-element arrays. Unwrap
      them with `np.asarray(x).item()`, never `int(x)` / `float(x)`. `AGENTS.md`:
      "STKO scalar attrs are 1-element arrays".
- [ ] **[lint Q1]** Every text read of a sidecar (`.cdata`, `sections.tcl`) passes
      `encoding="utf-8"`, and ideally `errors="replace"`. `AGENTS.md`: "Text-mode
      file I/O: always pass `encoding=`".
- [ ] A new format convention belongs in `MpcoFormatPolicy` (`io/format_policy.py`)
      or `CDataFormatPolicy`, not inline. `docs/architecture-refactor-proposal.md`
      §2: "Format assumptions are named, not assumed".

## New element class or result bucket

- [ ] Gauss catalog entry plus shape functions in `format/`: see `docs/ElementResults.md`
      "Multi-dimensional integration points (shells & solids)". Import from
      `STKO_to_python.format.*`. The snippet there still shows the deprecated
      `utilities.*` path, and inside the library that deprecated import raises an
      error under the test warning filter (`AGENTS.md`, trap 3).
- [ ] A new column-name suffix (for example `_l<L>_ip<K>`) also needs the suffix
      stripper in `elements/canonical.py` (e8fae9d). `docs/api/canonical-names.md`:
      "Column-name suffix conventions".
- [ ] Pin the new bucket in `tests/integration/test_fixture_snapshots.py`.
      `tests/unit/io/test_meta_parser.py::test_parse_real_fixture_bucket`
      auto-discovers buckets in the fixtures, so read what it now covers.
- [ ] Multi-partition: exercise `elasticFrame/QuadFrame_results/` (partition-boundary
      nodes are duplicated). File handles go through `Hdf5PartitionPool`
      (`io/partition_pool.py`), not a raw `h5py.File`.

## Public surface

- [ ] No public name, import path or pickle layout changes without a deprecation shim.
      `docs/architecture.md` "Back-compat contract", "Pickle compatibility"
      (`tests/unit/test_public_api.py` pins `__module__` / `__qualname__`).
- [ ] Logging, not `print` (`docs/architecture-refactor-proposal.md` §2).

## Before the PR

- [ ] `PYTHONPATH=src MPLBACKEND=Agg python -m pytest tests -q` passes, and you have
      read the skip lines. `AGENTS.md` "Install, test, docs — exact commands".
- [ ] `python ci/check_quirk_patterns.py` is clean. If you touched docs, run
      `mkdocs build --strict` too.
- [ ] `CHANGELOG.md` `[Unreleased]` is updated. Every check is green. `AGENTS.md`:
      "Never merge on a red check".

Found a new trap? Add it to `docs/mpco_format_conventions.md` (format) or to `AGENTS.md`
"Lessons from incidents", then add a line here. If it is greppable and has a documented
incident, add a rule to `ci/check_quirk_patterns.py` instead.
