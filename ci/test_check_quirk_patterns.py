"""Self-test for ci/check_quirk_patterns.py.

One case per rule for the incident shape, the fix shape, every sanctioned
alternative found in the tree, and every hole a review finds. A hole found
is a test added.

Lives in ci/ on purpose: ``testpaths = ["tests"]`` keeps it out of the
library suite; CI runs it as its own step right before the lint itself
(``pytest -q ci/test_check_quirk_patterns.py``).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_quirk_patterns as cq  # noqa: E402


def lint(src: str, only: tuple[str, ...] = cq.RULES) -> list[cq.Finding]:
    return cq.check_source(textwrap.dedent(src), "mod.py", only)


def rules(src: str, only: tuple[str, ...] = cq.RULES) -> list[tuple[int, str]]:
    return [(f.line, f.rule) for f in lint(src, only)]


# ---------------------------------------------------------------- Q1 encoding
def test_q1_flags_incident_shape_pr57():
    # CDataReader before PR #57 (1f4c4c4): locale-encoded text read.
    src = """
        def read(file_path):
            with open(file_path, 'r') as file:
                return file.readlines()
    """
    assert rules(src, ("Q1",)) == [(3, "Q1")]


def test_q1_passes_fix_shape_pr57():
    src = """
        def read(file_path):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()
    """
    assert rules(src, ("Q1",)) == []


@pytest.mark.parametrize(
    "call",
    [
        "open(p)",  # default mode is text
        "open(p, 'w')",
        "open(p, mode='rt')",
        "io.open(p)",
        "Path(p).open()",
        "pathlib.Path(p).open('r')",
        "p.read_text()",
        "Path(p).read_text(errors='replace')",
        "p.write_text(s)",
    ],
)
def test_q1_flags_text_io_without_encoding(call):
    assert rules(f"import io, pathlib\nx = {call}\n", ("Q1",)) == [(2, "Q1")]


@pytest.mark.parametrize(
    "call",
    [
        "open(p, 'rb')",  # binary: no decoding
        "open(p, mode='wb')",
        "open(p, 'r', -1, 'utf-8')",  # encoding positional
        "Path(p).open('rb')",
        "Path(p).open('r', -1, 'utf-8')",
        "p.read_text('utf-8')",
        "p.write_text(s, 'utf-8')",
        "p.read_text(encoding='utf-8', errors='replace')",  # layered_section_reader
        "open(filename, 'r', encoding='utf-8', errors='ignore')",  # time_utils
    ],
)
def test_q1_passes_binary_or_explicit_encoding(call):
    assert rules(f"x = {call}\n", ("Q1",)) == []


@pytest.mark.parametrize(
    "call",
    [
        "open(p, mode)",  # variable mode: unreadable -> silent
        "open(*args)",
        "open(p, **kw)",
        "pool.open(0)",  # Hdf5PartitionPool.open(partition_idx)
        "self.open(idx)",
        "gzip.open(p, 'rb')",  # nodal_results pickle path
        "h5py.File(p, 'r')",
    ],
)
def test_q1_stays_silent_on_unreadable_or_foreign_open(call):
    assert rules(f"x = {call}\n", ("Q1",)) == []


def test_q1_multiline_call_is_one_finding_at_its_first_line():
    src = """
        f = open(
            path,
            "r",
        )
    """
    assert rules(src, ("Q1",)) == [(2, "Q1")]


# ------------------------------------------------------------------- Q2 attrs
def test_q2_flags_incident_shape_294b5fd():
    # ModelInfo._get_time_series_on_nodes_for_stage before 294b5fd.
    src = """
        def series(stage_group):
            out = {}
            for step_name, step_group in stage_group.items():
                step_value = step_group.attrs.get("STEP")
                time_value = step_group.attrs.get("TIME")
                if step_value is not None and time_value is not None:
                    out[int(step_value)] = float(time_value)
            return out
    """
    assert rules(src, ("Q2",)) == [(8, "Q2"), (8, "Q2")]


def test_q2_passes_fix_shape_294b5fd():
    src = """
        import numpy as np
        def series(stage_group):
            out = {}
            for step_name, step_group in stage_group.items():
                step_attr = step_group.attrs.get("STEP")
                time_attr = step_group.attrs.get("TIME")
                out[int(np.asarray(step_attr).item())] = float(
                    np.asarray(time_attr).item()
                )
            return out
    """
    assert rules(src, ("Q2",)) == []


@pytest.mark.parametrize(
    "expr",
    [
        "int(g.attrs['NUM_COLUMNS'])",
        "float(g.attrs.get('TIME'))",
        "float(f[stage].attrs['TIME'])",
    ],
)
def test_q2_flags_direct_conversion(expr):
    assert rules(f"def f(g, f, stage):\n    return {expr}\n", ("Q2",)) == [(2, "Q2")]


@pytest.mark.parametrize(
    "expr",
    [
        "int(np.asarray(g.attrs['STEP']).item())",
        "int(g.attrs['STEP'].item())",
        "int(g.attrs['STEP'][0])",
        "int(_scalar(g.attrs.get('NUM_COLUMNS')))",  # meta_parser helper
        "np.asarray(g.attrs['GP_X'])",  # element_manager GP_X read
        "len(g.attrs['GP_X'])",
        "int(g.attrs['STEP'], 10)",  # not the one-arg converter form
    ],
)
def test_q2_passes_sanctioned_unwraps(expr):
    assert rules(f"def f(g):\n    return {expr}\n", ("Q2",)) == []


def test_q2_skips_name_rebound_to_something_else():
    # Ambiguous: the rule cannot tell which binding reaches int(); stay silent.
    src = """
        import numpy as np
        def f(g):
            v = g.attrs["STEP"]
            v = np.asarray(v).item()
            return int(v)
    """
    assert rules(src, ("Q2",)) == []


def test_q2_skips_name_bound_in_enclosing_scope():
    src = """
        def outer(g):
            v = g.attrs["STEP"]
            def inner():
                return int(v)
            return inner
    """
    assert rules(src, ("Q2",)) == []


def test_q2_follows_walrus_binding():
    src = """
        def f(g):
            if (v := g.attrs.get("STEP")) is not None:
                return int(v)
    """
    assert rules(src, ("Q2",)) == [(4, "Q2")]


def test_q2_ignores_parameter_named_like_an_attr():
    src = """
        def f(step_value):
            return int(step_value)
    """
    assert rules(src, ("Q2",)) == []


# ------------------------------------------------------------------- waivers
def test_waiver_with_reason_suppresses_same_line():
    src = "x = open(p)  # stko-lint: encoding-ok ASCII-only generated log file\n"
    assert rules(src) == []


def test_waiver_on_line_above_suppresses():
    src = """
        # stko-lint: attrs-ok STEP written as a true 0-d scalar by this writer
        n = int(g.attrs["STEP"])
    """
    assert rules(src) == []


def test_waiver_without_reason_is_a_finding():
    src = "x = open(p)  # stko-lint: encoding-ok short\n"
    found = lint(src)
    assert [(f.line, f.rule) for f in found] == [(1, "Q1")]
    assert "reason" in found[0].message


def test_waiver_for_the_other_rule_does_not_suppress():
    src = "x = open(p)  # stko-lint: attrs-ok this is the wrong tag entirely\n"
    assert sorted(rules(src)) == [(1, "Q1"), (1, "STALE")]


def test_stale_waiver_is_a_finding():
    src = """
        # stko-lint: encoding-ok the call below was fixed long ago
        x = open(p, encoding="utf-8")
    """
    assert rules(src) == [(2, "STALE")]


def test_stale_check_respects_only():
    # With --only Q2, a Q1 waiver is not judged (its rule did not run).
    src = "x = open(p, encoding='utf-8')  # stko-lint: encoding-ok stale, not checked\n"
    assert rules(src, ("Q2",)) == []


# ---------------------------------------------------------------- driver/CLI
def _tree(tmp_path: Path, body: str) -> Path:
    pkg = tmp_path / "src" / "STKO_to_python"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text(textwrap.dedent(body), encoding="utf-8")
    egg = pkg / "STKO_to_python.egg-info"
    egg.mkdir()
    (egg / "junk.py").write_text("x = open(p)\n", encoding="utf-8")  # must be skipped
    return tmp_path


def test_cli_exit_codes_and_scope(tmp_path, capsys):
    root = _tree(tmp_path, "x = open(p)\n")
    assert cq.main(["--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "src/STKO_to_python/mod.py:1: Q1" in out
    assert "egg-info" not in out
    assert cq.main(["--root", str(root), "--only", "Q2"]) == 0


def test_cli_clean_tree_passes(tmp_path):
    root = _tree(tmp_path, "x = open(p, 'rb')\n")
    assert cq.main(["--root", str(root)]) == 0


def test_cli_rejects_unknown_rule(tmp_path):
    root = _tree(tmp_path, "x = 1\n")
    with pytest.raises(SystemExit):
        cq.main(["--root", str(root), "--only", "Q9"])


def test_parse_error_is_reported_not_swallowed():
    found = lint("def f(:\n")
    assert [f.rule for f in found] == ["PARSE"]
