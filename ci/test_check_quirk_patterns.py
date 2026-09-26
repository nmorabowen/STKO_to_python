"""Self-test for ci/check_quirk_patterns.py.

One case per rule for the incident shape, the fix shape, every sanctioned
alternative found in the tree, and every hole a review finds. A hole found
is a test added; a one-line mutant of the lint that survives is a test added.

Lives in ci/ on purpose: ``testpaths = ["tests"]`` keeps it out of the
library suite; CI runs it as its own step right before the lint itself
(``pytest -q ci/test_check_quirk_patterns.py``).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

CI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CI_DIR))
import check_quirk_patterns as cq  # noqa: E402

WAIVE_Q1 = "# stko-lint: encoding-ok"  # assembled below so no line here is a waiver
WAIVE_Q2 = "# stko-lint: attrs-ok"


def lint(src: str | bytes, only: tuple[str, ...] = cq.RULES) -> list[cq.Finding]:
    if isinstance(src, str):
        src = textwrap.dedent(src)
    return cq.check_source(src, "mod.py", only)


def rules(src: str | bytes, only: tuple[str, ...] = cq.RULES) -> list[tuple[int, str]]:
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
        "open(p, encoding=None)",  # None is the locale default: same bug
        "open(p, 'r', -1, None)",
        "io.open(p)",
        "builtins.open(p)",
        "codecs.open(p)",  # encoding=None falls back to builtin text open
        "codecs.open(p, 'w')",
        "Path(p).open()",
        "pathlib.Path(p).open('r')",
        "p.read_text()",
        "p.read_text(encoding=None)",
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
        "open(p, 'rb+')",  # 'b' not last
        "open(p, 'r+b')",
        "open(p, 'r', -1, 'utf-8')",  # encoding positional
        "open(p, encoding=enc)",  # a variable encoding is an explicit one
        "builtins.open(p, 'rb')",
        "codecs.open(p, 'r', 'utf-8')",
        "codecs.open(p, encoding='utf-8')",
        "codecs.open(p, 'rb')",
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


def test_q1_resolves_a_name_bound_only_to_path():
    src = """
        from pathlib import Path
        def f(x):
            p = Path(x)
            with p.open() as fh:
                return fh.read()
    """
    assert rules(src, ("Q1",)) == [(5, "Q1")]


def test_q1_path_name_in_binary_mode_passes():
    src = """
        def f(x):
            p = Path(x)
            return p.open("rb").read()
    """
    assert rules(src, ("Q1",)) == []


@pytest.mark.parametrize(
    "body",
    [
        "def f(p):\n    return p.open()\n",  # a parameter: type unknown
        "def f(x, pool):\n    p = Path(x)\n    p = pool\n    return p.open()\n",
        "p = Path(x)\ndef f():\n    return p.open()\n",  # enclosing scope
    ],
)
def test_q1_skips_path_names_it_cannot_resolve(body):
    assert rules(body, ("Q1",)) == []


def test_q1_skips_a_parameter_rebound_to_path():
    # Known hole: the parameter is a second binding, so `p` is ambiguous.
    src = """
        def f(p):
            p = Path(p)
            return p.open()
    """
    assert rules(src, ("Q1",)) == []


def test_q1_sees_lambda_and_class_bodies():
    src = """
        read = lambda p: open(p).read()
        class Reader:
            HEADER = open("header.txt").readline()
    """
    assert rules(src, ("Q1",)) == [(2, "Q1"), (4, "Q1")]


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
        "int(g.attrs['STEP'][()])",  # [()] keeps the 1-element array
        "int(g.attrs['STEP'][...])",
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
        "int(g.attrs)",  # the mapping itself, not a value
    ],
)
def test_q2_passes_sanctioned_unwraps(expr):
    assert rules(f"def f(g):\n    return {expr}\n", ("Q2",)) == []


@pytest.mark.parametrize(
    "body",
    [
        "a = g.attrs\n    return int(a['STEP'])",
        "a = g.attrs\n    return float(a.get('TIME'))",
        "a = g.attrs\n    v = a['STEP']\n    return int(v)",
        "v = g.attrs['STEP'][()]\n    return int(v)",
        "v = g.attrs['STEP']\n    return int(v[()])",
        "v: Any = g.attrs['STEP']\n    return int(v)",  # annotated binding
    ],
)
def test_q2_follows_single_bindings(body):
    found = rules(f"def f(g):\n    {body}\n", ("Q2",))
    assert [rule for _, rule in found] == ["Q2"]


@pytest.mark.parametrize(
    "body",
    [
        "a = g.attrs\n    return int(a['STEP'][0])",
        "a = g.attrs\n    a = other\n    return int(a['STEP'])",  # ambiguous
        "v: int\n    v = g.attrs['STEP']\n    return int(v)",  # bare annotation
        "v = g.attrs['STEP']\n    v += 1\n    return int(v)",  # augmented
    ],
)
def test_q2_skips_what_it_cannot_resolve(body):
    assert rules(f"def f(g, other):\n    {body}\n", ("Q2",)) == []


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


def test_q2_skips_a_parameter_rebound_to_an_attrs_read():
    # Known hole, same as Q1: a parameter is a binding, so `v` is ambiguous.
    src = """
        def f(g, v):
            v = g.attrs["STEP"]
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


def test_q2_lambda_parameter_shadows_module_binding():
    src = """
        v = g.attrs["STEP"]
        to_int = lambda v: int(v)
    """
    assert rules(src, ("Q2",)) == []


def test_q2_sees_lambda_bodies():
    assert rules("f = lambda g: int(g.attrs['STEP'])\n", ("Q2",)) == [(1, "Q2")]


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


@pytest.mark.parametrize(
    "expr",
    [
        "int(df.attrs['n_rows'])",  # pandas DataFrame.attrs
        "int(self.attrs['count'])",
        "float(da.attrs.get('scale'))",  # xarray
    ],
)
def test_q2_assumes_every_attrs_is_h5py(expr):
    # DELIBERATE: Q2 does not know the type behind `.attrs`. The library reads
    # only h5py attrs and the history sweep shows no such noise; waive a real one.
    assert rules(f"def f(df, self, da):\n    return {expr}\n", ("Q2",)) == [(2, "Q2")]


def test_q2_flags_a_guarded_conversion_too():
    # DELIBERATE: the rule does not read guards; this is a waiver case.
    src = """
        def f(g):
            v = g.attrs["STEP"]
            if np.ndim(v) == 0:
                return int(v)
    """
    assert rules(src, ("Q2",)) == [(5, "Q2")]


# ------------------------------------------------------------------- waivers
def test_waiver_with_reason_suppresses_same_line():
    src = f"x = open(p)  {WAIVE_Q1} ASCII-only generated log file\n"
    assert rules(src) == []


def test_waiver_on_line_above_suppresses():
    src = f"{WAIVE_Q2} STEP written as a true 0-d scalar by this writer\n"
    src += 'n = int(g.attrs["STEP"])\n'
    assert rules(src) == []


def test_trailing_waiver_does_not_cover_the_next_line():
    src = f"a = open(p)  {WAIVE_Q1} ASCII-only generated log\nb = open(q)\n"
    assert rules(src) == [(2, "Q1")]


def test_comment_only_waiver_two_lines_above_does_not_suppress():
    src = f"{WAIVE_Q1} ASCII-only generated log file\n\nx = open(p)\n"
    assert rules(src) == [(1, "STALE"), (3, "Q1")]


def test_trailing_waiver_on_the_last_line_of_a_multiline_call():
    src = (
        f'f = open(\n    path,\n    "r",\n)  {WAIVE_Q1} ASCII-only generated log file\n'
    )
    assert rules(src) == []


def test_comment_only_waiver_inside_a_multiline_call():
    src = f"f = open(\n    {WAIVE_Q1} ASCII-only generated log file\n    path,\n)\n"
    assert rules(src) == []


def test_waiver_text_in_a_string_is_not_a_waiver():
    src = f'MSG = "{WAIVE_Q1} documented in a string literal"\nb = open(q)\n'
    assert rules(src) == [(2, "Q1")]


def test_waiver_syntax_in_a_docstring_is_not_stale():
    src = f'"""Waive with:\n\n    {WAIVE_Q1} <reason>\n"""\nx = 1\n'
    assert rules(src) == []


def test_waiver_after_another_comment_on_the_same_line():
    src = f"x = open(p)  # noqa: SIM115  {WAIVE_Q1} ASCII-only generated log\n"
    assert rules(src) == []


def test_the_lint_and_its_self_test_carry_no_stale_waivers():
    for name in ("check_quirk_patterns.py", "test_check_quirk_patterns.py"):
        found = cq.check_source((CI_DIR / name).read_bytes(), name)
        assert [f for f in found if f.rule == "STALE"] == [], name


def test_waiver_without_reason_is_a_finding():
    src = f"x = open(p)  {WAIVE_Q1} short\n"
    found = lint(src)
    assert [(f.line, f.rule) for f in found] == [(1, "Q1")]
    assert "reason" in found[0].message


@pytest.mark.parametrize(("n_chars", "expected"), [(11, [(1, "Q1")]), (12, [])])
def test_waiver_reason_length_boundary(n_chars, expected):
    assert rules(f"x = open(p)  {WAIVE_Q1} {'x' * n_chars}\n") == expected


def test_waiver_for_the_other_rule_does_not_suppress():
    src = f"x = open(p)  {WAIVE_Q2} this is the wrong tag entirely\n"
    assert sorted(rules(src)) == [(1, "Q1"), (1, "STALE")]


def test_stale_waiver_is_a_finding():
    src = f"{WAIVE_Q1} the call below was fixed long ago\n"
    src += 'x = open(p, encoding="utf-8")\n'
    assert rules(src) == [(1, "STALE")]


def test_stale_check_respects_only():
    # With --only Q2, a Q1 waiver is not judged (its rule did not run).
    src = f"x = open(p, encoding='utf-8')  {WAIVE_Q1} stale, not checked\n"
    assert rules(src, ("Q2",)) == []


# ------------------------------------------------------------ source decoding
def test_utf8_bom_is_not_a_parse_error():
    assert rules(b"\xef\xbb\xbfx = open(p)\n") == [(1, "Q1")]


def test_latin1_coding_cookie_is_honoured():
    src = b"# -*- coding: latin-1 -*-\nname = '\xe9'\nx = open(p)\n"
    assert rules(src) == [(3, "Q1")]


def test_waiver_in_a_bom_file():
    src = b"\xef\xbb\xbfx = open(p)  " + WAIVE_Q1.encode() + b" ASCII-only log file\n"
    assert rules(src) == []


def test_parse_error_is_reported_not_swallowed():
    found = lint("def f(:\n")
    assert [f.rule for f in found] == ["PARSE"]


@pytest.mark.parametrize(
    "src", [b"x = 1\x00\n", b"# -*- coding: bogus -*-\nx = 1\n", b"\xff\xfe x\n"]
)
def test_undecodable_source_is_a_parse_finding_not_a_crash(src):
    assert [(f.line, f.rule) for f in lint(src)] == [(1, "PARSE")]


# ---------------------------------------------------------------- driver/CLI
def _tree(tmp_path: Path, body: str) -> Path:
    pkg = tmp_path / "src" / "STKO_to_python"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "mod.py").write_text(textwrap.dedent(body), encoding="utf-8")
    (pkg / "sub" / "deep.py").write_text("y = open(q, 'rb')\n", encoding="utf-8")
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
    assert "2 files under src/STKO_to_python" in out
    assert cq.main(["--root", str(root), "--only", "Q2"]) == 0


def test_cli_recurses_into_subpackages(tmp_path, capsys):
    root = _tree(tmp_path, "x = 1\n")
    (root / "src" / "STKO_to_python" / "sub" / "deep.py").write_text(
        "y = open(q)\n", encoding="utf-8"
    )
    assert cq.main(["--root", str(root)]) == 1
    assert "src/STKO_to_python/sub/deep.py:1: Q1" in capsys.readouterr().out


def test_cli_reads_bom_and_latin1_files(tmp_path, capsys):
    root = _tree(tmp_path, "x = 1\n")
    pkg = root / "src" / "STKO_to_python"
    (pkg / "bom.py").write_bytes(b"\xef\xbb\xbfx = 1\n")
    (pkg / "lat.py").write_bytes(b"# -*- coding: latin-1 -*-\ns = '\xe9'\n")
    assert cq.main(["--root", str(root)]) == 0, capsys.readouterr().out


def test_real_tree_is_walked_recursively():
    repo = CI_DIR.parent
    files = [p.relative_to(repo).as_posix() for p in cq.iter_files(repo)]
    assert len(files) >= 100
    assert "src/STKO_to_python/io/meta_parser.py" in files
    assert not any(".egg-info" in f for f in files)


def test_cli_clean_tree_passes(tmp_path):
    root = _tree(tmp_path, "x = open(p, 'rb')\n")
    assert cq.main(["--root", str(root)]) == 0


def test_cli_rejects_unknown_rule(tmp_path):
    root = _tree(tmp_path, "x = 1\n")
    with pytest.raises(SystemExit):
        cq.main(["--root", str(root), "--only", "Q9"])


def test_list_waivers_reads_comments_only(tmp_path, capsys):
    root = _tree(tmp_path, f'M = "{WAIVE_Q1} not a comment"\n')
    pkg = root / "src" / "STKO_to_python"
    (pkg / "w.py").write_text(
        f"x = open(p)  {WAIVE_Q1} ASCII-only log\n", encoding="utf-8"
    )
    assert cq.main(["--root", str(root), "--list-waivers"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["src/STKO_to_python/w.py:1: encoding-ok ASCII-only log"]
