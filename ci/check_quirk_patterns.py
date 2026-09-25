#!/usr/bin/env python3
"""Quirk-pattern lint for STKO_to_python. Dependency-free (stdlib ``ast``).

Turns lessons that name a greppable pattern AND have a documented incident
into CI checks, so a known trap fails CI instead of relying on someone
re-reading AGENTS.md. Why each rule exists, the mutation evidence, the known
holes and the rejected alternatives: ``docs/agent-surface.md``.

Scope: library code only, ``src/STKO_to_python/**/*.py``. Tests, bench and
examples are out of scope -- the survey found only ASCII test fixtures there
(see the plan doc), and the incidents both lived in the library. Files are
read as bytes, so a BOM or a PEP 263 coding cookie is honoured.

  Q1 encoding   Text-mode file I/O without an explicit encoding: ``open()``,
                ``io.open()``, ``builtins.open()``, ``codecs.open()``,
                ``Path(...).open()`` or ``p.open()`` where ``p`` is bound only
                to a ``Path(...)`` call in the same scope, and any
                ``.read_text()`` / ``.write_text()``. ``encoding=None`` counts
                as missing. Python's default text encoding on Windows is the
                ANSI code page (cp1252), so a non-ASCII byte in a ``.cdata`` /
                ``sections.tcl`` sidecar raises UnicodeDecodeError on the
                user's machine -- while the Ubuntu CI, whose locale is UTF-8,
                stays green. Only a static check can see it in CI.
                Incident: PR #57 (1f4c4c4), CHANGELOG 1.3.0 "Fixed".
                Binary modes are exempt. A mode or encoding the rule cannot
                read (a variable, ``*args``/``**kwargs``) is skipped, never
                guessed; so is ``X.open(...)`` for any other receiver
                (partition pool, gzip, a Path parameter, ...).
                Waiver tag: ``encoding-ok``.

  Q2 attrs      ``int()`` / ``float()`` applied to a raw attribute value:
                ``X.attrs[...]``, ``X.attrs.get(...)``, the same through a
                name bound only to ``X.attrs``, a no-op ``[()]`` / ``[...]``
                index of any of these, or a local name bound ONLY to one of
                these in the same scope. STKO writes scalar attrs such as
                STEP/TIME as 1-element arrays, and ``int(np.array([5]))``
                raises TypeError on numpy 2.x (deprecated since 1.25).
                Incident: 294b5fd (ModelInfo time series). Sanctioned:
                ``np.asarray(x).item()``, ``x.item()``, ``x[0]``, a helper
                such as ``meta_parser._scalar``. A name that is also bound to
                something else, or bound in an enclosing scope, is skipped.
                DELIBERATE ASSUMPTION (this rule does not "stay silent" on
                it): every ``.attrs`` is h5py's. pandas ``DataFrame.attrs``,
                xarray, ``self.attrs`` and an ``int(v)`` guarded by
                ``np.ndim(v) == 0`` are flagged too. The library reads only
                h5py attrs, the 321-commit history sweep shows zero such
                noise, and a waiver covers a legitimate case.
                Waiver tag: ``attrs-ok``.

Waivers are Python COMMENTS (``tokenize`` COMMENT tokens; text inside a string
or docstring is never a waiver): ``stko-lint: <tag> <reason>`` after a ``#``.
A trailing comment waives a finding on any line of that statement's call; a
comment-only line waives the call that starts on the next line. A reason needs
at least 12 characters, and a waiver that no longer suppresses anything is
itself a finding (STALE).

Usage:
    python ci/check_quirk_patterns.py              # all rules, exit 1 on any finding
    python ci/check_quirk_patterns.py --only Q1
    python ci/check_quirk_patterns.py --root DIR   # another tree (e.g. a git archive)
    python ci/check_quirk_patterns.py --list-waivers
"""

from __future__ import annotations

import argparse
import ast
import functools
import io
import re
import sys
import tokenize
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

SCOPE = Path("src") / "STKO_to_python"
MIN_REASON = 12
RULES = ("Q1", "Q2")
WAIVER_TAG = {"Q1": "encoding-ok", "Q2": "attrs-ok"}
WAIVER = re.compile(r"#\s*stko-lint:\s*(encoding-ok|attrs-ok)\b(.*)$")
CONVERTERS = ("int", "float")
SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

# callee -> (index of positional `mode`, index of positional `encoding`)
MODULE_OPENS = {
    ("io", "open"): (1, 3),
    ("builtins", "open"): (1, 3),
    ("codecs", "open"): (1, 2),
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


@dataclass
class Waiver:
    line: int
    tag: str
    reason: str
    standalone: bool  # the comment is the only thing on its line
    used: bool = False

    def covers(self, start: int, end: int) -> bool:
        first = start - 1 if self.standalone else start
        return first <= self.line <= end


Source = str | bytes


# --------------------------------------------------------------------------
# scopes and bindings (shared by Q1 and Q2)
# --------------------------------------------------------------------------
def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk ``scope`` without descending into nested functions/classes/lambdas."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, SCOPE_NODES):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _scopes(tree: ast.Module) -> Iterator[ast.AST]:
    """The module plus every function, lambda and class body: each node of the
    tree is yielded by ``_own_nodes`` of exactly one of these."""
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, SCOPE_NODES):
            yield node


def _bindings(scope: ast.AST) -> dict[str, list[ast.AST | None]]:
    """name -> the value of every binding of it in ``scope``.

    A plain ``name = value`` (also annotated, also ``:=``) records ``value``;
    every other kind of binding (parameter, unpacking, loop / with / except
    target, import, def, class, global, augmented assignment, a bare
    annotation) records ``None``, which makes the name ambiguous.
    """
    out: dict[str, list[ast.AST | None]] = {}

    def add(target: ast.AST, value: ast.AST | None) -> None:
        if isinstance(target, ast.Name):
            out.setdefault(target.id, []).append(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                add(elt, None)
        elif isinstance(target, ast.Starred):
            add(target.value, None)

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None:
                out.setdefault(arg.arg, []).append(None)
    for node in _own_nodes(scope):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add(t, node.value)
        elif isinstance(node, ast.AnnAssign):
            add(node.target, node.value)
        elif isinstance(node, ast.AugAssign):
            add(node.target, None)
        elif isinstance(node, ast.NamedExpr):
            add(node.target, node.value)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            add(node.target, None)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    add(item.optional_vars, None)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.setdefault(node.name, []).append(None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                out.setdefault(name, []).append(None)
        elif isinstance(node, SCOPE_NODES) and not isinstance(node, ast.Lambda):
            out.setdefault(node.name, []).append(None)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                out.setdefault(name, []).append(None)
    return out


def _names_bound_only_to(
    bound: dict[str, list[ast.AST | None]], pred: Callable[[ast.AST], bool]
) -> set[str]:
    return {
        n
        for n, values in bound.items()
        if values and all(v is not None and pred(v) for v in values)
    }


# --------------------------------------------------------------------------
# Q1 -- text-mode file I/O without an explicit encoding
# --------------------------------------------------------------------------
def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_none(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _has_star(call: ast.Call) -> bool:
    return any(isinstance(a, ast.Starred) for a in call.args) or any(
        k.arg is None for k in call.keywords
    )


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _is_path_ctor(node: ast.AST) -> bool:
    """``Path(...)`` or ``pathlib.Path(...)`` (also PurePath/WindowsPath/...)."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return f.id.endswith("Path")
    if isinstance(f, ast.Attribute):
        return f.attr.endswith("Path")
    return False


def _q1_kind(f: ast.AST, path_names: set[str]) -> tuple[str, int | None, int] | None:
    """(label, positional mode index or None, positional encoding index)."""
    if isinstance(f, ast.Name):
        return ("open()", 1, 3) if f.id == "open" else None
    if not isinstance(f, ast.Attribute):
        return None
    if f.attr == "read_text":
        return ".read_text()", None, 0
    if f.attr == "write_text":
        return ".write_text()", None, 1
    if f.attr != "open":
        return None
    recv = f.value
    if isinstance(recv, ast.Name) and (recv.id, "open") in MODULE_OPENS:
        mode_idx, enc_idx = MODULE_OPENS[(recv.id, "open")]
        return f"{recv.id}.open()", mode_idx, enc_idx
    if _is_path_ctor(recv) or (isinstance(recv, ast.Name) and recv.id in path_names):
        return "Path.open()", 0, 2
    return None


def _q1_call(call: ast.Call, path_names: set[str]) -> str | None:
    """Return a finding message, or None (fine / not applicable / unreadable)."""
    kind = _q1_kind(call.func, path_names)
    if kind is None or _has_star(call):
        return None
    label, mode_idx, enc_idx = kind

    enc = _kw(call, "encoding")
    if enc is None and len(call.args) > enc_idx:
        enc = call.args[enc_idx]
    if enc is not None and not _is_none(enc):
        return None  # an explicit encoding (literal or variable)

    if mode_idx is not None:
        mode_node = _kw(call, "mode")
        if mode_node is None and len(call.args) > mode_idx:
            mode_node = call.args[mode_idx]
        mode = "r" if mode_node is None else _literal_str(mode_node)
        if mode is None:
            return None  # variable mode: cannot read it, stay silent
        if "b" in mode:
            return None
    return (
        f"text-mode {label} without encoding= -- Windows defaults to cp1252 and "
        f'fails on non-ASCII sidecar bytes (#57); pass encoding="utf-8"'
    )


def check_q1(tree: ast.Module) -> list[tuple[int, int, str]]:
    out = []
    for scope in _scopes(tree):
        path_names = _names_bound_only_to(_bindings(scope), _is_path_ctor)
        for node in _own_nodes(scope):
            if isinstance(node, ast.Call):
                msg = _q1_call(node, path_names)
                if msg:
                    out.append((node.lineno, node.end_lineno or node.lineno, msg))
    return out


# --------------------------------------------------------------------------
# Q2 -- int()/float() on a raw (h5py) attribute value
# --------------------------------------------------------------------------
def _strip_noop_index(node: ast.AST) -> ast.AST:
    """``x[()]`` and ``x[...]`` return ``x`` unchanged for an ndarray."""
    while isinstance(node, ast.Subscript) and (
        (isinstance(node.slice, ast.Tuple) and not node.slice.elts)
        or (isinstance(node.slice, ast.Constant) and node.slice.value is Ellipsis)
    ):
        node = node.value
    return node


def _is_attrs_mapping(node: ast.AST) -> bool:
    """``X.attrs`` -- assumed to be an h5py AttributeManager (see Q2)."""
    return isinstance(node, ast.Attribute) and node.attr == "attrs"


def _is_attrs_obj(node: ast.AST, attrs_objs: set[str]) -> bool:
    """``X.attrs``, or a name bound only to one."""
    if _is_attrs_mapping(node):
        return True
    return isinstance(node, ast.Name) and node.id in attrs_objs


def _is_attrs_read(node: ast.AST, attrs_objs: set[str]) -> bool:
    node = _strip_noop_index(node)
    if isinstance(node, ast.Subscript):
        return _is_attrs_obj(node.value, attrs_objs)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ):
        return _is_attrs_obj(node.func.value, attrs_objs)
    return False


def check_q2(tree: ast.Module) -> list[tuple[int, int, str]]:
    out = []
    for scope in _scopes(tree):
        bound = _bindings(scope)
        attrs_objs = _names_bound_only_to(bound, _is_attrs_mapping)
        attrs_values = _names_bound_only_to(
            bound, functools.partial(_is_attrs_read, attrs_objs=attrs_objs)
        )
        for node in _own_nodes(scope):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in CONVERTERS or len(node.args) != 1 or node.keywords:
                continue
            arg = node.args[0]
            direct = _is_attrs_read(arg, attrs_objs)
            inner = _strip_noop_index(arg)
            via_name = isinstance(inner, ast.Name) and inner.id in attrs_values
            if direct or via_name:
                what = (
                    "an h5py attrs read"
                    if direct
                    else f"'{inner.id}' (bound from an h5py attrs read)"
                )
                out.append(
                    (
                        node.lineno,
                        node.end_lineno or node.lineno,
                        f"{node.func.id}() on {what} -- STKO writes scalar attrs "
                        "as 1-element arrays; numpy 2.x raises TypeError "
                        "(294b5fd). Use np.asarray(x).item()",
                    )
                )
    return out


CHECKS = {"Q1": check_q1, "Q2": check_q2}


# --------------------------------------------------------------------------
# waivers + driver
# --------------------------------------------------------------------------
def _tokens(source: Source) -> Iterator[tokenize.TokenInfo]:
    if isinstance(source, bytes):
        return tokenize.tokenize(io.BytesIO(source).readline)
    return tokenize.generate_tokens(io.StringIO(source).readline)


def _waivers(source: Source) -> list[Waiver]:
    """Waivers from COMMENT tokens only (never from strings or docstrings)."""
    out = []
    try:
        for tok in _tokens(source):
            if tok.type != tokenize.COMMENT:
                continue
            m = WAIVER.search(tok.string)
            if m:
                out.append(
                    Waiver(
                        line=tok.start[0],
                        tag=m.group(1),
                        reason=m.group(2).strip(),
                        standalone=not tok.line[: tok.start[1]].strip(),
                    )
                )
    except (tokenize.TokenError, SyntaxError):
        return out  # the ast.parse in check_source reports the file as PARSE
    return out


def check_source(
    source: Source, rel: str, only: tuple[str, ...] = RULES
) -> list[Finding]:
    """Run the selected rules over one file. ``source`` is the file's bytes
    (BOM and coding cookie honoured) or, in the self-test, a str. Pure."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # also null bytes and a bad coding cookie (3.11+)
        return [Finding(rel, exc.lineno or 1, "PARSE", f"cannot parse: {exc.msg}")]
    waivers = _waivers(source)
    findings: list[Finding] = []
    for rule in only:
        tag = WAIVER_TAG[rule]
        for start, end, msg in CHECKS[rule](tree):
            waived = False
            for w in waivers:
                if w.tag == tag and w.covers(start, end):
                    w.used = True
                    if len(w.reason) < MIN_REASON:
                        findings.append(
                            Finding(
                                rel,
                                w.line,
                                rule,
                                f"waiver '{tag}' needs a reason of at least "
                                f"{MIN_REASON} characters",
                            )
                        )
                    waived = True
            if not waived:
                findings.append(Finding(rel, start, rule, msg))
    active_tags = {WAIVER_TAG[r] for r in only}
    for w in waivers:
        if w.tag in active_tags and not w.used:
            findings.append(
                Finding(
                    rel,
                    w.line,
                    "STALE",
                    f"waiver '{w.tag}' suppresses nothing -- delete it",
                )
            )
    return sorted(findings, key=lambda f: (f.path, f.line, f.rule))


def iter_files(root: Path) -> Iterator[Path]:
    base = root / SCOPE
    for p in sorted(base.rglob("*.py")):
        if any(part.endswith(".egg-info") for part in p.parts):
            continue
        yield p


def run(root: Path, only: tuple[str, ...] = RULES) -> list[Finding]:
    if not (root / SCOPE).is_dir():
        raise SystemExit(f"error: {root / SCOPE} does not exist (wrong --root?)")
    findings: list[Finding] = []
    for p in iter_files(root):
        rel = p.relative_to(root).as_posix()
        findings.extend(check_source(p.read_bytes(), rel, only))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--only", default=",".join(RULES), help="comma-separated rule ids")
    ap.add_argument("--list-waivers", action="store_true")
    args = ap.parse_args(argv)

    only = tuple(r.strip().upper() for r in args.only.split(",") if r.strip())
    unknown = [r for r in only if r not in RULES]
    if unknown:
        ap.error(f"unknown rule(s): {', '.join(unknown)}")

    root = args.root.resolve()
    if args.list_waivers:
        for p in iter_files(root):
            for w in _waivers(p.read_bytes()):
                print(f"{p.relative_to(root).as_posix()}:{w.line}: {w.tag} {w.reason}")
        return 0

    findings = run(root, only)
    for f in findings:
        print(f)
    n_files = sum(1 for _ in iter_files(root))
    status = "FAIL" if findings else "ok"
    print(
        f"quirk-pattern lint: {status} -- {len(findings)} finding(s), "
        f"rules {','.join(only)}, {n_files} files under {SCOPE.as_posix()}"
    )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
