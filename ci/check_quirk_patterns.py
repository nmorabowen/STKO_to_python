#!/usr/bin/env python3
"""Quirk-pattern lint for STKO_to_python. Dependency-free (stdlib ``ast``).

Turns lessons that name a greppable pattern AND have a documented incident
into CI checks, so a known trap fails CI instead of relying on someone
re-reading AGENTS.md. Why each rule exists, the mutation evidence and the
rejected alternatives: ``docs/agent-surface.md``.

Scope: library code only, ``src/STKO_to_python/**/*.py``. Tests, bench and
examples are out of scope -- the survey found only ASCII test fixtures there
(see the plan doc), and the incidents both lived in the library.

  Q1 encoding   A text-mode ``open()`` / ``io.open()`` / ``Path(...).open()``,
                or a ``.read_text()`` / ``.write_text()`` call, without an
                explicit encoding. Python's default text encoding on Windows
                is the ANSI code page (cp1252), so a non-ASCII byte in a
                ``.cdata`` / ``sections.tcl`` sidecar raises UnicodeDecodeError
                on the user's machine -- while the Ubuntu CI, whose locale is
                UTF-8, stays green. Only a static check can see it in CI.
                Incident: PR #57 (1f4c4c4), CHANGELOG 1.3.0 "Fixed".
                Binary modes are exempt. A mode or encoding the rule cannot
                read (a variable mode, ``*args``/``**kwargs``) is skipped,
                never guessed; so is ``X.open(...)`` for any receiver other
                than a literal ``Path(...)`` call (partition pool, gzip, ...).
                Waive (same line, or the line above the call):
                    # stko-lint: encoding-ok <reason>

  Q2 attrs      ``int()`` / ``float()`` applied directly to an h5py attribute
                value -- ``X.attrs[...]``, ``X.attrs.get(...)``, or a local name
                bound ONLY to one of those in the same function. STKO writes
                scalar attrs such as STEP/TIME as 1-element arrays, and
                ``int(np.array([5]))`` raises TypeError on numpy 2.x
                (deprecated since 1.25). Incident: 294b5fd (ModelInfo time
                series). Sanctioned: ``np.asarray(x).item()``, ``x.item()``,
                ``x[0]``, a helper such as ``meta_parser._scalar``. A name that
                is also bound to something else, or bound in an enclosing
                function, is skipped, never guessed.
                Waive (same line, or the line above the call):
                    # stko-lint: attrs-ok <reason>

A waiver needs a reason of at least 12 characters, and a waiver that no longer
suppresses anything is itself a finding (stale).

Usage:
    python ci/check_quirk_patterns.py              # all rules, exit 1 on any finding
    python ci/check_quirk_patterns.py --only Q1
    python ci/check_quirk_patterns.py --root DIR   # another tree (e.g. a git archive)
    python ci/check_quirk_patterns.py --list-waivers
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SCOPE = Path("src") / "STKO_to_python"
MIN_REASON = 12
RULES = ("Q1", "Q2")
WAIVER_TAG = {"Q1": "encoding-ok", "Q2": "attrs-ok"}
WAIVER = re.compile(r"#\s*stko-lint:\s*(encoding-ok|attrs-ok)\b(.*)$")
CONVERTERS = ("int", "float")


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
    used: bool = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Walk ``scope`` without descending into nested functions/classes/lambdas."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _scopes(tree: ast.Module) -> Iterator[ast.AST]:
    yield tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield node


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


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
    name = (
        f.id
        if isinstance(f, ast.Name)
        else f.attr
        if isinstance(f, ast.Attribute)
        else ""
    )
    return name.endswith("Path")


# --------------------------------------------------------------------------
# Q1 -- text-mode file I/O without an explicit encoding
# --------------------------------------------------------------------------
def _q1_call(call: ast.Call) -> str | None:
    """Return a finding message, or None (fine / not applicable / unreadable)."""
    f = call.func
    if _has_star(call) or _kw(call, "encoding") is not None:
        return None

    # (callable kind, index of the positional `mode`, index of positional `encoding`)
    if isinstance(f, ast.Name) and f.id == "open":
        kind, mode_idx, enc_idx = "open()", 1, 3
    elif (
        isinstance(f, ast.Attribute)
        and f.attr == "open"
        and isinstance(f.value, ast.Name)
        and f.value.id == "io"
    ):
        kind, mode_idx, enc_idx = "io.open()", 1, 3
    elif isinstance(f, ast.Attribute) and f.attr == "open" and _is_path_ctor(f.value):
        kind, mode_idx, enc_idx = "Path.open()", 0, 2
    elif isinstance(f, ast.Attribute) and f.attr == "read_text":
        kind, mode_idx, enc_idx = ".read_text()", None, 0
    elif isinstance(f, ast.Attribute) and f.attr == "write_text":
        kind, mode_idx, enc_idx = ".write_text()", None, 1
    else:
        return None

    if len(call.args) > enc_idx:  # encoding passed positionally
        return None
    if mode_idx is not None:
        mode_node = _kw(call, "mode")
        if mode_node is None and len(call.args) > mode_idx:
            mode_node = call.args[mode_idx]
        if mode_node is None:
            mode = "r"
        else:
            mode = _literal_str(mode_node)
            if mode is None:
                return None  # variable mode: cannot read it, stay silent
        if "b" in mode:
            return None
    return (
        f"text-mode {kind} without encoding= -- Windows defaults to cp1252 and "
        f'fails on non-ASCII sidecar bytes (#57); pass encoding="utf-8"'
    )


def check_q1(tree: ast.Module) -> list[tuple[int, int, str]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            msg = _q1_call(node)
            if msg:
                out.append((node.lineno, node.end_lineno or node.lineno, msg))
    return out


# --------------------------------------------------------------------------
# Q2 -- int()/float() on a raw h5py attribute value
# --------------------------------------------------------------------------
def _is_attrs_read(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Subscript):
        v = node.value
        return isinstance(v, ast.Attribute) and v.attr == "attrs"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ):
        v = node.func.value
        return isinstance(v, ast.Attribute) and v.attr == "attrs"
    return False


def _bindings(scope: ast.AST) -> dict[str, list[bool]]:
    """name -> [is_attrs_read, ...] for every binding of that name in ``scope``."""
    out: dict[str, list[bool]] = {}

    def add(target: ast.AST, from_attrs: bool) -> None:
        if isinstance(target, ast.Name):
            out.setdefault(target.id, []).append(from_attrs)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                add(elt, False)
        elif isinstance(target, ast.Starred):
            add(target.value, False)

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None:
                out.setdefault(arg.arg, []).append(False)
    for node in _own_nodes(scope):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add(t, _is_attrs_read(node.value) and isinstance(t, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            add(
                node.target,
                isinstance(node, ast.AnnAssign) and _is_attrs_read(node.value),
            )
        elif isinstance(node, ast.NamedExpr):
            add(node.target, _is_attrs_read(node.value))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            add(node.target, False)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    add(item.optional_vars, False)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            out.setdefault(node.name, []).append(False)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.setdefault((alias.asname or alias.name).split(".")[0], []).append(
                    False
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(node.name, []).append(False)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                out.setdefault(name, []).append(False)
    return out


def check_q2(tree: ast.Module) -> list[tuple[int, int, str]]:
    out = []
    for scope in _scopes(tree):
        bound = _bindings(scope)
        attrs_only = {n for n, kinds in bound.items() if kinds and all(kinds)}
        for node in _own_nodes(scope):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id not in CONVERTERS or len(node.args) != 1 or node.keywords:
                continue
            arg = node.args[0]
            direct = _is_attrs_read(arg)
            via_name = isinstance(arg, ast.Name) and arg.id in attrs_only
            if direct or via_name:
                what = (
                    "an h5py attrs read"
                    if direct
                    else f"'{arg.id}' (bound from an h5py attrs read)"
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
def _waivers(lines: list[str]) -> list[Waiver]:
    out = []
    for i, text in enumerate(lines, start=1):
        m = WAIVER.search(text)
        if m:
            out.append(Waiver(line=i, tag=m.group(1), reason=m.group(2).strip()))
    return out


def check_source(text: str, rel: str, only: tuple[str, ...] = RULES) -> list[Finding]:
    """Run the selected rules over one file's source. Pure; used by the self-test."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [Finding(rel, exc.lineno or 1, "PARSE", f"cannot parse: {exc.msg}")]
    lines = text.splitlines()
    waivers = _waivers(lines)
    findings: list[Finding] = []
    for rule in only:
        tag = WAIVER_TAG[rule]
        for start, end, msg in CHECKS[rule](tree):
            waived = False
            for w in waivers:
                if w.tag == tag and start - 1 <= w.line <= end:
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
        if any(part.endswith(".egg-info") or part == "__pycache__" for part in p.parts):
            continue
        yield p


def run(root: Path, only: tuple[str, ...] = RULES) -> list[Finding]:
    if not (root / SCOPE).is_dir():
        raise SystemExit(f"error: {root / SCOPE} does not exist (wrong --root?)")
    findings: list[Finding] = []
    for p in iter_files(root):
        text = p.read_text(encoding="utf-8")
        findings.extend(check_source(text, p.relative_to(root).as_posix(), only))
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
            for w in _waivers(p.read_text(encoding="utf-8").splitlines()):
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
