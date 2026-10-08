"""Baselines: two static analysers and a classical machine-learning pipeline.

The project's motivation is that *existing static analysis tools often produce
high false-positive rates*.  To test that claim honestly the comparison must
not be against a strawman, so two static analysers of increasing sophistication
are implemented:

M1  Pattern-matching analyser.
    Reimplements the published semgrep rule family
    `python.lang.security.audit.formatted-sql-query` / Bandit B608
    ("hardcoded_sql_expressions"): flag a query argument of an `execute`-like
    call whenever that argument is produced by f-string interpolation,
    `%`-formatting, `str.format` or `+` concatenation.  This is what most
    CI-integrated tooling actually does.

M2  AST taint analyser.
    A genuine intra-procedural forward taint analysis: function parameters and
    request objects are sources, assignments / string operations propagate
    taint, `int()` and `float()` are recognised sanitisers, and `execute`-like
    calls are sinks.  It resolves variables to their definitions, so it follows
    multi-hop flows that M1 cannot see.  This is a fair stand-in for the
    taint-mode rules of commercial scanners.

M3  TF-IDF + logistic regression over the same token stream the neural models
    receive, so any gain the neural models show is attributable to the
    architecture and not to better lexical preprocessing.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

SQL_DML = re.compile(
    r"\b(select|insert|update|delete|union|drop|alter|truncate|exec|call)\b[\s\S]*?"
    r"\b(from|into|set|where|values|table|join)\b", re.I)
SQL_HINT = re.compile(r"\b(select|insert\s+into|update|delete\s+from|exec|call)\b", re.I)

SINK_ATTRS = {"execute", "executemany", "executescript", "executemany",
              "raw", "callproc", "extra", "execute_sql"}
SINK_FUNCS = {"text"}

SANITISERS = {"int", "float", "operator.index", "len", "bool", "abs", "round"}

TAINT_SOURCE_ROOTS = {"request", "flask", "message", "event", "payload",
                      "raw_body", "body", "form", "args", "environ", "input"}


# --------------------------------------------------------------------------- #
# Shared AST helpers
# --------------------------------------------------------------------------- #

def _attr_chain(node: ast.AST) -> str:
    """Dotted name for Name/Attribute chains, else ''."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        return _attr_chain(node.func)
    return ""


def _is_sink(call: ast.Call) -> bool:
    if isinstance(call.func, ast.Attribute) and call.func.attr in SINK_ATTRS:
        return True
    if isinstance(call.func, ast.Name) and call.func.id in SINK_FUNCS:
        return True
    return False


def _contains_sql(node: ast.AST) -> bool:
    """True if any string constant under `node` looks like SQL."""
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if SQL_HINT.search(n.value):
                return True
        if isinstance(n, ast.JoinedStr):
            txt = "".join(v.value for v in n.values
                          if isinstance(v, ast.Constant) and isinstance(v.value, str))
            if SQL_HINT.search(txt):
                return True
    return False


def _assignments(tree: ast.AST) -> Dict[str, List[ast.AST]]:
    """name -> list of right-hand sides assigned to it anywhere in the tree."""
    out: Dict[str, List[ast.AST]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(n.value)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            out.setdefault(n.target.id, []).append(n.value)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value:
            out.setdefault(n.target.id, []).append(n.value)
    return out


def _dynamic_string_build(node: ast.AST) -> bool:
    """True if `node` builds a string by interpolation / concatenation."""
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "format", "format_map", "join", "substitute", "safe_substitute"}:
            return True
    return False


# --------------------------------------------------------------------------- #
# M1 -- pattern-matching static analyser (semgrep / Bandit B608 class)
# --------------------------------------------------------------------------- #

def static_regex_predict(code: str) -> int:
    """1 if the rule fires (predicted vulnerable), else 0."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return int(bool(SQL_DML.search(code)) and ("%" in code or ".format(" in code))

    defs = _assignments(tree)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_sink(node) and node.args):
            continue
        q = node.args[0]
        # Inline construction in the call itself.
        if _dynamic_string_build(q) and _contains_sql(q):
            return 1
        # Query held in a variable: inspect its definitions.
        if isinstance(q, ast.Name):
            for rhs in defs.get(q.id, []):
                if _dynamic_string_build(rhs) and _contains_sql(rhs):
                    return 1
                # `q = "..." ; q += ...`  -- any augmented build counts.
                if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, (ast.Add, ast.Mod)):
                    return 1
    return 0


# --------------------------------------------------------------------------- #
# M2 -- intra-procedural AST taint analyser
# --------------------------------------------------------------------------- #

class TaintAnalyser:
    """Forward, flow-insensitive taint propagation over one function."""

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self.defs = _assignments(tree)
        self.tainted: Set[str] = set()
        self._seed_sources()
        self._propagate()

    # ---- sources ---------------------------------------------------------- #
    def _seed_sources(self) -> None:
        for n in ast.walk(self.tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = n.args
                for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                    if arg.arg != "self":
                        self.tainted.add(arg.arg)
                if a.vararg:
                    self.tainted.add(a.vararg.arg)
                if a.kwarg:
                    self.tainted.add(a.kwarg.arg)

    # ---- propagation ------------------------------------------------------ #
    def _expr_tainted(self, node: Optional[ast.AST]) -> bool:
        if node is None:
            return False
        # A recognised sanitiser launders whatever it wraps.
        if isinstance(node, ast.Call):
            fname = _attr_chain(node.func)
            if fname in SANITISERS or fname.split(".")[-1] in {"int", "float"}:
                return False
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in self.tainted:
                return True
            if isinstance(n, ast.Attribute):
                root = _attr_chain(n).split(".")[0]
                if root in TAINT_SOURCE_ROOTS:
                    return True
            if isinstance(n, ast.Name) and n.id in TAINT_SOURCE_ROOTS:
                return True
        return False

    def _propagate(self) -> None:
        # Iterate to a fixed point: handles chains  a = p; b = a; c = b; ...
        for _ in range(8):
            grew = False
            for name, rhss in self.defs.items():
                if name in self.tainted:
                    continue
                if any(self._expr_tainted(r) for r in rhss):
                    self.tainted.add(name)
                    grew = True
            # For-loop targets inherit the taint of the iterable, e.g.
            # `for col, val in filters.items():` with `filters` a parameter.
            for n in ast.walk(self.tree):
                if isinstance(n, (ast.For, ast.AsyncFor)) and self._expr_tainted(n.iter):
                    for t in ast.walk(n.target):
                        if isinstance(t, ast.Name) and t.id not in self.tainted:
                            self.tainted.add(t.id)
                            grew = True
            # Comprehension targets, e.g. `[f"{k}={v}" for k, v in flt.items()]`
            for n in ast.walk(self.tree):
                if isinstance(n, (ast.ListComp, ast.SetComp, ast.GeneratorExp,
                                  ast.DictComp)):
                    for gen in n.generators:
                        if self._expr_tainted(gen.iter):
                            for t in ast.walk(gen.target):
                                if isinstance(t, ast.Name) and t.id not in self.tainted:
                                    self.tainted.add(t.id)
                                    grew = True
            # Loop-carried accumulation: `parts.append(f"...{val}...")`
            for n in ast.walk(self.tree):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr in {"append", "extend", "add"}
                        and isinstance(n.func.value, ast.Name)
                        and n.func.value.id not in self.tainted
                        and any(self._expr_tainted(a) for a in n.args)):
                    self.tainted.add(n.func.value.id)
                    grew = True
            if not grew:
                break

    # ---- sinks ------------------------------------------------------------ #
    def flows_to_sink(self) -> bool:
        for node in ast.walk(self.tree):
            if not (isinstance(node, ast.Call) and _is_sink(node)):
                continue
            if node.args and self._query_arg_tainted(node.args[0]):
                return True
            # Query supplied by keyword, e.g. `.extra(where=[...], select={...})`.
            for kw in node.keywords:
                if kw.arg in {"where", "select", "tables", "sql", "query", "statement"}:
                    if self._query_arg_tainted(kw.value):
                        return True
                    if isinstance(kw.value, (ast.List, ast.Tuple)):
                        if any(self._query_arg_tainted(e) for e in kw.value.elts):
                            return True
        return False

    def _query_arg_tainted(self, q: ast.AST, depth: int = 0) -> bool:
        if depth > 6:
            return False
        # Tainted data bound as a *parameter* is safe; only the query string
        # itself matters, which is why we inspect argument 0 alone.
        if isinstance(q, ast.JoinedStr):
            return any(self._expr_tainted(v.value) for v in q.values
                       if isinstance(v, ast.FormattedValue))
        if isinstance(q, ast.BinOp) and isinstance(q.op, (ast.Mod, ast.Add)):
            return self._expr_tainted(q.left) or self._expr_tainted(q.right)
        if isinstance(q, ast.Call):
            fn = q.func
            if isinstance(fn, ast.Attribute) and fn.attr in {
                    "format", "format_map", "substitute", "safe_substitute", "join"}:
                if any(self._expr_tainted(a) for a in q.args):
                    return True
                if any(self._expr_tainted(k.value) for k in q.keywords):
                    return True
                if fn.attr == "join" and q.args:
                    return self._expr_tainted(q.args[0])
            # A nested builder call: follow its arguments.
            if any(self._expr_tainted(a) for a in q.args) and _contains_sql(q):
                return True
        if isinstance(q, ast.Name):
            for rhs in self.defs.get(q.id, []):
                if self._query_arg_tainted(rhs, depth + 1):
                    return True
            # Query assembled by a local helper that returns an f-string.
            if self._helper_returns_tainted(q.id, depth):
                return True
        return False

    def _helper_returns_tainted(self, name: str, depth: int) -> bool:
        for rhs in self.defs.get(name, []):
            if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Name):
                target = rhs.func.id
                arg_tainted = any(self._expr_tainted(a) for a in rhs.args)
                for n in ast.walk(self.tree):
                    if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and n.name == target):
                        for r in ast.walk(n):
                            if isinstance(r, ast.Return) and r.value is not None:
                                if arg_tainted and _dynamic_string_build(r.value) \
                                        and _contains_sql(r.value):
                                    return True
        return False


def static_taint_predict(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return static_regex_predict(code)
    try:
        return int(TaintAnalyser(tree).flows_to_sink())
    except RecursionError:
        return 0


# --------------------------------------------------------------------------- #
# M3 -- TF-IDF + logistic regression over the shared token stream
# --------------------------------------------------------------------------- #

def build_tfidf_lr(token_strings_train: Sequence[str], y_train: np.ndarray,
                   C: float = 4.0, seed: int = 0, max_features: int = 60000):
    """Fit TF-IDF (token 1-2 grams) + L2 logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+",
                                  ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                                  max_features=max_features)),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=seed,
                                   solver="liblinear")),
    ])
    pipe.fit(list(token_strings_train), y_train)
    return pipe
