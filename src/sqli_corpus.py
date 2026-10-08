"""Synthetic, taint-annotated Python SQL-injection corpus generator.

Why synthesise?
---------------
The datasets named in the project brief (Juliet/SARD, Devign, Big-Vul,
CodeSearchNet) are either C/C++/Java, or they carry only *function-level*
binary labels.  This project needs three things at once:

  1. a binary vulnerability label,
  2. a **statement-level taint annotation** (which lines carry the
     source -> sink flow) so that the attention mechanism can be *guided*, and
  3. a **vulnerability-family label** so that neural code *search* has a
     ground-truth notion of "semantically similar code".

No public corpus supplies all three for Python, so we generate them from a
taint-aware grammar.  Because the generator knows, by construction, which
statements form the taint path, annotation (2) is exact and free of the label
noise that plagues mined vulnerability datasets.

Design rules that keep the task hard and honest
-----------------------------------------------
* Vulnerable and safe samples are drawn from the *same* identifier, table,
  column, docstring, comment and decorator pools, so no name is predictive.
* Comments are deliberately adversarial: "# input is already sanitised"
  appears on both vulnerable and safe functions.
* 8 of the 15 safe families are **hard negatives**: safe code that still
  interpolates a value into a SQL string (integer casts, allow-listed
  identifiers, regex-validated inputs, literal-only f-strings, LIKE wildcards
  passed as a *parameter*).  These are precisely the patterns that make
  regular-expression and pattern-matching static analysers emit false
  positives.
* Every emitted sample is checked with `ast.parse`, so the corpus is 100%
  syntactically valid Python.
"""

from __future__ import annotations

import ast
import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .vocabpools import (
    API_SPECS, COLUMNS, COMMENTS_MISLEADING, COMMENTS_NEUTRAL, CONN_VARS,
    CURSOR_VARS, DECORATORS, DISTRACTORS, DOCSTRINGS, LOCAL_VARS, LOGGERS,
    NOUNS, QUALIFIERS, QUERY_VARS, TABLES, USER_PARAMS, VERBS, VERBS_WRITE,
)

Q = '"'          # double-quote literal, keeps the templates below readable
SQ = "'"         # single-quote literal (SQL string delimiter)

# --------------------------------------------------------------------------- #
# Family registry
# --------------------------------------------------------------------------- #

VULN_FAMILIES: List[str] = [
    "V01_fstring_direct", "V02_percent_operator", "V03_str_format",
    "V04_concat_plus", "V05_multi_hop_alias", "V06_helper_builder",
    "V07_dict_request_field", "V08_loop_built_where", "V09_orm_raw_text",
    "V10_executescript", "V11_template_substitute", "V12_order_by_identifier",
    "V13_join_fragments", "V14_partial_parameterisation", "V15_useless_sanitiser",
]

SAFE_FAMILIES: List[str] = [
    "S01_qmark_params", "S02_pyformat_params", "S03_named_params",
    "S04_sqlalchemy_bindparams", "S05_orm_query_builder", "S06_int_cast",
    "S07_allowlisted_identifier", "S08_enum_fragment_lookup",
    "S09_regex_validated", "S10_literal_only_fstring", "S11_executemany_params",
    "S12_sql_string_not_executed", "S13_fstring_non_sql_sink",
    "S14_stored_procedure", "S15_composable_identifier_api",
]

ALL_FAMILIES = VULN_FAMILIES + SAFE_FAMILIES

# Safe families whose *surface form* still drops a value into a SQL string.
HARD_NEGATIVE_FAMILIES = {
    "S06_int_cast", "S07_allowlisted_identifier", "S08_enum_fragment_lookup",
    "S09_regex_validated", "S10_literal_only_fstring",
    "S12_sql_string_not_executed", "S13_fstring_non_sql_sink",
    "S15_composable_identifier_api",
}

# Difficulty: how far the sample is from a textbook pattern-match.
DIFFICULTY = {
    "V01_fstring_direct": "easy", "V02_percent_operator": "easy",
    "V03_str_format": "easy", "V04_concat_plus": "easy",
    "V05_multi_hop_alias": "medium", "V06_helper_builder": "hard",
    "V07_dict_request_field": "medium", "V08_loop_built_where": "hard",
    "V09_orm_raw_text": "medium", "V10_executescript": "medium",
    "V11_template_substitute": "hard", "V12_order_by_identifier": "hard",
    "V13_join_fragments": "hard", "V14_partial_parameterisation": "hard",
    "V15_useless_sanitiser": "hard",
    "S01_qmark_params": "easy", "S02_pyformat_params": "easy",
    "S03_named_params": "easy", "S04_sqlalchemy_bindparams": "medium",
    "S05_orm_query_builder": "easy", "S06_int_cast": "hard",
    "S07_allowlisted_identifier": "hard", "S08_enum_fragment_lookup": "hard",
    "S09_regex_validated": "hard", "S10_literal_only_fstring": "hard",
    "S11_executemany_params": "medium", "S12_sql_string_not_executed": "hard",
    "S13_fstring_non_sql_sink": "medium", "S14_stored_procedure": "medium",
    "S15_composable_identifier_api": "hard",
}


@dataclass
class Sample:
    code: str
    label: int                          # 1 = SQL-injection vulnerable, 0 = safe
    family: str
    api: str
    rationale_lines: Tuple[int, ...]    # 0-based indices of taint-path statements
    difficulty: str
    cwe: str
    n_lines: int
    hard_negative: int
    uid: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d["rationale_lines"] = ",".join(str(i) for i in self.rationale_lines)
        return d


# --------------------------------------------------------------------------- #
# Function builder
# --------------------------------------------------------------------------- #

class Fn:
    """Accumulates (indent, text, is_rationale) records for one function."""

    def __init__(self) -> None:
        self.rec: List[Tuple[int, str, bool]] = []

    def add(self, text: str, indent: int = 1, rationale: bool = False) -> None:
        self.rec.append((indent, text, rationale))

    def blank(self) -> None:
        self.rec.append((0, "", False))

    def render(self) -> Tuple[str, Tuple[int, ...]]:
        out: List[str] = []
        rat: List[int] = []
        for i, (ind, txt, r) in enumerate(self.rec):
            out.append(("    " * ind + txt) if txt else "")
            if r:
                rat.append(i)
        return "\n".join(out), tuple(rat)


@dataclass
class Ctx:
    """Every random naming decision for one sample, so families stay readable."""
    rng: random.Random
    api: str
    fn_name: str
    user_param: str
    extra_params: List[str]
    table: str
    cols: List[str]
    key_col: str
    quoted_key: bool
    q: str          # query variable name
    cur: str        # cursor variable name
    conn: str       # connection variable name
    res: str        # result variable name
    noun: str
    logger: str
    ind: int = 1    # base indentation for family bodies
    sig_extra: List[str] = None   # extra parameters a family needs

    def __post_init__(self):
        if self.sig_extra is None:
            self.sig_extra = []

    @property
    def spec(self) -> dict:
        return API_SPECS[self.api]

    @property
    def ph(self) -> str:
        return self.spec["qmark"]

    @property
    def col_list(self) -> str:
        return ", ".join(self.cols)


def _mk_ctx(rng: random.Random, api: Optional[str] = None) -> Ctx:
    api = api or rng.choice(list(API_SPECS))
    noun = rng.choice(NOUNS)
    verb = rng.choice(VERBS if rng.random() < 0.8 else VERBS_WRITE)
    qual = rng.choice(QUALIFIERS)
    fn_name = "_".join(x for x in (verb, noun, qual) if x)
    key_col = rng.choice(["id", "username", "email", "status", "sku", "name",
                          "country", "role", "account_no", "category_id"])
    quoted_key = key_col not in ("id", "category_id", "quantity")
    return Ctx(
        rng=rng,
        api=api,
        fn_name=fn_name,
        user_param=rng.choice(USER_PARAMS),
        extra_params=rng.sample(["limit", "offset", "tenant", "include_archived",
                                 "fields", "page"], rng.randint(0, 2)),
        table=rng.choice(TABLES),
        cols=rng.sample(COLUMNS, rng.randint(2, 5)),
        key_col=key_col,
        quoted_key=quoted_key,
        q=rng.choice(QUERY_VARS),
        cur=rng.choice(CURSOR_VARS),
        conn=rng.choice(CONN_VARS),
        res=rng.choice(LOCAL_VARS),
        noun=noun,
        logger=rng.choice(LOGGERS),
    )


# --------------------------------------------------------------------------- #
# SQL shapes.  The same SQL skeletons are reused by vulnerable *and* safe
# families, so the SQL vocabulary itself is never predictive of the label.
# --------------------------------------------------------------------------- #

@dataclass
class Shape:
    """A SQL statement split around the single slot that receives a value."""
    head: str          # SQL text before the value slot
    tail: str          # SQL text after the value slot
    quoted: bool       # does real code wrap the slot in single quotes?
    kind: str

    def fstring_body(self, expr: str) -> str:
        """Body of an f-string with `expr` dropped into the slot."""
        slot = SQ + "{" + expr + "}" + SQ if self.quoted else "{" + expr + "}"
        return self.head + slot + self.tail

    def percent_body(self) -> str:
        """Body carrying a `%s` conversion in the slot (string-% idiom)."""
        slot = SQ + "%s" + SQ if self.quoted else "%s"
        return self.head + slot + self.tail

    def brace_body(self) -> str:
        """Body carrying a `{}` field in the slot (str.format idiom)."""
        slot = SQ + "{}" + SQ if self.quoted else "{}"
        return self.head + slot + self.tail

    def dollar_body(self, key: str) -> str:
        """Body carrying a `$key` field (string.Template idiom)."""
        slot = SQ + "$" + key + SQ if self.quoted else "$" + key
        return self.head + slot + self.tail

    @property
    def tail_only(self) -> str:
        """Tail usable by a family that supplies its own SQL head.

        The `select_in` shape carries a leading ")" that closes its `IN (`
        head; a family that rebuilds the head must not inherit it.
        """
        return self.tail[1:] if self.tail.startswith(")") else self.tail

    def with_placeholder(self, ph: str) -> str:
        """Body carrying a driver placeholder -- the *safe* idiom."""
        return self.head + ph + self.tail

    def concat_expr(self, var: str) -> str:
        """A Python concatenation expression that splices `var` into the slot."""
        if self.quoted:
            left = Q + self.head + SQ + Q + " + "
            right = " + " + Q + SQ + self.tail + Q
        else:
            left = Q + self.head + Q + " + "
            right = (" + " + Q + self.tail + Q) if self.tail else ""
        return left + "str(" + var + ")" + right


def _mk_shape(rng: random.Random, ctx: Ctx) -> Shape:
    t, k, cl = ctx.table, ctx.key_col, ctx.col_list
    quoted = ctx.quoted_key
    kind = rng.choice(["select", "select", "select", "count", "update",
                       "delete", "select_in", "select_like"])
    tail = rng.choice(["", "", " LIMIT 100", " AND is_active = 1",
                       " ORDER BY " + rng.choice(ctx.cols) + " DESC",
                       " ORDER BY id ASC LIMIT 50"])
    if kind == "select":
        return Shape("SELECT " + cl + " FROM " + t + " WHERE " + k + " = ", tail, quoted, kind)
    if kind == "count":
        return Shape("SELECT COUNT(*) FROM " + t + " WHERE " + k + " = ", "", quoted, kind)
    if kind == "update":
        col = rng.choice([c for c in ctx.cols if c != k] or ["status"])
        return Shape("UPDATE " + t + " SET " + col + " = " + SQ + "pending" + SQ
                     + " WHERE " + k + " = ", "", quoted, kind)
    if kind == "delete":
        return Shape("DELETE FROM " + t + " WHERE " + k + " = ", "", quoted, kind)
    if kind == "select_in":
        return Shape("SELECT " + cl + " FROM " + t + " WHERE " + k + " IN (",
                     ")" + tail, False, kind)
    return Shape("SELECT " + cl + " FROM " + t + " WHERE " + k + " LIKE ",
                 tail, True, "select_like")


# --------------------------------------------------------------------------- #
# Shared emission helpers
# --------------------------------------------------------------------------- #

def _cursor_open(fn: Fn, ctx: Ctx) -> None:
    """Emit a plausible cursor-acquisition line (never part of the taint path)."""
    r = ctx.rng
    i = ctx.ind
    style = r.random()
    if style < 0.45:
        fn.add(ctx.conn + " = " + ctx.spec["connect"], i)
        fn.add(ctx.cur + " = " + ctx.conn + ".cursor()", i)
    elif style < 0.75:
        fn.add(ctx.cur + " = get_cursor()", i)
    else:
        fn.add(ctx.cur + " = " + ctx.conn + ".cursor()", i)


def _fetch_and_return(fn: Fn, ctx: Ctx, from_cursor: bool = True) -> None:
    r = ctx.rng
    i = ctx.ind
    if not from_cursor:
        fn.add("return " + ctx.res, i)
        return
    style = r.random()
    if style < 0.40:
        fn.add(ctx.res + " = " + ctx.cur + ".fetchall()", i)
        fn.add("return " + ctx.res, i)
    elif style < 0.65:
        fn.add(ctx.res + " = " + ctx.cur + ".fetchone()", i)
        fn.add("return " + ctx.res + " if " + ctx.res + " else None", i)
    elif style < 0.80:
        fn.add(ctx.conn + ".commit()", i)
        fn.add("return " + ctx.cur + ".rowcount", i)
    elif style < 0.92:
        fn.add("return [dict(zip(" + str(ctx.cols) + ", row)) for row in "
               + ctx.cur + ".fetchall()]", i)
    else:
        fn.add("return " + ctx.cur + ".fetchmany(" + str(r.choice([10, 50, 100])) + ")", i)


def _maybe_log(fn: Fn, ctx: Ctx, qvar: Optional[str] = None) -> None:
    """A logging line.  Never the sink, so never part of the rationale."""
    r = ctx.rng
    if r.random() > 0.35:
        return
    i = ctx.ind
    if qvar and r.random() < 0.5:
        fn.add(ctx.logger + ".debug(" + Q + "executing %s" + Q + ", " + qvar + ")", i)
    else:
        fn.add(ctx.logger + ".info(" + Q + "lookup on " + ctx.table + " by %r" + Q
               + ", " + ctx.user_param + ")", i)


def _distractors(fn: Fn, ctx: Ctx, lo: int = 0, hi: int = 3) -> None:
    r = ctx.rng
    for stmt in r.sample(DISTRACTORS, r.randint(lo, hi)):
        fn.add(stmt, ctx.ind)


def _comment(fn: Fn, ctx: Ctx, p: float = 0.45) -> None:
    r = ctx.rng
    if r.random() < p:
        pool = COMMENTS_MISLEADING if r.random() < 0.40 else COMMENTS_NEUTRAL
        fn.add(r.choice(pool), ctx.ind)


def _params_tuple(ctx: Ctx, expr: str) -> str:
    """Render a 1-element parameter sequence in one of the idiomatic forms."""
    r = ctx.rng
    choice = r.random()
    if choice < 0.55:
        return "(" + expr + ",)"
    if choice < 0.85:
        return "[" + expr + "]"
    return "(" + expr + ", )"
