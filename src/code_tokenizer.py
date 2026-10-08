"""A SQL-injection-aware lexer and vocabulary for Python source code.

Three design decisions carry most of the model's inductive bias.

1. **Hand-rolled lexer instead of `tokenize`.**
   CPython 3.12 changed f-string tokenisation (PEP 701), so the standard
   library would give different token streams on different interpreters.  The
   scanner below is version-independent, which matters because the notebook has
   to reproduce on the marker's machine.

2. **Per-sample identifier indexing.**
   Every identifier that is not part of a fixed API lexicon is rewritten to
   `<ID_k>`, where `k` is the order of first appearance *within that sample*.
   This destroys name memorisation (`user_input` is no more suspicious than
   `tenant`) while *preserving aliasing*: if the same variable is written on
   line 4 and read on line 7, both become `<ID_3>`, so the network can still
   follow the data flow.  That is exactly the signal a taint analysis needs.

3. **String-literal decomposition.**
   A SQL string is not an opaque blob.  Each literal is expanded into
   structural markers so that the model can tell apart the three cases that
   decide the label:

       f"... WHERE x = '{v}'"   ->  <SQ> <FFIELD_BEG> <ID_k> <FFIELD_END> <SQ>
       "... WHERE x = ?"        ->  <QMARK>
       "... WHERE x = %s"       ->  <PCT_FMT>

   Without this, "interpolated into a quoted SQL slot" and "bound as a driver
   parameter" look nearly identical, and no amount of model capacity recovers
   the difference.

The lexer also returns a line index per token, which is what lets us project
statement-level taint annotations onto the token sequence to build the
attention-guidance target.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Fixed lexicons
# --------------------------------------------------------------------------- #

PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
}

# Security-relevant API surface.  Kept verbatim because these names *are* the
# semantics: `execute` is a sink, `int` is a sanitiser, `bindparams` is a
# mitigation.  Deliberately excludes every identifier the generator invents for
# constants (e.g. allow-list names), so that no generator artefact leaks the
# label.
API_LEXICON = {
    # sinks / cursors / connections
    "execute", "executemany", "executescript", "executescript", "cursor",
    "connect", "commit", "rollback", "fetchone", "fetchall", "fetchmany",
    "rowcount", "callproc", "connection", "close", "closing", "contextlib",
    # ORM / query builders
    "objects", "raw", "extra", "session", "query", "filter", "select",
    "scalars", "mappings", "text", "bindparams", "first", "all", "limit",
    "order_by", "values", "where",
    # composable SQL
    "sql", "SQL", "Identifier", "Composed", "Literal",
    # string building
    "format", "format_map", "join", "substitute", "safe_substitute", "Template",
    "string", "replace", "strip", "lstrip", "rstrip", "lower", "upper", "title",
    "split", "rsplit", "startswith", "endswith", "encode", "decode", "removeprefix",
    # sanitisers / validators
    "int", "float", "str", "bool", "index", "operator", "escape", "html",
    "quote", "quote_plus", "urllib", "parse", "re", "sub", "fullmatch", "match",
    "search", "compile", "isalnum", "isdigit", "isnumeric", "isidentifier",
    "isalpha", "assert", "ValueError", "TypeError", "RuntimeError", "Exception",
    "len", "min", "max", "abs", "round", "zip", "map", "list", "dict", "set",
    "tuple", "sorted", "enumerate", "any", "all",
    # request / taint sources
    "request", "get_json", "args", "form", "to_dict", "json", "loads", "dumps",
    "body", "payload", "environ", "headers", "cookies", "GET", "POST", "input",
    "argv", "getenv", "environ",
    # mappings
    "get", "items", "keys", "pop", "setdefault", "append", "extend", "update",
    # misc plumbing that appears in both classes
    "self", "logger", "log", "LOG", "info", "debug", "warning", "error",
    "exception", "critical", "os", "path", "basename", "dirname", "exists",
    "time", "datetime", "utcnow", "perf_counter", "functools", "lru_cache",
    "cache", "metrics", "increment", "app", "route", "router", "staticmethod",
    "classmethod", "property",
}

SQL_KEYWORDS = {
    "select", "from", "where", "insert", "into", "values", "update", "set",
    "delete", "and", "or", "order", "by", "group", "having", "limit", "offset",
    "join", "inner", "outer", "left", "right", "on", "union", "all", "distinct",
    "count", "sum", "avg", "min", "max", "like", "ilike", "in", "between",
    "is", "null", "not", "as", "asc", "desc", "exec", "execute", "call",
    "drop", "create", "alter", "table", "truncate", "grant", "declare",
    "case", "when", "then", "end", "exists", "top", "with", "returning",
}

# Special tokens
PAD, UNK = "<PAD>", "<UNK>"
STR_BEG, STR_END = "<STR_BEG>", "<STR_END>"
FSTR, RAWSTR, BYTESTR = "<FSTR>", "<RAWSTR>", "<BYTESTR>"
FFIELD_BEG, FFIELD_END = "<FFIELD_BEG>", "<FFIELD_END>"
BRACE_FIELD = "<BRACE_FIELD>"
PCT_FMT, PCT_NAMED = "<PCT_FMT>", "<PCT_NAMED>"
QMARK, COLON_PARAM, DOLLAR_PARAM = "<QMARK>", "<COLON_PARAM>", "<DOLLAR_PARAM>"
SQUOTE, DQUOTE = "<SQ>", "<DQ>"
STR_WORD, STR_NUM = "<SW>", "<SNUM>"
NUM = "<NUM>"
CMT, CMT_WORD = "<CMT>", "<CMT_W>"
ID_TPL = "<ID_%d>"
ID_OOV = "<ID_OOV>"
MAX_IDS = 24

SPECIALS = [
    PAD, UNK, STR_BEG, STR_END, FSTR, RAWSTR, BYTESTR, FFIELD_BEG, FFIELD_END,
    BRACE_FIELD, PCT_FMT, PCT_NAMED, QMARK, COLON_PARAM, DOLLAR_PARAM,
    SQUOTE, DQUOTE, STR_WORD, STR_NUM, NUM, CMT, CMT_WORD, ID_OOV,
] + [ID_TPL % k for k in range(MAX_IDS)]

OPERATORS = [
    "**=", "//=", ">>=", "<<=", "...", "->", ":=", "==", "!=", "<=", ">=",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ">>", "<<", "**", "//",
    "+", "-", "*", "/", "%", "@", "&", "|", "^", "~", "<", ">", "=",
    "(", ")", "{", "}", "[", "]", ",", ":", ".", ";",
]
_OPS_SORTED = sorted(OPERATORS, key=len, reverse=True)

STR_PREFIX_RE = re.compile(r"[rRbBuUfF]{0,3}$")


# --------------------------------------------------------------------------- #
# Stage 1: raw scanning
# --------------------------------------------------------------------------- #

def scan(code: str) -> List[Tuple[str, str, int]]:
    """Scan `code` into (kind, text, line) triples.

    kind in {NAME, NUM, STR, OP, CMT, NL}.  `line` is 0-based.
    """
    out: List[Tuple[str, str, int]] = []
    i, n, line = 0, len(code), 0
    while i < n:
        ch = code[i]
        if ch == "\n":
            out.append(("NL", "\n", line))
            line += 1
            i += 1
            continue
        if ch in " \t\r":
            i += 1
            continue
        if ch == "#":
            j = code.find("\n", i)
            j = n if j < 0 else j
            out.append(("CMT", code[i:j], line))
            i = j
            continue
        # string literal (with optional prefix immediately before the quote)
        if ch in "\"'" or (ch.isalpha() and _string_ahead(code, i)):
            k = i
            while code[k] not in "\"'":
                k += 1
            prefix = code[i:k]
            q = code[k]
            triple = code[k:k + 3] in ('"""', "'''")
            delim = code[k:k + 3] if triple else q
            raw = "r" in prefix.lower()
            j = _find_close(code, k + len(delim), delim, raw)
            text = code[i:j]
            out.append(("STR", text, line))
            line += text.count("\n")
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and code[i + 1].isdigit()):
            m = re.match(r"0[xXoObB][0-9a-fA-F_]+|\d[\d_]*\.?[\d_]*(?:[eE][-+]?\d+)?[jJ]?",
                         code[i:])
            txt = m.group(0) if m else ch
            out.append(("NUM", txt, line))
            i += len(txt)
            continue
        if ch.isalpha() or ch == "_":
            m = re.match(r"[A-Za-z_]\w*", code[i:])
            txt = m.group(0)
            out.append(("NAME", txt, line))
            i += len(txt)
            continue
        for op in _OPS_SORTED:
            if code.startswith(op, i):
                out.append(("OP", op, line))
                i += len(op)
                break
        else:
            i += 1   # unknown byte: skip
    return out


def _string_ahead(code: str, i: int) -> bool:
    """True if a string-prefix run starting at `i` is followed by a quote."""
    k = i
    while k < len(code) and k - i < 3 and code[k] in "rRbBuUfF":
        k += 1
    return k > i and k < len(code) and code[k] in "\"'"


def _find_close(code: str, start: int, delim: str, raw: bool) -> int:
    """Index just past the closing delimiter of a string that opened at `start`."""
    i, n = start, len(code)
    while i < n:
        if not raw and code[i] == "\\":
            i += 2
            continue
        if code.startswith(delim, i):
            return i + len(delim)
        if len(delim) == 1 and code[i] == "\n":
            return i          # unterminated single-quoted string: bail out
        i += 1
    return n


# --------------------------------------------------------------------------- #
# Stage 2: normalisation into model tokens
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z_]\w*")
_SQL_SPLIT_RE = re.compile(r"[A-Za-z_]\w*|\d+|.", re.S)


class IdMap:
    """Per-sample identifier -> `<ID_k>` map, ordered by first appearance."""

    def __init__(self) -> None:
        self._m: Dict[str, str] = {}

    def __call__(self, name: str) -> str:
        tok = self._m.get(name)
        if tok is None:
            k = len(self._m)
            tok = ID_TPL % k if k < MAX_IDS else ID_OOV
            self._m[name] = tok
        return tok


def _string_body(text: str) -> Tuple[str, str]:
    """Split a raw string literal into (prefix, inner content)."""
    k = 0
    while text[k] not in "\"'":
        k += 1
    prefix = text[:k].lower()
    delim = text[k:k + 3] if text[k:k + 3] in ('"""', "'''") else text[k]
    inner = text[k + len(delim):]
    if inner.endswith(delim):
        inner = inner[: -len(delim)]
    return prefix, inner


def _tokenise_string(text: str, line: int, idmap: IdMap,
                     out: List[Tuple[str, int]]) -> None:
    """Expand one string literal into structural tokens (see module docstring)."""
    prefix, inner = _string_body(text)
    out.append((STR_BEG, line))
    if "f" in prefix:
        out.append((FSTR, line))
    if "r" in prefix:
        out.append((RAWSTR, line))
    if "b" in prefix:
        out.append((BYTESTR, line))

    is_f = "f" in prefix
    i, n = 0, len(inner)
    while i < n:
        c = inner[i]

        # ---- replacement field  {expr}  /  {}  /  {0}  /  {name}
        if c == "{":
            if inner.startswith("{{", i):
                out.append((STR_WORD, line))
                i += 2
                continue
            j = inner.find("}", i)
            if j < 0:
                out.append((STR_WORD, line))
                i += 1
                continue
            field = inner[i + 1:j]
            # Strip format spec / conversion:  {v!r:>10}
            expr = re.split(r"![sra]|:(?![=:])", field, maxsplit=1)[0].strip()
            if is_f and expr:
                out.append((FFIELD_BEG, line))
                for kind, txt, _ln in scan(expr):
                    _emit_code_token(kind, txt, line, idmap, out)
                out.append((FFIELD_END, line))
            else:
                # str.format placeholder: the value is not visible here.
                out.append((BRACE_FIELD, line))
            i = j + 1
            continue

        # ---- %-conversion
        if c == "%":
            m = re.match(r"%\((\w+)\)[-#0-9. +]*[sdifrxX%]", inner[i:])
            if m:
                out.append((PCT_NAMED, line))
                i += len(m.group(0))
                continue
            m = re.match(r"%[-#0-9. +]*[sdifrxX]", inner[i:])
            if m:
                out.append((PCT_FMT, line))
                i += len(m.group(0))
                continue
            out.append((STR_WORD, line))
            i += 1
            continue

        # ---- driver placeholders
        if c == "?":
            out.append((QMARK, line))
            i += 1
            continue
        m = re.match(r":(\w+)", inner[i:])
        if m:
            out.append((COLON_PARAM, line))
            i += len(m.group(0))
            continue
        m = re.match(r"\$\{?(\w+)\}?", inner[i:])
        if m:
            out.append((DOLLAR_PARAM, line))
            i += len(m.group(0))
            continue

        # ---- quote characters inside the literal mark a SQL string context
        if c == "'":
            out.append((SQUOTE, line))
            i += 1
            continue
        if c == '"':
            out.append((DQUOTE, line))
            i += 1
            continue

        # ---- words: keep SQL keywords, collapse the rest
        m = _WORD_RE.match(inner, i)
        if m:
            w = m.group(0)
            lw = w.lower()
            out.append(("sql:" + lw if lw in SQL_KEYWORDS else STR_WORD, line))
            i = m.end()
            continue
        if c.isdigit():
            m = re.match(r"\d+", inner[i:])
            out.append((STR_NUM, line))
            i += len(m.group(0))
            continue
        if c in "()=,*<>.;|/+-":
            out.append(("s" + c, line))
            i += 1
            continue
        if c in " \t\n":
            i += 1
            continue
        out.append((STR_WORD, line))
        i += 1

    out.append((STR_END, line))


def _emit_code_token(kind: str, txt: str, line: int, idmap: IdMap,
                     out: List[Tuple[str, int]]) -> None:
    if kind == "NAME":
        if txt in PY_KEYWORDS or txt in API_LEXICON:
            out.append((txt, line))
        else:
            out.append((idmap(txt), line))
    elif kind == "NUM":
        out.append((NUM, line))
    elif kind == "OP":
        out.append((txt, line))
    elif kind == "STR":
        _tokenise_string(txt, line, idmap, out)
    elif kind == "CMT":
        out.append((CMT, line))
        for w in _WORD_RE.findall(txt.lower())[:6]:
            out.append(("c:" + w, line))
    # NL is handled by the caller


def tokenise(code: str, keep_newlines: bool = True) -> Tuple[List[str], List[int]]:
    """Return (tokens, line_of_token) for one source snippet."""
    idmap = IdMap()
    out: List[Tuple[str, int]] = []
    for kind, txt, line in scan(code):
        if kind == "NL":
            if keep_newlines:
                out.append(("<NL>", line))
            continue
        _emit_code_token(kind, txt, line, idmap, out)
    toks = [t for t, _ in out]
    lines = [l for _, l in out]
    return toks, lines


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

class Vocab:
    """Frequency-pruned token vocabulary built on the *training* split only."""

    def __init__(self, itos: Sequence[str]) -> None:
        self.itos: List[str] = list(itos)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks: Sequence[str]) -> List[int]:
        s = self.stoi
        u = self.unk_id
        return [s.get(t, u) for t in toks]

    @classmethod
    def build(cls, token_lists: Sequence[Sequence[str]], min_freq: int = 5,
              max_size: int = 20000) -> "Vocab":
        cnt: Counter = Counter()
        for toks in token_lists:
            cnt.update(toks)
        base = list(SPECIALS) + ["<NL>"]
        base += sorted(PY_KEYWORDS) + sorted(API_LEXICON)
        base += ["sql:" + k for k in sorted(SQL_KEYWORDS)]
        base += list(OPERATORS)
        base += ["s" + c for c in "()=,*<>.;|/+-"]
        seen = set()
        itos: List[str] = []
        for t in base:
            if t not in seen:
                seen.add(t)
                itos.append(t)
        for tok, c in cnt.most_common():
            if len(itos) >= max_size:
                break
            if tok in seen or c < min_freq:
                continue
            seen.add(tok)
            itos.append(tok)
        return cls(itos)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        return cls(d["itos"])


# --------------------------------------------------------------------------- #
# Encoding a dataset
# --------------------------------------------------------------------------- #

def rationale_mask(lines: Sequence[int], rationale_lines: Sequence[int]) -> List[int]:
    """Project statement-level annotations onto the token sequence."""
    rset = set(rationale_lines)
    return [1 if l in rset else 0 for l in lines]


def parse_rationale(field) -> List[int]:
    """Parse the comma-separated `rationale_lines` column."""
    if field is None:
        return []
    s = str(field).strip()
    if not s or s == "nan":
        return []
    return [int(x) for x in s.split(",") if x != ""]
