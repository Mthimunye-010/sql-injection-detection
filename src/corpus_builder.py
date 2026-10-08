"""Assembles family bodies into complete, syntactically valid Python functions
and drives bulk corpus generation.

The assembler adds the realism layer that keeps the task non-trivial:
decorators, type hints, docstrings, `try/except`, `with` blocks, class methods,
logging and distractor statements.  All of this is sampled *independently of the
label*, so surface structure carries no information about vulnerability.
"""

from __future__ import annotations

import ast
import hashlib
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .families_safe import SAFE_BUILDERS
from .families_vuln import VULN_BUILDERS
from .sqli_corpus import (
    ALL_FAMILIES, DIFFICULTY, HARD_NEGATIVE_FAMILIES, Q, SAFE_FAMILIES,
    VULN_FAMILIES, Ctx, Fn, Sample, _mk_ctx, _mk_shape,
)
from .vocabpools import DECORATORS, DOCSTRINGS

BUILDERS = dict(VULN_BUILDERS)
BUILDERS.update(SAFE_BUILDERS)

# Parameters that should carry a type hint when hints are switched on.
HINTS = {
    "filters": "dict", "records": "list", "limit": "int", "offset": "int",
    "page": "int", "fields": "list", "include_archived": "bool",
    "tenant": "str", "tenant_id": "int", "owner_id": "int", "org_id": "int",
    "account_id": "int", "limit_rows": "int",
}
DEFAULTED = {"limit": "50", "offset": "0", "page": "1", "fields": "None",
             "include_archived": "False", "tenant": "'public'"}

RETURNS = ["", "", " -> list", " -> dict", " -> Any", " -> Optional[dict]",
           " -> Sequence[tuple]"]


# --------------------------------------------------------------------------- #
# Signature construction
# --------------------------------------------------------------------------- #

def _signature(ctx: Ctx, is_method: bool, use_hints: bool) -> str:
    r = ctx.rng
    seen = set()
    plain: List[str] = []
    for name in [ctx.user_param] + list(ctx.sig_extra) + list(ctx.extra_params):
        if name in seen:
            continue
        seen.add(name)
        plain.append(name)

    parts: List[str] = ["self"] if is_method else []
    tail: List[str] = []
    for name in plain:
        hint = HINTS.get(name, "str")
        txt = name + ((": " + hint) if use_hints else "")
        if name in DEFAULTED:
            tail.append(txt + " = " + DEFAULTED[name])
        else:
            parts.append(txt)
    return ", ".join(parts + tail)


def _header(fn: Fn, ctx: Ctx, is_method: bool, base: int) -> None:
    """Emit class line (if any), decorators, def line and docstring."""
    r = ctx.rng
    def_ind = base - 1

    if is_method:
        cls = ctx.noun.capitalize() + r.choice(["Repository", "Store", "Dao",
                                                "Gateway", "Service"])
        fn.add("class " + cls + ":", 0)

    n_dec = 0 if r.random() < 0.62 else r.randint(1, 2)
    for dec in r.sample(DECORATORS, n_dec):
        fn.add(dec.replace("{noun}", ctx.noun), def_ind)

    use_hints = r.random() < 0.45
    sig = _signature(ctx, is_method, use_hints)
    ret = r.choice(RETURNS) if use_hints else ""
    fn.add("def " + ctx.fn_name + "(" + sig + "):" if not ret
           else "def " + ctx.fn_name + "(" + sig + ")" + ret + ":", def_ind)

    p = r.random()
    if p < 0.40:
        doc = r.choice(DOCSTRINGS).replace("{noun}", ctx.noun)
        fn.add(Q * 3 + doc + Q * 3, base)
    elif p < 0.58:
        doc = r.choice(DOCSTRINGS).replace("{noun}", ctx.noun)
        fn.add(Q * 3 + doc, base)
        fn.add("", 0)
        fn.add("Args:", base)
        fn.add(ctx.user_param + ": value supplied by the caller.", base + 1)
        fn.add(Q * 3, base)


# --------------------------------------------------------------------------- #
# One sample
# --------------------------------------------------------------------------- #

WRAPS = ["plain", "plain", "plain", "try", "with", "method", "method_try"]


def make_sample(rng: random.Random, family: str,
                api: Optional[str] = None) -> Optional[Sample]:
    ctx = _mk_ctx(rng, api)
    shape = _mk_shape(rng, ctx)
    wrap = rng.choice(WRAPS)
    is_method = wrap.startswith("method")
    use_try = wrap.endswith("try")
    use_with = wrap == "with"

    base = 2 if is_method else 1              # indent of the function body
    ctx.ind = base + (1 if (use_try or use_with) else 0)

    body = Fn()
    BUILDERS[family](body, ctx, shape)

    head = Fn()
    _header(head, ctx, is_method, base)

    mid = Fn()
    if use_try:
        mid.add("try:", base)
    elif use_with:
        mid.add("with contextlib.closing(" + ctx.spec["connect"] + ") as "
                + ctx.conn + ":", base)

    foot = Fn()
    if use_try:
        foot.add("except " + ctx.spec["error"] + " as exc:", base)
        if rng.random() < 0.5:
            foot.add(ctx.logger + ".exception(" + Q + "query failed: %s" + Q
                     + ", exc)", base + 1)
            foot.add("raise", base + 1)
        else:
            foot.add("raise RuntimeError(" + Q + "database lookup failed" + Q
                     + ") from exc", base + 1)

    merged = Fn()
    merged.rec = head.rec + mid.rec + body.rec + foot.rec
    offset = len(head.rec) + len(mid.rec)
    merged.rec = head.rec + mid.rec + body.rec + foot.rec

    code, rationale = merged.render()
    # Rationale indices already absolute because render() walks the merged list.
    try:
        ast.parse(code)
    except SyntaxError:
        return None

    return Sample(
        code=code,
        label=1 if family in VULN_FAMILIES else 0,
        family=family,
        api=ctx.api,
        rationale_lines=rationale,
        difficulty=DIFFICULTY[family],
        cwe="CWE-89" if family in VULN_FAMILIES else "none",
        n_lines=code.count("\n") + 1,
        hard_negative=int(family in HARD_NEGATIVE_FAMILIES),
        uid=hashlib.sha1(code.encode("utf-8")).hexdigest()[:16],
    )


# --------------------------------------------------------------------------- #
# Bulk generation
# --------------------------------------------------------------------------- #

def generate_corpus(n_per_family: Dict[str, int], seed: int = 20260404,
                    max_attempts_factor: int = 12,
                    progress: bool = False) -> List[Sample]:
    """Generate `n_per_family[f]` *unique* samples for every family `f`.

    Uniqueness is enforced on the SHA-1 of the rendered source, so the corpus
    contains no duplicated functions -- an important guard against train/test
    leakage inflating the reported scores.
    """
    rng = random.Random(seed)
    out: List[Sample] = []
    seen: set = set()
    n_syntax_fail = 0

    items = sorted(n_per_family.items())
    for fi, (fam, want) in enumerate(items):
        got = 0
        attempts = 0
        cap = want * max_attempts_factor + 500
        while got < want and attempts < cap:
            attempts += 1
            s = make_sample(rng, fam)
            if s is None:
                n_syntax_fail += 1
                continue
            if s.uid in seen:
                continue
            seen.add(s.uid)
            out.append(s)
            got += 1
        if progress:
            print("  %-32s %6d/%-6d (attempts %d)" % (fam, got, want, attempts))
        if got < want:
            raise RuntimeError(
                "family %s exhausted its template space: %d/%d" % (fam, got, want))

    if n_syntax_fail:
        raise RuntimeError("%d generated samples failed ast.parse" % n_syntax_fail)
    rng.shuffle(out)
    return out


def balanced_plan(n_total: int, families: Sequence[str]) -> Dict[str, int]:
    """Spread `n_total` samples as evenly as possible over `families`."""
    k = len(families)
    base, rem = divmod(n_total, k)
    plan = {f: base for f in families}
    for f in sorted(families)[:rem]:
        plan[f] += 1
    return plan
