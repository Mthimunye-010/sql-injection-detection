"""The 15 SQL-injection-vulnerable code families.

Each builder emits the *body* of one function into an `Fn` and marks, with
`rationale=True`, exactly the statements that constitute the taint path:

        untrusted source  ->  (optional propagation)  ->  SQL string
                          ->  execute() sink

Those marked statements become the ground-truth attention target used by the
attention-guidance loss.  Because the generator *constructs* the flow, the
annotation is exact -- no manual labelling and no label noise.
"""

from __future__ import annotations

from .sqli_corpus import (
    Q, SQ, Ctx, Fn, Shape, _comment, _cursor_open, _distractors,
    _fetch_and_return, _maybe_log, _params_tuple,
)


# --------------------------------------------------------------------------- #
# V01 -- direct f-string interpolation into the SQL string
# --------------------------------------------------------------------------- #

def v01_fstring_direct(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    if r.random() < 0.55:
        fn.add(ctx.q + " = f" + Q + sh.fstring_body(u) + Q, i, rationale=True)
        _maybe_log(fn, ctx, ctx.q)
        fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    else:
        # The sink and the interpolation collapse onto one statement.
        fn.add(ctx.cur + ".execute(f" + Q + sh.fstring_body(u) + Q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V02 -- old-style %-formatting of the SQL string
# --------------------------------------------------------------------------- #

def v02_percent_operator(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _cursor_open(fn, ctx)
    _distractors(fn, ctx, 0, 2)
    _comment(fn, ctx)
    body = Q + sh.percent_body() + Q
    if r.random() < 0.6:
        fn.add(ctx.q + " = " + body + " % " + u, i, rationale=True)
        _maybe_log(fn, ctx, ctx.q)
        fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    else:
        fn.add(ctx.cur + ".execute(" + body + " % " + u + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V03 -- str.format() into the SQL string
# --------------------------------------------------------------------------- #

def v03_str_format(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    body = Q + sh.brace_body() + Q
    if r.random() < 0.6:
        fn.add(ctx.q + " = " + body + ".format(" + u + ")", i, rationale=True)
        fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    else:
        fn.add(ctx.cur + ".execute(" + body + ".format(" + u + "))", i, rationale=True)
    _maybe_log(fn, ctx)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V04 -- string concatenation
# --------------------------------------------------------------------------- #

def v04_concat_plus(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    if r.random() < 0.5:
        fn.add(ctx.q + " = " + sh.concat_expr(u), i, rationale=True)
    else:
        # Built up with augmented assignment across two statements.
        fn.add(ctx.q + " = " + Q + sh.head + (SQ if sh.quoted else "") + Q, i,
               rationale=True)
        closing = (SQ + sh.tail) if sh.quoted else sh.tail
        fn.add(ctx.q + " += str(" + u + ") + " + Q + closing + Q, i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V05 -- multi-hop taint propagation through intermediate aliases
# --------------------------------------------------------------------------- #

def v05_multi_hop_alias(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _cursor_open(fn, ctx)
    hops = r.randint(2, 4)
    names = r.sample(["raw", "val", "term", "needle", "arg", "candidate",
                      "key", "ident", "token_value"], hops)
    prev = u
    # Each hop is an identity-preserving transformation: the taint survives.
    transforms = [".strip()", "", ".rstrip()", "", ".lstrip()", ""]
    for nm in names:
        fn.add(nm + " = " + prev + r.choice(transforms), i, rationale=True)
        prev = nm
    _distractors(fn, ctx, 0, 2)
    _comment(fn, ctx)
    fn.add(ctx.q + " = f" + Q + sh.fstring_body(prev) + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V06 -- query assembled by a nested helper (inter-procedural flow)
# --------------------------------------------------------------------------- #

def v06_helper_builder(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    helper = r.choice(["_build_sql", "_compose", "_make_query", "_render_stmt",
                       "_assemble", "_to_sql"])
    arg = r.choice(["value", "v", "needle", "arg"])
    fn.add("def " + helper + "(" + arg + "):", i)
    if r.random() < 0.4:
        fn.add(Q * 3 + "Return the lookup statement for the given value." + Q * 3, i + 1)
    fn.add("return f" + Q + sh.fstring_body(arg) + Q, i + 1, rationale=True)
    fn.blank()
    _cursor_open(fn, ctx)
    _distractors(fn, ctx, 0, 2)
    fn.add(ctx.q + " = " + helper + "(" + u + ")", i, rationale=True)
    _maybe_log(fn, ctx, ctx.q)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V07 -- taint read out of a request payload / mapping
# --------------------------------------------------------------------------- #

def v07_dict_request_field(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i = ctx.rng, ctx.ind
    src = r.choice([
        ("payload", "request.get_json(silent=True) or {}"),
        ("body", "json.loads(raw_body)"),
        ("params", "dict(request.args)"),
        ("form", "request.form.to_dict()"),
        ("event", "json.loads(message.value)"),
    ])
    holder, rhs = src
    field = r.choice(["q", "name", "email", "term", "filter", "id", "ref"])
    local = r.choice(["value", "needle", "wanted", "chosen"])
    fn.add(holder + " = " + rhs, i)
    getter = r.choice([
        holder + "[" + Q + field + Q + "]",
        holder + ".get(" + Q + field + Q + ", " + Q + Q + ")",
    ])
    fn.add(local + " = " + getter, i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    fn.add(ctx.cur + ".execute(f" + Q + sh.fstring_body(local) + Q + ")", i,
           rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V08 -- WHERE clause assembled inside a loop over caller-supplied filters
# --------------------------------------------------------------------------- #

def v08_loop_built_where(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i = ctx.rng, ctx.ind
    ctx.sig_extra.append("filters")
    acc = r.choice(["clauses", "parts", "conds", "predicates"])
    fn.add(acc + " = []", i)
    fn.add("for col, val in filters.items():", i)
    fn.add(acc + ".append(f" + Q + "{col} = " + SQ + "{val}" + SQ + Q + ")",
           i + 1, rationale=True)
    joiner = r.choice([Q + " AND " + Q, Q + " OR " + Q])
    where = r.choice(["where", "where_sql", "predicate"])
    fn.add(where + " = " + joiner + ".join(" + acc + ") or " + Q + "1=1" + Q, i,
           rationale=True)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    sql_txt = ("SELECT " + ctx.col_list + " FROM " + ctx.table + " WHERE {"
               + where + "}")
    fn.add(ctx.q + " = f" + Q + sql_txt + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V09 -- raw SQL escape hatch of an ORM
# --------------------------------------------------------------------------- #

def v09_orm_raw_text(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    model = ctx.noun.capitalize()
    variant = r.randint(0, 3)
    _distractors(fn, ctx, 0, 2)
    _comment(fn, ctx)
    if variant == 0:
        fn.add("return " + model + ".objects.raw(f" + Q + sh.fstring_body(u) + Q + ")",
               i, rationale=True)
    elif variant == 1:
        fn.add(ctx.q + " = text(f" + Q + sh.fstring_body(u) + Q + ")", i, rationale=True)
        fn.add(ctx.res + " = session.execute(" + ctx.q + ")", i, rationale=True)
        fn.add("return " + ctx.res + ".fetchall()", i)
    elif variant == 2:
        fn.add("with connection.cursor() as " + ctx.cur + ":", i)
        fn.add(ctx.cur + ".execute(f" + Q + sh.fstring_body(u) + Q + ")", i + 1,
               rationale=True)
        fn.add("return " + ctx.cur + ".fetchall()", i + 1)
    else:
        fn.add("return list(" + model + ".objects.extra(where=[f"
               + Q + "{" + ctx.key_col + "_col} = " + SQ + "{" + u + "}" + SQ + Q
               + "]))", i, rationale=True)
        ctx.sig_extra.append(ctx.key_col + "_col")


# --------------------------------------------------------------------------- #
# V10 -- multi-statement execution sink
# --------------------------------------------------------------------------- #

def v10_executescript(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _cursor_open(fn, ctx)
    script = r.choice(["script", "batch", "ddl", "sql_batch"])
    stmt = sh.percent_body() + ";"
    fn.add(script + " = " + Q + stmt + Q + " % " + u, i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    _comment(fn, ctx)
    sink = r.choice([".executescript(", ".executescript(", ".execute("])
    fn.add(ctx.cur + sink + script + ")", i, rationale=True)
    fn.add(ctx.conn + ".commit()", i)
    fn.add("return " + ctx.cur + ".rowcount", i)


# --------------------------------------------------------------------------- #
# V11 -- string.Template / %-mapping substitution
# --------------------------------------------------------------------------- #

def v11_template_substitute(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _cursor_open(fn, ctx)
    key = r.choice(["val", "needle", "target", "v"])
    if r.random() < 0.55:
        tpl = r.choice(["tpl", "template", "sql_tpl"])
        fn.add(tpl + " = string.Template(" + Q + sh.dollar_body(key) + Q + ")", i)
        fn.add(ctx.q + " = " + tpl + ".substitute(" + key + "=" + u + ")", i,
               rationale=True)
    else:
        body = sh.head + ((SQ + "%(" + key + ")s" + SQ) if sh.quoted
                          else ("%(" + key + ")s")) + sh.tail
        fn.add(ctx.q + " = " + Q + body + Q + " % {" + Q + key + Q + ": " + u + "}",
               i, rationale=True)
    _comment(fn, ctx)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V12 -- values are parameterised but an *identifier* is interpolated
# --------------------------------------------------------------------------- #

def v12_order_by_identifier(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i = ctx.rng, ctx.ind
    col_param = r.choice(["sort_column", "order_by", "sort_field", "group_col"])
    ctx.sig_extra.append(col_param)
    key_arg = r.choice(["tenant_id", "owner_id", "account_id"])
    ctx.sig_extra.append(key_arg)
    _cursor_open(fn, ctx)
    direction = r.choice(["ASC", "DESC"])
    sql_txt = ("SELECT " + ctx.col_list + " FROM " + ctx.table + " WHERE "
               + ctx.key_col + " = " + ctx.ph + " ORDER BY {" + col_param + "} "
               + direction)
    _comment(fn, ctx, p=0.6)
    fn.add(ctx.q + " = f" + Q + sql_txt + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ", " + _params_tuple(ctx, key_arg) + ")",
           i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V13 -- SQL assembled from a list of fragments
# --------------------------------------------------------------------------- #

def v13_join_fragments(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    parts = r.choice(["parts", "frags", "pieces", "chunks", "tokens"])
    fn.add(parts + " = [", i)
    fn.add(Q + "SELECT " + ctx.col_list + " FROM " + ctx.table + Q + ",", i + 1)
    fn.add(Q + "WHERE " + ctx.key_col + " =" + Q + ",", i + 1)
    quote = SQ if ctx.quoted_key else ""
    fn.add(Q + quote + Q + " + str(" + u + ") + " + Q + quote + Q + ",", i + 1,
           rationale=True)
    if r.random() < 0.5:
        fn.add(Q + "LIMIT 200" + Q + ",", i + 1)
    fn.add("]", i)
    _distractors(fn, ctx, 0, 2)
    fn.add(ctx.q + " = " + Q + " " + Q + ".join(" + parts + ")", i, rationale=True)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V14 -- partial parameterisation: one bound value, one interpolated value
# --------------------------------------------------------------------------- #

def v14_partial_parameterisation(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    bound = r.choice(["tenant_id", "org_id", "limit_rows", "owner_id"])
    ctx.sig_extra.append(bound)
    _cursor_open(fn, ctx)
    _comment(fn, ctx, p=0.6)
    sql_txt = ("SELECT " + ctx.col_list + " FROM " + ctx.table + " WHERE "
               + r.choice(ctx.cols) + " = " + ctx.ph + " AND " + ctx.key_col
               + " = " + SQ + "{" + u + "}" + SQ)
    fn.add(ctx.q + " = f" + Q + sql_txt + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ", " + _params_tuple(ctx, bound) + ")",
           i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# V15 -- an *ineffective* sanitiser is applied before interpolation
# --------------------------------------------------------------------------- #

def v15_useless_sanitiser(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    cleaned = r.choice(["cleaned", "safe_value", "sanitised", "escaped", "scrubbed"])
    # None of these neutralise SQL metacharacters in a quoted-string context.
    sanitiser = r.choice([
        u + ".strip()",
        u + ".replace(" + Q + ";" + Q + ", " + Q + Q + ")",
        u + ".lower()",
        "html.escape(" + u + ")",
        u + ".replace(" + Q + "--" + Q + ", " + Q + Q + ")",
        "re.sub(r" + Q + "\\s+" + Q + ", " + Q + " " + Q + ", " + u + ")",
        "urllib.parse.quote(" + u + ")",
        u + "[:64]",
    ])
    fn.add(cleaned + " = " + sanitiser, i, rationale=True)
    if r.random() < 0.55:
        fn.add("# value has been escaped, safe to inline", i)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    fn.add(ctx.q + " = f" + Q + sh.fstring_body(cleaned) + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


VULN_BUILDERS = {
    "V01_fstring_direct": v01_fstring_direct,
    "V02_percent_operator": v02_percent_operator,
    "V03_str_format": v03_str_format,
    "V04_concat_plus": v04_concat_plus,
    "V05_multi_hop_alias": v05_multi_hop_alias,
    "V06_helper_builder": v06_helper_builder,
    "V07_dict_request_field": v07_dict_request_field,
    "V08_loop_built_where": v08_loop_built_where,
    "V09_orm_raw_text": v09_orm_raw_text,
    "V10_executescript": v10_executescript,
    "V11_template_substitute": v11_template_substitute,
    "V12_order_by_identifier": v12_order_by_identifier,
    "V13_join_fragments": v13_join_fragments,
    "V14_partial_parameterisation": v14_partial_parameterisation,
    "V15_useless_sanitiser": v15_useless_sanitiser,
}
