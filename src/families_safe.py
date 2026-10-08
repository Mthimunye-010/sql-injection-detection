"""The 15 non-vulnerable code families.

Eight of them are deliberate **hard negatives**: code that still drops a value
into a SQL string (or calls `.format()` on something SQL-shaped) and is
nevertheless safe, because the value is an integer cast, an allow-listed
identifier, a regex-validated token, a module constant, or because the sink is
not a database at all.  These are exactly the patterns that drive the
false-positive rate of pattern-matching static analysers, and they are what the
proposed model has to learn to clear.

For safe samples the `rationale=True` statements mark the **safety evidence**
-- the placeholder/binding statement, the validation guard, or the cast -- so
that attention guidance supplies a signal for *both* classes rather than only
for the positive class.
"""

from __future__ import annotations

from .sqli_corpus import (
    Q, SQ, Ctx, Fn, Shape, _comment, _cursor_open, _distractors,
    _fetch_and_return, _maybe_log, _params_tuple,
)


# --------------------------------------------------------------------------- #
# S01 -- question-mark placeholders with a parameter sequence
# --------------------------------------------------------------------------- #

def s01_qmark_params(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    ph = "?"
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    arg = u
    if sh.kind == "select_like":
        arg = "f" + Q + "%{" + u + "}%" + Q
    fn.add(ctx.q + " = " + Q + sh.with_placeholder(ph) + Q, i, rationale=True)
    _maybe_log(fn, ctx, ctx.q)
    fn.add(ctx.cur + ".execute(" + ctx.q + ", " + _params_tuple(ctx, arg) + ")",
           i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S02 -- %s / pyformat placeholders with a parameter sequence
# --------------------------------------------------------------------------- #

def s02_pyformat_params(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    _cursor_open(fn, ctx)
    _distractors(fn, ctx, 0, 2)
    _comment(fn, ctx)
    arg = u
    if sh.kind == "select_like":
        arg = Q + "%" + Q + " + " + u + " + " + Q + "%" + Q
    body = Q + sh.with_placeholder("%s") + Q
    if r.random() < 0.5:
        fn.add(ctx.cur + ".execute(" + body + ", " + _params_tuple(ctx, arg) + ")",
               i, rationale=True)
    else:
        fn.add(ctx.q + " = " + body, i, rationale=True)
        fn.add(ctx.cur + ".execute(" + ctx.q + ", " + _params_tuple(ctx, arg) + ")",
               i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S03 -- named placeholders bound from a mapping
# --------------------------------------------------------------------------- #

def s03_named_params(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    key = r.choice(["val", "needle", "target", ctx.key_col])
    colon_style = r.random() < 0.5
    ph = (":" + key) if colon_style else ("%(" + key + ")s")
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    binds = r.choice(["binds", "params", "args", "values"])
    fn.add(binds + " = {" + Q + key + Q + ": " + u + "}", i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    fn.add(ctx.q + " = " + Q + sh.with_placeholder(ph) + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ", " + binds + ")", i, rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S04 -- SQLAlchemy textual SQL with bound parameters
# --------------------------------------------------------------------------- #

def s04_sqlalchemy_bindparams(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    key = r.choice(["val", "needle", ctx.key_col])
    stmt = r.choice(["stmt", "statement", "sel"])
    _distractors(fn, ctx, 0, 2)
    _comment(fn, ctx)
    if r.random() < 0.5:
        fn.add(stmt + " = text(" + Q + sh.with_placeholder(":" + key) + Q
               + ").bindparams(" + key + "=" + u + ")", i, rationale=True)
        fn.add(ctx.res + " = session.execute(" + stmt + ")", i, rationale=True)
    else:
        fn.add(stmt + " = text(" + Q + sh.with_placeholder(":" + key) + Q + ")", i,
               rationale=True)
        fn.add(ctx.res + " = session.execute(" + stmt + ", {" + Q + key + Q
               + ": " + u + "})", i, rationale=True)
    fn.add("return " + ctx.res + "." + r.choice(["fetchall()", "mappings().all()",
                                                 "scalars().all()"]), i)


# --------------------------------------------------------------------------- #
# S05 -- ORM query builder, no textual SQL at all
# --------------------------------------------------------------------------- #

def s05_orm_query_builder(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    model = ctx.noun.capitalize()
    _distractors(fn, ctx, 0, 2)
    _comment(fn, ctx)
    variant = r.randint(0, 2)
    if variant == 0:
        fn.add(ctx.res + " = session.query(" + model + ").filter(" + model + "."
               + ctx.key_col + " == " + u + ")", i, rationale=True)
        fn.add("return " + ctx.res + "." + r.choice(["all()", "first()",
                                                     "limit(100).all()"]), i)
    elif variant == 1:
        fn.add("return " + model + ".objects.filter(" + ctx.key_col + "=" + u
               + ")." + r.choice(["values()", "all()", "order_by("
                                  + Q + "-id" + Q + ")"]), i, rationale=True)
    else:
        fn.add(ctx.q + " = select(" + model + ").where(" + model + "."
               + ctx.key_col + " == " + u + ")", i, rationale=True)
        fn.add("return session.scalars(" + ctx.q + ").all()", i)


# --------------------------------------------------------------------------- #
# S06 -- HARD NEGATIVE: integer cast before interpolation
# --------------------------------------------------------------------------- #

def s06_int_cast(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    num = r.choice(["numeric_id", "row_id", "ident", "n", "pk"])
    cast = r.choice([
        "int(" + u + ")",
        "int(" + u + ", 10)",
        "operator.index(int(" + u + "))",
    ])
    if r.random() < 0.45:
        fn.add("try:", i)
        fn.add(num + " = " + cast, i + 1, rationale=True)
        fn.add("except (TypeError, ValueError):", i)
        fn.add("raise ValueError(" + Q + "identifier must be an integer" + Q + ")", i + 1)
    else:
        fn.add(num + " = " + cast, i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    # Numeric slot: no quotes, and the value is provably an int.
    sql_txt = ("SELECT " + ctx.col_list + " FROM " + ctx.table + " WHERE "
               + ctx.key_col + " = {" + num + "}" + sh.tail_only)
    fn.add(ctx.q + " = f" + Q + sql_txt + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S07 -- HARD NEGATIVE: identifier checked against an allow-list
# --------------------------------------------------------------------------- #

def s07_allowlisted_identifier(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i = ctx.rng, ctx.ind
    col_param = r.choice(["sort_column", "order_by", "sort_field"])
    ctx.sig_extra.append(col_param)
    key_arg = r.choice(["tenant_id", "owner_id", "account_id"])
    ctx.sig_extra.append(key_arg)
    allowed = r.choice(["ALLOWED_SORT_COLUMNS", "SORTABLE", "_SAFE_COLUMNS"])
    allow_literal = "{" + ", ".join(Q + c + Q for c in ctx.cols) + "}"
    fn.add(allowed + " = " + allow_literal, i, rationale=True)
    guard = r.random()
    if guard < 0.5:
        fn.add("if " + col_param + " not in " + allowed + ":", i, rationale=True)
        fn.add("raise ValueError(" + Q + "unsupported sort column" + Q + ")", i + 1,
               rationale=True)
    else:
        fn.add(col_param + " = " + col_param + " if " + col_param + " in "
               + allowed + " else " + Q + "id" + Q, i, rationale=True)
    _cursor_open(fn, ctx)
    _comment(fn, ctx, p=0.6)
    sql_txt = ("SELECT " + ctx.col_list + " FROM " + ctx.table + " WHERE "
               + ctx.key_col + " = " + ctx.ph + " ORDER BY {" + col_param + "} "
               + r.choice(["ASC", "DESC"]))
    fn.add(ctx.q + " = f" + Q + sql_txt + Q, i)
    fn.add(ctx.cur + ".execute(" + ctx.q + ", " + _params_tuple(ctx, key_arg) + ")", i)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S08 -- HARD NEGATIVE: user key maps to a fixed SQL fragment
# --------------------------------------------------------------------------- #

def s08_enum_fragment_lookup(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    table_name = r.choice(["FILTER_SQL", "_CLAUSES", "PREDICATES"])
    clause = r.choice(["clause", "predicate", "cond"])
    fn.add(table_name + " = {", i, rationale=True)
    fn.add(Q + "active" + Q + ": " + Q + "is_active = 1" + Q + ",", i + 1, rationale=True)
    fn.add(Q + "archived" + Q + ": " + Q + "is_active = 0" + Q + ",", i + 1, rationale=True)
    fn.add(Q + "recent" + Q + ": " + Q + "created_at > date('now','-7 day')" + Q + ",",
           i + 1, rationale=True)
    fn.add("}", i)
    fn.add(clause + " = " + table_name + ".get(" + u + ", " + Q + "1=1" + Q + ")", i,
           rationale=True)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    sql_txt = ("SELECT " + ctx.col_list + " FROM " + ctx.table + " WHERE {"
               + clause + "}" + sh.tail_only)
    fn.add(ctx.q + " = f" + Q + sql_txt + Q, i)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S09 -- HARD NEGATIVE: value accepted only if it matches a strict pattern
# --------------------------------------------------------------------------- #

def s09_regex_validated(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    pat = r.choice([
        r"[A-Za-z0-9_]{1,32}",
        r"[a-z]{2,16}",
        r"\d{1,12}",
        r"[A-Z]{2,3}-\d{4,8}",
        r"[a-f0-9]{32}",
    ])
    style = r.random()
    if style < 0.45:
        fn.add("if not re.fullmatch(r" + Q + pat + Q + ", " + u + " or " + Q + Q
               + "):", i, rationale=True)
        fn.add("raise ValueError(" + Q + "malformed identifier" + Q + ")", i + 1,
               rationale=True)
    elif style < 0.75:
        fn.add("if not " + u + ".isalnum() or len(" + u + ") > 32:", i, rationale=True)
        fn.add("return []", i + 1, rationale=True)
    else:
        fn.add("assert re.fullmatch(r" + Q + pat + Q + ", " + u
               + "), " + Q + "invalid token" + Q, i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    fn.add(ctx.q + " = f" + Q + sh.fstring_body(u) + Q, i)
    fn.add(ctx.cur + ".execute(" + ctx.q + ")", i)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S10 -- HARD NEGATIVE: f-string interpolates module constants only
# --------------------------------------------------------------------------- #

def s10_literal_only_fstring(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    tv = r.choice(["table", "tbl", "target_table"])
    cv = r.choice(["cols", "column_list", "projection"])
    fn.add(tv + " = " + r.choice(["TABLE_NAME", Q + ctx.table + Q,
                                  "SCHEMA[" + Q + ctx.noun + Q + "]"]), i,
           rationale=True)
    fn.add(cv + " = " + Q + ", " + Q + ".join(DEFAULT_COLUMNS)", i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx, p=0.6)
    sql_txt = ("SELECT {" + cv + "} FROM {" + tv + "} WHERE " + ctx.key_col
               + " = " + ctx.ph + sh.tail_only)
    fn.add(ctx.q + " = f" + Q + sql_txt + Q, i, rationale=True)
    fn.add(ctx.cur + ".execute(" + ctx.q + ", " + _params_tuple(ctx, u) + ")", i,
           rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S11 -- batched parameterised write
# --------------------------------------------------------------------------- #

def s11_executemany_params(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i = ctx.rng, ctx.ind
    ctx.sig_extra.append("records")
    rows = r.choice(["rows", "batch", "payloads", "tuples"])
    c1, c2 = (ctx.cols + ["name", "status"])[:2]
    ph = ctx.ph
    fn.add(rows + " = [(r[" + Q + c1 + Q + "], r[" + Q + c2 + Q
           + "]) for r in records]", i, rationale=True)
    _distractors(fn, ctx, 0, 2)
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    sql_txt = ("INSERT INTO " + ctx.table + " (" + c1 + ", " + c2 + ") VALUES ("
               + ph + ", " + ph + ")")
    fn.add(ctx.q + " = " + Q + sql_txt + Q, i, rationale=True)
    fn.add(ctx.cur + ".executemany(" + ctx.q + ", " + rows + ")", i, rationale=True)
    fn.add(ctx.conn + ".commit()", i)
    fn.add("return len(" + rows + ")", i)


# --------------------------------------------------------------------------- #
# S12 -- HARD NEGATIVE: SQL-shaped text that is never executed
# --------------------------------------------------------------------------- #

def s12_sql_string_not_executed(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    desc = r.choice(["description", "human_filter", "preview", "explain_text"])
    _distractors(fn, ctx, 0, 2)
    # Renders a description of the query for the UI / audit trail.  There is no
    # cursor and no execute() anywhere in this function.
    fn.add(desc + " = f" + Q + sh.fstring_body(u) + Q, i, rationale=True)
    fn.add("# documentation only: this string is rendered, never executed", i,
           rationale=True)
    if r.random() < 0.5:
        fn.add(ctx.logger + ".info(" + Q + "filter preview: %s" + Q + ", " + desc + ")", i)
    fn.add("return {" + Q + "table" + Q + ": " + Q + ctx.table + Q + ", "
           + Q + "preview" + Q + ": " + desc + "}", i, rationale=True)


# --------------------------------------------------------------------------- #
# S13 -- user data interpolated into a non-SQL sink
# --------------------------------------------------------------------------- #

def s13_fstring_non_sql_sink(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    variant = r.randint(0, 3)
    _distractors(fn, ctx, 0, 2)
    if variant == 0:
        fn.add("safe_text = html.escape(" + u + ")", i, rationale=True)
        fn.add("return f" + Q + "<li class='row'>{safe_text}</li>" + Q, i,
               rationale=True)
    elif variant == 1:
        fn.add("name = os.path.basename(str(" + u + "))", i, rationale=True)
        fn.add("path = os.path.join(EXPORT_DIR, f" + Q + "{name}.json" + Q + ")", i,
               rationale=True)
        fn.add("return path", i)
    elif variant == 2:
        fn.add(ctx.logger + ".info(f" + Q + "lookup requested for {" + u + "!r}" + Q
               + ")", i, rationale=True)
        fn.add("return {" + Q + "echo" + Q + ": str(" + u + ")}", i)
    else:
        fn.add("key = f" + Q + ctx.table + ":{" + u + "}" + Q, i, rationale=True)
        fn.add("return cache.get(key)", i, rationale=True)


# --------------------------------------------------------------------------- #
# S14 -- stored procedure / prepared call
# --------------------------------------------------------------------------- #

def s14_stored_procedure(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    proc = r.choice(["sp_get_" + ctx.noun, "usp_lookup", ctx.noun + "_by_key",
                     "fetch_" + ctx.noun])
    _cursor_open(fn, ctx)
    _comment(fn, ctx)
    if r.random() < 0.55:
        fn.add(ctx.cur + ".callproc(" + Q + proc + Q + ", [" + u + "])", i,
               rationale=True)
    else:
        fn.add(ctx.q + " = " + Q + "{CALL " + proc + "(" + ctx.ph + ")}" + Q, i,
               rationale=True)
        fn.add(ctx.cur + ".execute(" + ctx.q + ", " + _params_tuple(ctx, u) + ")", i,
               rationale=True)
    _fetch_and_return(fn, ctx)


# --------------------------------------------------------------------------- #
# S15 -- HARD NEGATIVE: composable-SQL API (.format on a SQL object)
# --------------------------------------------------------------------------- #

def s15_composable_identifier_api(fn: Fn, ctx: Ctx, sh: Shape) -> None:
    r, i, u = ctx.rng, ctx.ind, ctx.user_param
    col_param = r.choice(["sort_column", "target_column", "col"])
    ctx.sig_extra.append(col_param)
    stmt = r.choice(["stmt", "composed", "query_obj"])
    _cursor_open(fn, ctx)
    _comment(fn, ctx, p=0.6)
    # The connection line may name any of the five drivers even though
    # `psycopg2.sql` is psycopg2-specific.  That mismatch is deliberate: the API
    # is sampled independently of the family throughout the corpus, so that no
    # driver name becomes predictive of a label.  Pinning this family to
    # psycopg2 would make `psycopg2.connect` a giveaway for S15 and weaken the
    # very control the generator exists to enforce.
    #
    # psycopg2.sql quotes identifiers at the driver level -- injection-proof.
    fn.add(stmt + " = sql.SQL(" + Q + "SELECT {} FROM {} WHERE {} = %s" + Q
           + ").format(", i, rationale=True)
    fn.add("sql.SQL(" + Q + ", " + Q + ").join(map(sql.Identifier, DEFAULT_COLUMNS)),",
           i + 1, rationale=True)
    fn.add("sql.Identifier(" + Q + ctx.table + Q + "),", i + 1, rationale=True)
    fn.add("sql.Identifier(" + col_param + "),", i + 1, rationale=True)
    fn.add(")", i)
    fn.add(ctx.cur + ".execute(" + stmt + ", " + _params_tuple(ctx, u) + ")", i,
           rationale=True)
    _fetch_and_return(fn, ctx)


SAFE_BUILDERS = {
    "S01_qmark_params": s01_qmark_params,
    "S02_pyformat_params": s02_pyformat_params,
    "S03_named_params": s03_named_params,
    "S04_sqlalchemy_bindparams": s04_sqlalchemy_bindparams,
    "S05_orm_query_builder": s05_orm_query_builder,
    "S06_int_cast": s06_int_cast,
    "S07_allowlisted_identifier": s07_allowlisted_identifier,
    "S08_enum_fragment_lookup": s08_enum_fragment_lookup,
    "S09_regex_validated": s09_regex_validated,
    "S10_literal_only_fstring": s10_literal_only_fstring,
    "S11_executemany_params": s11_executemany_params,
    "S12_sql_string_not_executed": s12_sql_string_not_executed,
    "S13_fstring_non_sql_sink": s13_fstring_non_sql_sink,
    "S14_stored_procedure": s14_stored_procedure,
    "S15_composable_identifier_api": s15_composable_identifier_api,
}
