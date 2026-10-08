"""Lexical pools used by the synthetic corpus generator.

Keeping the pools in one place lets us (a) guarantee that the *same* identifier
distribution is shared by vulnerable and safe samples -- so a classifier cannot
win by memorising names -- and (b) report corpus statistics in the write-up.
"""

TABLES = [
    "users", "orders", "products", "accounts", "customers", "invoices",
    "employees", "sessions", "payments", "audit_log", "tickets", "devices",
    "subscriptions", "inventory", "shipments", "reviews", "categories",
    "transactions", "credentials", "permissions", "projects", "tasks",
    "messages", "attachments", "branches", "suppliers", "contracts",
]

COLUMNS = [
    "id", "uuid", "name", "email", "username", "status", "created_at",
    "updated_at", "amount", "total", "quantity", "price", "country",
    "city", "role", "department", "is_active", "last_login", "score",
    "category_id", "owner_id", "tenant_id", "sku", "title", "body",
    "priority", "due_date", "currency", "balance", "region",
]

# Parameter names that plausibly carry *untrusted* data.
USER_PARAMS = [
    "user_id", "username", "email", "search_term", "query", "keyword",
    "customer_name", "order_ref", "filter_value", "token", "account_no",
    "product_code", "city", "status_filter", "sort_column", "page_token",
    "client_ref", "tag", "category", "lookup_key", "raw_input", "user_input",
    "request_id", "external_id", "supplier_code", "search_text",
]

VERBS = [
    "get", "fetch", "load", "find", "lookup", "query", "retrieve", "read",
    "select", "search", "list", "count", "resolve", "collect", "pull",
]
VERBS_WRITE = ["update", "set", "upsert", "save", "store", "delete", "remove", "archive"]

NOUNS = [
    "user", "order", "product", "account", "customer", "invoice", "employee",
    "session", "payment", "ticket", "device", "subscription", "record",
    "entry", "row", "profile", "detail", "summary", "report", "item",
]
QUALIFIERS = ["by_id", "by_name", "by_email", "for_tenant", "by_status", "by_code", "", "", ""]

LOCAL_VARS = [
    "rows", "result", "records", "data", "out", "items", "payload", "cursor_rows",
    "resp", "buf", "acc", "found", "matches", "hits", "entries",
]

QUERY_VARS = ["sql", "query", "stmt", "q", "sql_text", "statement", "cmd", "query_str"]
CURSOR_VARS = ["cur", "cursor", "c", "db_cur", "crs"]
CONN_VARS = ["conn", "connection", "db", "cnx", "handle"]

LOGGERS = ["logger", "log", "LOG", "_log"]

# Comments. Some are deliberately misleading ("sanitised", "safe") so that a
# bag-of-words model cannot rely on comment text.
COMMENTS_NEUTRAL = [
    "# build the lookup statement",
    "# talk to the primary replica",
    "# single round-trip is enough here",
    "# NOTE: keep the column list explicit",
    "# cheap path, no join required",
    "# called from the REST layer",
    "# see ticket OPS-1182",
    "# legacy behaviour, do not change silently",
]
COMMENTS_MISLEADING = [
    "# input is already sanitised upstream",
    "# safe: validated by the API schema",
    "# trusted caller only",
    "# escaping handled by the driver",
    "# no injection risk here",
]

DOCSTRINGS = [
    "Return the matching {noun} rows.",
    "Look up a single {noun} and return it.",
    "Fetch {noun} records for the caller.",
    "Resolve the {noun} referenced by the request.",
    "Read {noun} data from the operational store.",
    "Return every {noun} that matches the supplied filter.",
    "Count the {noun} rows currently visible to the tenant.",
]

DECORATORS = [
    "@app.route('/{noun}', methods=['GET'])",
    "@router.get('/{noun}')",
    "@require_login",
    "@retry(times=3)",
    "@functools.lru_cache(maxsize=256)",
    "@staticmethod",
    "@trace_span('db.{noun}')",
]

# Statements that have nothing to do with the SQL flow -- they lengthen the
# function and act as distractors for the attention mechanism.
DISTRACTORS = [
    "started = time.perf_counter()",
    "request_ctx = get_request_context()",
    "page_size = min(int(limit or 50), 500)",
    "trace_id = request_ctx.get('trace_id', '-')",
    "metrics.increment('db.lookup.attempt')",
    "cache_key = 'lookup:%d' % 7",
    "deadline = time.time() + 2.5",
    "retries_left = 3",
    "normalised = {}",
    "flags = set()",
    "columns_wanted = list(DEFAULT_COLUMNS)",
    "tenant = request_ctx.get('tenant', 'public')",
    "audit = {'actor': request_ctx.get('actor')}",
    "backoff = 0.25",
    "started_at = datetime.datetime.utcnow()",
    "opts = dict(timeout=5, readonly=True)",
]

API_SPECS = {
    "sqlite3": {
        "qmark": "?",
        "named": ":{k}",
        "connect": "sqlite3.connect(DB_PATH)",
        "error": "sqlite3.Error",
    },
    "psycopg2": {
        "qmark": "%s",
        "named": "%({k})s",
        "connect": "psycopg2.connect(DSN)",
        "error": "psycopg2.Error",
    },
    "mysql.connector": {
        "qmark": "%s",
        "named": "%({k})s",
        "connect": "mysql.connector.connect(**DB_CONFIG)",
        "error": "mysql.connector.Error",
    },
    "pymysql": {
        "qmark": "%s",
        "named": "%({k})s",
        "connect": "pymysql.connect(**DB_CONFIG)",
        "error": "pymysql.MySQLError",
    },
    "pyodbc": {
        "qmark": "?",
        "named": None,
        "connect": "pyodbc.connect(ODBC_DSN)",
        "error": "pyodbc.Error",
    },
}
