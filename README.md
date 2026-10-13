# DB-Sentinel

DB-Sentinel is a Python conversational data-analysis prototype. A user connects a supported database, asks a natural-language question, and the application plans an investigation, retrieves live schema context, generates read-only SQL or a MongoDB aggregation, validates it, executes it under limits, checks the results, and returns an evidence-backed answer.

Query generation is schema-grounded rather than tied to a sample database: each investigation step retrieves matching live table/collection metadata, and the SQL or MongoDB generator receives that schema and must use only its identifiers. SQL generation also receives the active SQL dialect. Date ranges are treated as filters; DB-Sentinel does not add period or segment comparisons unless the question asks for them.

It is not a general-purpose SQL terminal. The database is the source of evidence; the language model proposes plans and queries but does not decide whether a query is safe or whether evidence has passed deterministic checks.

## Current capabilities and boundaries

- Database adapters: PostgreSQL, MySQL, Microsoft SQL Server, and MongoDB.
- Natural-language question correction for spelling/grammar before intent classification and planning; the original question remains authoritative and numeric, quoted, and negation constraints are checked.
- Natural-language planning with sub-questions and real dependencies; explicit two-period comparisons and change requests use one required query that returns raw values per period. Differences and percentage changes are calculated only from verified values, never from a collapsed percentage-only query.
- Relative multi-year requests anchored to a stated year expand to inclusive calendar-year periods. Unanchored requests such as “the last 2 years” use the latest distinct calendar years actually present in the database rather than assuming the current year; trend summaries derive change from verified annual values.
- Why/driver questions require a verified baseline and add a schema-grounded segment comparison when useful. Explicitly requested dimensions are required; exploratory breakdowns stay supplemental. Results can identify observed associations, but unsupported causal explanations are rejected.
- Live metadata discovery. MongoDB metadata samples bounded documents to expose observed nested object/array field paths, such as item price and quantity fields. A data dictionary is optional and is not needed to discover tables or columns.
- Read-only query policy, dialect-aware SQL validation, MongoDB pipeline validation, bounded per-query pages with complete ordered pagination, and semantic/logical verification.
- Conversation context for follow-up questions within the running CLI session.
- Dashboard/report data structures returned with verified investigations.
- Local connection-profile persistence in the project `.env`; profiles are restored for the selected organization when the CLI starts. Restarting closes the old process's network clients; selecting a restored MongoDB profile opens a new client on demand.
- Optional JSON, CSV, or Excel dictionaries are supported by the loader, but dictionary loading is intentionally not prompted in the current CLI.
- Removing a saved connection removes only its local profile. It never deletes a database or its records.

This repository currently runs as a CLI prototype. Tenant and workspace identifiers are application context, not an authentication system. Connections are held in memory during a process and profiles are stored locally as plaintext URLs in `.env`. Do not treat it as a hosted multi-tenant SaaS or give it write-capable database credentials without adding production authentication, encrypted secret storage, persistent audit storage, and operational controls.

## Requirements

- Python 3.10 or newer (the project uses modern built-in generic/type syntax).
- Network access to the database and configured LLM provider.
- PostgreSQL: `psycopg2-binary`; MySQL: PyMySQL; SQL Server: `pyodbc` plus a compatible ODBC driver; MongoDB: PyMongo.
- Provider credentials for at least one enabled language-model provider. Current providers are configured from environment variables (for example, `GEMINI_API_KEY` or `GROQ_API_KEY`).

## Install and run (Windows PowerShell)

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -m app.main
```

Edit `.env` and set at least one supported LLM provider key before starting. Do not add database URLs, provider keys, or passwords to Git. Python-dotenv loads `.env` at startup.

The CLI asks for an organization ID. It defaults to the sole organization with saved profiles when exactly one exists; otherwise it uses `local`. Connected databases are restored from the local profile list for that organization. Use `connect` to add a connection:

- `url`: enter a SQLAlchemy database URL or MongoDB URI at the hidden prompt. For MongoDB, if the URI omits its database path, the CLI prompts for the database name and adds it to the saved URI.
- `fields`: enter database type, host, port, database, username, and SSL settings.
- On failure, select a displayed field to correct it and retry; passwords/URLs are hidden when displayed.

Supported URL schemes include `postgresql://`, `postgres://`, `mysql://`, `mssql://`, `sqlserver://`, `mongodb://`, and `mongodb+srv://`. Percent-encode reserved characters in credentials (for example, `@` as `%40` and `#` as `%23`); a raw `#` starts a URI fragment and cannot be used within a password.

Main CLI commands:

| Command | What it does |
| --- | --- |
| `connect` | Add and test a database connection; successful profiles are saved locally. |
| `use` | Select a connected database and ask questions. |
| `list` | Refresh the connected-database list. |
| `delete` | Remove a selected saved connection after confirmation. This does not modify the remote database. |
| `quit` | Exit the application. |
| `/back` | Leave the current database conversation and return to the connection menu. |

## Configuration

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY`, `GROQ_API_KEY` | Credentials for the corresponding model providers. |
| `DB_SENTINEL_DATABASE_URL` | Optional generic default connection URL. |
| `DATABASE_URL`, `DB_URL` | Generic URL aliases accepted by the CLI. |
| `DB_TYPE`, `DB_HOST`, `DB_PORT`, `DB_DATABASE_NAME`/`DB_NAME`, `DB_USER`, `DB_PASSWORD` | Optional defaults for field-based connections. |
| `DB_DISPLAY_NAME`, `DB_WORKSPACE_ID`, `DB_SSL_ENABLED` | Optional connection metadata and SSL defaults. |
| `DB_SENTINEL_ENV_FILE` | Override the `.env` file path used when saving/removing profiles. |
| `DB_SENTINEL_LOG_DIR` | Directory for `db-sentinel.log` (defaults to `logs`). |
| `DB_SENTINEL_LOG_LEVEL` | Python logging level (defaults to `DEBUG`). |

Successful connection profiles are written to `DB_SENTINEL_CONNECTIONS_JSON` in the selected environment file, which is the source of truth when restoring profiles. This value includes connection URLs and credentials. Restrict access to the file and machine account. Never commit or share `.env`.

## How an answer is produced

```text
Question
  -> request policy and operation permissions
  -> constrained grammar/spelling normalization (original retained)
  -> normalized-question policy check
  -> intent classification (metadata exploration or analysis)
  -> model-driven investigation plan
  -> live schema selection and optional dictionary hints
  -> SQL/Mongo query generation
  -> deterministic query validation
  -> read-only, time-bounded execution
  -> result, semantic and logical verification
  -> evidence store
  -> grounded insight + insight verification
  -> CLI answer, evidence, dashboard/report artifacts
```

Complex questions can have dependent steps. A failed required step prevents a final answer from claiming the whole investigation was verified; optional contextual steps may fail without discarding verified required evidence, and their limitations are reported. Grammar/spelling correction is constrained to preserve numeric filters, quoted values, and negation, and the original wording remains authoritative for user intent. Comparison/change requests between explicitly named years use one grouped query even when phrased without the word “compare.” If the verified date range covers both years but the database returns a grouped result for only one, DB-Sentinel reports that the comparison is unavailable rather than claiming an increase/decrease or repeatedly retrying the same range. Time-scoped metric questions may include optional preceding-period and schema-grounded segment comparisons. These are plans, not hardcoded dataset values: actual tables, columns, metrics and available periods are discovered from the connected database.

For supported relational complete-list requests, DB-Sentinel wraps the generated result in one read-only SQL statement that counts the full result on the database server and returns only an ordered preview of at most 100 rows. The exact total and preview are reported in the terminal, which displays up to 100 rows and clearly indicates when the output cap is reached. The full result is not copied into application memory. This reduces client-side memory and avoids repeated page requests, but the database still performs the work needed to compute the complete result and may hit its query timeout on expensive scans. Queries that cannot be safely wrapped use the existing bounded, ordered pagination path; Top-N requests return only the requested ranking.

Implicit maximum/minimum questions return every entity tied at the requested extreme rather than silently selecting one row. The ranking query is expected to reduce relationship rows to one metric per requested entity, and DB-Sentinel wraps supported relational queries to calculate the exact tie count while retaining at most 100 deterministically ordered evidence rows. The terminal summary reports the extreme metric, total ties, and preview cap. If a query cannot be safely transformed, DB-Sentinel fails closed instead of presenting a partial top-one result.

## Architecture map

- `app/main.py`: CLI, connection wizard, saved-profile lifecycle, result presentation.
- `app/agent/`: planner, investigation orchestration, query generation, evidence and verification.
- `app/analysis/`: intent classification and metadata exploration.
- `app/database/`: connection manager, schema discovery, dictionaries, adapters and bounded executor.
- `app/security/`: request policy, read-only permissions, SQL and MongoDB query guards.
- `app/llm/`: provider abstraction, model registry, complexity classification and routing.
- `app/conversation/`, `app/tenant/`: conversation history and organization/workspace boundaries.
- `app/reporting/`: dashboard/report artifact models.
- `app/models/`: typed database, SQL, schema and MongoDB data contracts.
- `docs/ARCHITECTURE_GUIDE.md`: detailed package/file explanations, data flow, invariants, troubleshooting and investor explanation.
- `docs/DB-Sentinel-Architecture-Guide.docx`: Word version of the detailed guide.

## Logs and privacy

`logs/db-sentinel.log` records investigation/step transitions, organization ID, workspace/team ID, connection ID, SQL fingerprints, the executed SQL text, and execution outcomes. It does not intentionally store query result rows or connection URLs. SQL literals can contain user-provided filter values, so log files are sensitive operational data: restrict access and retention. There is no authenticated end-user identity in this CLI prototype, so logs cannot identify an individual beyond organization/workspace/connection context.

## Data dictionary

Dictionary parsing accepts JSON, CSV, and Excel (`.xlsx`/`.xlsm`) and validates descriptions against live metadata. CSV/Excel support schema/table/column descriptions and WHO-style `variable_name`, `dataset`, `definition`, and optional `code_list` rows. The dictionary is supplemental semantic context only; it cannot create database objects or replace live evidence. The current CLI does not prompt for dictionary loading while query handling is being stabilized.

## Tests and checks

Run tests:

```powershell
python -m pytest -q
python -m compileall -q app
git diff --check
```

Tests use local fixtures/mocks and do not constitute proof that every database engine, provider, schema, or live natural-language question is correct. Live connection and query validation must be performed against a test database with read-only credentials and reviewed SQL.

## Documentation maintenance

Update `docs/ARCHITECTURE_GUIDE.md` when code contracts or the flow change, then regenerate the Word file:

```powershell
python scripts\build_architecture_docx.py
```
