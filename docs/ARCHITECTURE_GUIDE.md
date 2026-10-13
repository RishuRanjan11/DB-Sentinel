# DB-Sentinel: Architecture and Developer Guide

**Purpose:** Explain how the current codebase is organized, how an answer travels through it, why major safety and verification layers exist, how to run and test it, and how to diagnose common failures.

**Scope note:** This guide describes the implementation in this repository today. DB-Sentinel is presently a local Python CLI prototype with modular adapters and analysis components. It is not yet a deployed, authenticated SaaS service. Do not infer production tenancy, encrypted credentials, or write support from the architecture vocabulary.

## 1. What the application does

DB-Sentinel turns a natural-language question into a bounded, read-only database investigation. The language model helps interpret intent and propose a plan/query. Python code is responsible for enforcing tenant context, discovering real schema, validating the query, executing it under limits, checking the returned evidence, and deciding whether the user-facing claim is supported.

The most useful mental model is:

```text
User question
  -> deterministic request policy
  -> constrained grammar/spelling normalization
  -> normalized-request policy check and intent classifier
  -> planner (one step or dependent sub-questions)
  -> schema retrieval from the selected database
  -> SQL or MongoDB pipeline proposal
  -> deterministic allow-list and read-only validation
  -> bounded database execution
  -> result + semantic + logical verification
  -> evidence store
  -> insight generation and verification
  -> CLI answer and report/dashboard structures
```

The database, not the language model, is the source of factual evidence. If a model proposes a column absent from live metadata, the query guard rejects it. If a query executes but its evidence is incomplete or its insight is unsupported, the system must not present it as a fully verified answer.

## 2. Source layout and ownership

```text
app/
  main.py                    CLI and connection/profile lifecycle
  agent/                     planning, investigation, SQL, evidence, verification
  analysis/                  request intent and database metadata exploration
  conversation/              conversation history and analyst façade
  core/                      configuration, terminal formatting and logging
  database/                  connection models, adapters, discovery, execution
  llm/                       providers, model registry, routing, complexity
  models/                    typed data contracts
  reporting/                 dashboard and report structures
  security/                  operation policies and query validation
  tenant/                    organization-scoped database context
docs/                        this guide and its Word version
scripts/                     documentation generation utility
```

The package boundaries are intentional. The CLI should not decide whether SQL is safe; the model provider should not decide which organization owns a connection; the insight generator should not invent data facts. Each layer has a narrower responsibility so errors can be diagnosed at the boundary where they occur.

## 3. End-to-end request lifecycle

### 3.1 Start and select a connection

`app/main.py` loads environment settings and reads the saved profile array from the selected `.env` file. A profile records a connection ID, organization ID, workspace ID, display name, and URL. The URL necessarily contains authentication information; this is local plaintext persistence, not a secrets vault.

At startup the CLI selects an organization, restores its profiles into `DatabaseManager`, and presents `connect`, `use`, `list`, `delete`, and `quit`. A connection is tested before it is persisted. `delete` removes the local profile and closes that adapter; it does not issue SQL against the remote database and cannot delete remote records.

Connection input can use a URL/URI or separate fields. If the connection test fails, the wizard displays editable field choices and retries with the changed field rather than restarting the whole form. Secret-bearing inputs are hidden; connection URLs and passwords are not written to the SQL audit event.

### 3.2 Enforce the request boundary

`InvestigationEngine.investigate` first verifies that the selected connection belongs to the requested organization. It next applies `RequestPolicy` and `DatabasePermissions`. The current product is deliberately read-only: requests to write rows, alter schema, change privileges, transact, or retrieve expressly sensitive data are blocked before query generation.

This policy exists separately from the SQL guard. Natural language is screened before asking a model to generate a query; generated SQL is independently inspected before execution. Neither check is a substitute for read-only database credentials.

### 3.3 Classify and plan

`AnalysisIntentClassifier` routes database-metadata questions (such as asking for tables or columns) to `DatabaseExplorer`. Analytical questions go to `QuestionPlanner`.

The planner returns typed `InvestigationPlan` and `SubQuestion` objects. Each step contains a question, objective, required information, dependency IDs, and an optional flag. The planner is schema-agnostic: it must not invent SQL objects. Self-contained metric rankings are normalized to one step before invoking the model planner, preserving ascending versus descending direction and avoiding redundant prerequisite queries. Rankings that contrast measures retain all requested measures in that one step. Relative multi-year questions that request an entity breakdown retain both the entity and year grouping, rather than truncating all entities to a limit equal to the year count. When the user leaves a relative benchmark unspecified, the insight layer uses the median among returned entities and states that basis. Dependency validation checks unique IDs, earlier-step ordering, and cycles.

Date/time ranges are treated as filters on the requested records or measurements. Prior-period, trend, and segment-comparison steps are added only when the user explicitly requests comparison/change analysis. Generated plans are pruned of unsolicited period-comparison steps before execution.

### 3.4 Retrieve live schema

Schema discovery is done by the active database adapter. `schema_retriever.py` ranks tables/columns against the business language and can use optional dictionary descriptions as matching context. The live schema is authoritative for query validation. A dictionary adds descriptions; it cannot manufacture a table or column.

The PostgreSQL adapter filters platform-managed schemas only for recognized Supabase hosts. This avoids presenting provider internals as customer data while leaving non-Supabase PostgreSQL schema discovery alone. A single-visible-table fallback may provide that table's real metadata when no business words overlap its abbreviated names; it is not a TB-specific table rule.

### 3.5 Generate, validate and execute

`SQLGenerator` or `MongoQueryGenerator` asks the configured model router for a typed query result using the question, selected schema, feedback, and prior attempts. The pipeline rejects repeated failed SQL and validates table/collection and column references against discovered schema.

Relational queries pass through `SQLSecurityValidator`, which parses SQL and enforces the current read-only policy, allowed objects, blocked sensitive columns, dialect rules, and result-size requirements. MongoDB aggregation uses its own validator, including allowed collection/field checks and blocked stages/expressions.

`QueryIntentPlanner` also produces a typed metric contract. A metric can be a source field or a schema-grounded arithmetic formula. For sales/revenue over line items, its standard definition is the sum of each line's unit price multiplied by its quantity. Formula fields must exist in the selected live schema; MongoDB generation enforces the contract in its grouped accumulator, while the relational pipeline verifies the generated aggregate expression. A query that cannot honor the formula is not executed as valid evidence. When a query attempt fails, the intent is re-evaluated with the exact repair diagnosis and prior typed contract, so a wrong interpretation can be repaired as well as the generated SQL/pipeline while preserving unrelated valid constraints.

`SecureQueryExecutor` delegates to an adapter's read-only execution path with a statement timeout and maximum returned row count. It writes a SQL audit event for rejected, executed, and failed attempts. Events include organization, workspace/team, and connection identifiers, a SQL fingerprint, query text, and an outcome. The CLI has no authenticated individual-user identity, so no per-person actor is asserted.

### 3.6 Verify evidence and answer

Execution success is not equivalent to answer correctness. The pipeline checks result shape and semantics, then the logical verifier checks whether the requested filters, aggregations, ranking, limits, and comparison survived generation. For relative entity trends, complete entity/year-grouped annual metric rows are valid evidence even when the database pipeline does not itself filter entities by trend; the deterministic insight analyzer compares the annual values and reports only consistent increases or declines. Evidence objects retain the step, original intent, SQL/pipeline, returned rows, schema context, verification reason, confidence, dependencies, and pagination completeness.

The `EvidenceStore` is the set of verified step results. Required plan failures block a complete final answer. Optional plan failures can be disclosed as uncertainty/limitations. `InsightGenerator` summarizes evidence, using deterministic summaries where possible and a model only when necessary. Contrastive high/low rankings are evaluated using both measures on each entity row; the high measure and the low measure normalized by the high measure are compared to their respective medians across returned entities. The resulting entity/value associations are rechecked by `InsightVerifier`. Numeric/entity/ranking claims are checked against rows and unsupported claims are flagged. Recommendations are kept separate and must be evidence-supported; unsupported causal language is not a valid recommendation.

The CLI prints status, insight, findings, recommended actions, limitations, and relevant evidence. `build_artifacts` constructs typed dashboard and report objects from the same final insight and evidence payload.

## 4. Trust boundaries and invariants

1. **Connection ownership:** a connection belongs to one organization ID; access with a mismatched organization is denied.
2. **Schema truth:** generated identifiers must be present in live metadata. Dictionary prose is a hint, not proof.
3. **Read-only operation:** request policy, database permission policy, query validators, adapter read-only methods, timeouts, and row caps all constrain execution.
4. **Evidence-grounded output:** numeric claims and ranking/entity mappings must be traceable to result rows.
5. **Completeness is explicit and memory-bounded:** supported relational complete-list requests use a single read-only statement with a window count over the full result and an outer limit of 100 preview rows. Implicit maximum/minimum requests likewise compute the extreme across the complete generated result, retain tied rows only up to the 100-row evidence cap, and carry the exact tie count and metric into deterministic summary and verification. Ranking queries must reduce relationship rows to one metric per requested entity. The application retains only the preview and exact total. The database still computes the complete result and may hit its query timeout on expensive scans. If a query cannot be safely wrapped, the existing deterministic bounded-pagination path remains available for ordinary collection requests; unsupported extreme-tie transformations fail closed rather than accepting an arbitrary top-one row. Exact requested Top-N can complete without fetching unrelated rows.
6. **Dependencies are real:** a step is blocked if a required predecessor has no verified evidence.
7. **Optional context is not the answer:** failure to retrieve optional context does not negate successful required evidence, but it must be disclosed.
8. **Local profiles are secrets:** saved connection URLs are plaintext in `.env`; protect it and never commit it.
9. **Logs can be sensitive:** SQL text may contain user-entered literal filters. Logs intentionally omit result rows and connection URLs; protect log files and set a retention policy.
10. **No destructive database delete feature:** `delete` is only for a saved DB-Sentinel connection profile. Database writes remain blocked.

## 5. Module-by-module code guide

### Application and tests

| File | Responsibility |
| --- | --- |
| `app/main.py` | CLI entry point; parses URL/URI connections; manages profile persistence, restoration and removal; performs connection tests; runs organization/database selection and conversations; renders results. |
| `app/test_connection_url.py` | URL scheme/credential parsing, profile persistence/restoration/deletion, and correction-after-failure tests. |
| `app/test_main_presentation.py` | CLI presentation regression tests for metadata, recommendation and limitation output. |
| `app/__init__.py` | Marks `app` as an importable Python package. |

### Investigation agent (`app/agent/`)

| File | Responsibility |
| --- | --- |
| `question_normalizer.py` | Corrects grammar/spelling and rejects rewrites that alter numeric constraints, quoted values, or negation. |
| `planner.py` | Pydantic plan/step contracts, model-driven decomposition, direct lookup/ranking normalization, time-period supplemental analysis, dependency validation and conservative “why” cleanup. |
| `query_intent.py` | Typed, schema-grounded per-step query contract, including field-based and arithmetic metrics, filters, dimensions, ranking and result requirements. |
| `investigation.py` | Main orchestrator: tenant check, policy, metadata route, planning, bounded attempts, dependency context, evidence collection, insight verification, status and artifacts. |
| `sql_pipeline.py` | Relational/Mongo query generation, schema grounding, security validation, execution, pagination, result verification and query retry feedback. |
| `sql_generator.py` | Creates typed relational SQL proposals from business wording and schema context; it is not the security authority. |
| `mongo_generator.py` | Creates typed MongoDB aggregation proposals using collection/field context. Each generated stage has an explicit operator and JSON-encoded argument, then is serialized into MongoDB's one-operator-per-stage pipeline shape; this avoids structured-output providers collapsing arbitrary-key stage objects to `{}`. |
| `evidence.py` | `Evidence` record and `EvidenceStore`; stores only step-level verified query results for final synthesis. |
| `result_verifier.py` | Deterministic structural checks on executed query result sets. |
| `semantic_verifier.py` | Checks whether returned values/results appear relevant to the analytical request. |
| `logical_verifier.py` | Checks preservation of question constraints such as grouping, ranking direction, Top-N, and requested dimensions. |
| `insight_generator.py` | Builds a user-facing structured insight from evidence; deterministic aggregate/ranking paths reduce opportunities for unsupported narrative. |
| `insight_verifier.py` | Validates numeric/entity/ranking claims and flags unsupported or potentially causal claims. |
| `llm_client.py` | Legacy/compatibility LLM client entry point; inspect call sites before changing or removing it. |
| `test_security_pipeline.py` | Regressions for request/query security boundaries. |
| `test_sql_pipeline.py` | Query-pipeline behavior tests. |
| `test_verification_guards.py` | Regressions for ranking, pagination, optional steps, plan normalization and recommendation propagation. |
| `agent/__init__.py` | Package marker. |

### Analysis (`app/analysis/`)

| File | Responsibility |
| --- | --- |
| `intent.py` | Classifies metadata exploration versus analytical questions and contains classifier testable rules. |
| `explorer.py` | Reads live schemas/tables/relationships and builds database-overview evidence without text-to-SQL. |
| `test_intent.py` | Intent classifier tests. |
| `test_explorer.py` | Metadata exploration tests. |

### Conversations and tenant context

| File | Responsibility |
| --- | --- |
| `conversation/analyst.py` | Thin façade passing session organization, allowed connection IDs, history, and the new question to the investigation engine. |
| `conversation/session.py` | In-memory conversation turns, timestamps, and history projection. |
| `conversation/__init__.py` | Package marker. |
| `tenant/context.py` | Organization context plus allowed connection IDs; rejects database selection outside the current session boundary. |
| `tenant/__init__.py` | Package marker. |

### Core (`app/core/`)

| File | Responsibility |
| --- | --- |
| `config.py` | Loads environment-backed settings and application metadata. |
| `console.py` | Terminal encoding/format helpers and safe rendering of text. |
| `trace.py` | Configures the `db_sentinel` logger and file handler under the configured log directory. |

### Database access (`app/database/`)

| File | Responsibility |
| --- | --- |
| `connection.py` | Engine-neutral immutable `DatabaseConnection` configuration, including tenant/workspace, endpoint and options. |
| `manager.py` | Adapter/connection registry, organization ownership checks, list/remove/close lifecycle. |
| `discovery.py` | Shared schema-discovery behavior and database metadata traversal. |
| `metadata.py` | Metadata representation/utilities used to normalize discovered objects. |
| `schema_retriever.py` | Selects the most relevant live table/column metadata for a question and optional dictionary context. |
| `data_dictionary.py` | Parses supported dictionary formats and validates descriptions against discovered live objects. |
| `executor.py` | `SecureQueryExecutor`: runs only validated read-only adapter calls under timeout/row cap and writes contextual SQL audit events. |
| `test_connection.py` | Connection setup/testing regressions. |
| `test_data_dictionary.py` | Dictionary parsing and validation tests. |
| `test_executor_logging.py` | SQL and organization/workspace/connection audit-context tests. |
| `test_mongodb_adapter.py` | MongoDB adapter tests. |
| `test_multi_tenant.py` | Connection ownership/tenant isolation tests. |
| `test_relational_adapters.py` | Shared SQLAlchemy relational adapter tests. |
| `test_schema_retriever.py` | Schema selection, lexical matching and one-table fallback tests. |
| `database/__init__.py` | Package marker. |

### Adapter system (`app/database/adapters/`)

| File | Responsibility |
| --- | --- |
| `base.py` | Abstract adapter contract: connection lifecycle, execution, read-only execution, schema/table/column discovery. |
| `factory.py` | Normalized database-type-to-adapter registry. |
| `postgresql.py` | PostgreSQL adapter; includes Supabase-managed-schema filtering scoped to Supabase endpoint hosts. |
| `sqlalchemy_relational.py` | Shared SQLAlchemy adapter for supported MySQL and SQL Server connections. |
| `mongodb.py` | MongoDB connection, bounded sampling/discovery and read-only aggregation execution. |

To add an engine, implement `DatabaseAdapter`, register it in `DatabaseAdapterRegistry`/`DatabaseManager`, provide a dialect/security path, and test metadata discovery, read-only execution, bounds, and invalid query rejection. Do not enable a new engine merely by accepting a URL scheme.

### Model integration (`app/llm/`)

| File | Responsibility |
| --- | --- |
| `base.py` | Language-model provider protocol expected by the router. |
| `gemini.py` | Google Gemini provider adapter. |
| `groq.py` | Groq provider adapter. |
| `model_registry.py` | Model metadata, tiers, capability filters, priority/cost ordering. |
| `model_router.py` | Chooses eligible healthy providers/models, invokes structured output and records route metadata. |
| `complexity.py` | Classifies question complexity and derives model capability needs. |

Providers are interchangeable behind the provider protocol. Secrets belong in environment configuration, never in source or committed test fixtures. Provider availability, model capabilities, and generation quality are separate failure sources; check routing logs before debugging SQL.

### Data contracts, security and artifacts

| File | Responsibility |
| --- | --- |
| `models/database.py` | Database-related request/configuration data models. |
| `models/mongo.py` | MongoDB query/pipeline typed output contract. |
| `models/schema.py` | Column/table/schema and relationship metadata contracts. |
| `models/sql.py` | Typed generated-SQL result contract. |
| `models/__init__.py` | Model package exports. |
| `security/policies.py` | Natural-language operation detection; rejects writes, DDL, privileges, transactions, and expressly sensitive reads. |
| `security/permissions.py` | Second request-level read-only operation allow-list. |
| `security/sql_validator.py` | Dialect-aware SQL parser and guard for read-only statements, objects, fields, and query shape. |
| `security/mongodb_validator.py` | MongoDB aggregation allow-list, live-field checks, and rejection of unsafe/unbounded constructs. |
| `security/test_mongodb_validator.py` | MongoDB validator positive and negative tests. |
| `reporting/artifacts.py` | Typed dashboard/report structures assembled from the verified insight and evidence. |
| `reporting/__init__.py` | Package marker. |
| `security/__init__.py` | Package marker. |

## 6. Core function walkthrough

This section follows the main code contracts in execution order. It is intended as a debugging map, not a substitute for reading the named implementation and tests.

### Connection and CLI functions (`app/main.py`)

- `prompt_required(label, default)` handles required terminal values and defaults. `prompt_port(default)` additionally requires a positive integer.
- `_connection_from_url(...)` recognizes MongoDB URIs separately, normalizes common SQL URL aliases, parses SQLAlchemy URLs, extracts the engine/database/host/credentials/options, and derives SSL behavior. It returns a `DatabaseConnection`; it does not make a network connection.
- `_connection_url(connection)` serializes a configuration back into a URL for local persistence. Since that serialization contains the secret, do not print or log its result.
- `_saved_connection_profiles()` parses and validates the profile array from the environment. Malformed configuration raises a descriptive error rather than pretending there are no saved connections.
- `_write_saved_connection_profiles(...)` writes the serialized array into the configured `.env` and updates this process's environment.
- `_persist_connection(connection)` replaces the same organization/connection profile if present, then adds the successful connection. Other profiles remain intact.
- `_remove_persisted_connection(connection_id, organization_id)` removes only the matching tenant-scoped profile.
- `_restore_connection_profiles(manager, organization_id, profiles)` converts only matching organization entries into `DatabaseConnection` objects and registers them. It deliberately does not test a network connection at startup.
- `_register_and_test(manager, connection)` registers the adapter, calls its connection test, removes it on failure, redacts the password from user-visible errors, and persists only a successful profile.
- `create_database(...)` chooses URL or fields mode. Both wizards retain entered values after a failure and let the user select a field to edit before retrying.
- `_create_database_from_fields(...)` prompts engine connection values, creates the engine-neutral object, then loops on test/edit. A typo in a hostname/database/user can therefore be corrected without repeating the whole wizard.
- `delete_saved_connection(...)` verifies ownership, removes local profile data, closes/removes the adapter from `DatabaseManager`, and logs the profile deletion. The interactive caller asks for confirmation. It performs no remote data operation.
- `print_result(result)` renders answer status, insight, findings, recommendations, uncertainty, verification, and evidence in the CLI.
- `run()` initializes the CLI, reads the saved organization profiles, restores connections, handles menu commands, starts conversation sessions and closes manager connections at shutdown.

### Planning and orchestration (`app/agent/`)

- `QuestionNormalizer.normalize(question, conversation_history)` corrects wording without adding analysis or changing scope. It rejects rewrites that change numeric constraints, quoted filter values, or negation. The CLI displays a changed interpretation, while the original question remains authoritative.
- `QuestionPlanner.create_plan(question, conversation_history, authoritative_question)` uses corrected wording for query phrasing while preserving the original request's entities, filters, measures, and requested output. It applies deterministic normalizations, then verifies dependency IDs and ordering. It does not produce SQL.
- `_normalize_period_scoped_plan(question, plan)` only runs when the question explicitly requests a comparison/change, mentions a period, and asks about a measurable metric. It retains a required baseline and ensures distinct optional period-comparison and segment-analysis steps with dependencies. It adds no table or column names.
- `_sanitize_why_plan(question, plan)` removes unsupported causal assertions from model-generated supporting steps, ensures a required measurable baseline, and adds a schema-grounded dimension comparison. An explicitly requested breakdown is required; exploratory breakdowns are optional. It deduplicates and bounds steps and must remain domain-neutral.
- `_validate_and_normalize_dependencies(plan)` rejects empty plans, duplicate IDs, self/later/unknown dependencies and dependency cycles, while removing duplicate edges.
- `InvestigationEngine.investigate(...)` performs organization ownership and read-only checks on the original request, normalizes wording, rechecks the normalized request against read-only policy, classifies intent, routes metadata questions to the explorer, plans analytical questions, executes each step with bounded retries, passes prerequisite evidence to dependents, checks required-step completion, generates/verifies the final insight, and builds artifacts. `QueryRepairAnalyzer` classifies each failed attempt (security/schema, execution, result verification, pagination, logical evidence, or repeated query), preserves earlier constraints, and gives the query generator the exact failure and a focused repair direction. Every regenerated SQL or MongoDB query still passes the regular security and result verification gates; the retry budget remains bounded.
- `_failed_required_steps(...)` excludes failed optional steps from the required-answer failure set. Optional failures are appended as explicit uncertainty.
- Both query paths receive live schema selected for the current step from the active adapter. The retriever ranks tables/collections using requested information and available live column, key, and relationship metadata; query generation must use only the supplied identifiers. SQL generation additionally receives the active SQL dialect. Neither generator is configured with a particular database's table or field names.
- `SQLPipeline.generate_validate_and_execute(...)` retrieves schema if absent, creates or reuses a typed `AnalyticalQueryIntent` from the user's exact step and current live schema, rejects intent fields and sort dimensions absent from that schema, and requires ranking intents to order by a requested metric. It then gives the same intent to SQL/Mongo generation and logical verification. Retries reuse the intent so repair feedback can change query mechanics without changing the user's measure, filters, periods, dimensions, ranking direction, or result requirements. Every proposal still passes the existing live-object/sensitive-field checks, security validation, secure execution, and result verification. Supported implicit-extreme SQL requests are wrapped to count every tied result while keeping the evidence preview within 100 rows; unsupported SQL shapes fail closed.
- Complete relational collection requests are wrapped in one bounded SQL statement when the generated query has a supported deterministic ordering: the inner query retains its full distinct/grouped result, `COUNT(*) OVER()` records the total, and the outer query returns at most 100 ordered rows. The count column is removed before evidence verification and presentation. `collect_additional_pages(...)` remains the fallback for queries that cannot be safely wrapped and for MongoDB pipelines; it fetches bounded pages to completion while retaining at most 100 preview rows.
- `InsightGenerator.generate(...)` favors deterministic aggregate, all-extreme-ties, single-metric ranking, and contrastive multi-measure output; its shared Decimal parser handles native numeric values and MongoDB Extended JSON numeric wrappers without converting financial values through binary floating point. Contrastive high/low output uses disclosed median baselines and preserves both measure values for each listed entity. Otherwise it asks a model for structured insight, checks its numeric/causal grounding, retries with feedback, and falls back to a restrained answer if claims remain unsupported.

Date/time ranges constrain the requested records or measurement, and the typed intent carries the explicit periods and result shape through generation and verification. Explicit two-period comparison/change requests (including wording such as “amount change” or a request for percentage change) are planned as one required grouped evidence query that returns each requested period's raw metric in a separate row. The final insight computes direction, absolute difference, and (when requested and the baseline is nonzero) percentage change only from those verified values. A collapsed percentage-only result is rejected. Relative multi-year windows with a stated ending year are expanded to inclusive calendar years. An unanchored request such as “the last 2 years” queries the latest distinct calendar years available in the database, rather than assuming that the current calendar year has data. For entity-grouped temporal queries, the MongoDB generator retains the compound entity/year grouping, sorts by year first with a stable tie-break, and uses bounded pagination rather than limiting the whole result to the requested number of years. The deterministic insight layer then selects the latest requested distinct years from complete evidence and classifies monotonic per-entity increases or declines. A relative growth/change request for “last year” compares the two latest distinct years available; the logical verifier rejects metric-first year ordering or source-record limits before year grouping. For grouped maximum/minimum/ranking requests, the intent specifies the metric and direction; the MongoDB generator can recognize accumulator outputs and entity keys in the generated pipeline, repair the sort to use the requested aggregate metric and direction with a stable entity tie-break, then re-run normal security and logical verification. Deterministic summaries for implicit maximum/minimum questions report only entities tied at that extreme. Trend summaries use the returned annual values and end-to-end growth. The planner does not add preceding-period or segment analysis unless the user explicitly requests a comparison, trend, or change. This contract improves consistency but does not make unsupported, ambiguous, or data-absent questions answerable; those must remain explicit failures/limitations rather than invented results.

Questions asking why, for reasons, drivers, or contributing factors establish a required measured baseline and add a schema-grounded dimension comparison where possible. An explicitly requested breakdown is required; exploratory breakdowns are supplemental so their failure does not block a baseline answer. The insight generator synthesizes verified evidence across steps, reports observed associations as associations, and does not claim causation from segment differences alone. If evidence cannot support a specific explanation, it must say so rather than inventing one.
- `InsightVerifier.verify(...)` compares claims to evidence values/entity rows, checks ranking order or exact extreme-tie count and metric, and flags unsupported causality or limited completeness. A failed verification changes the final status to review-required.

### Database and model boundaries

- `DatabaseManager.register(...)` creates an adapter for a connection and associates it with its organization. `get(..., organization_id=...)` enforces ownership. `remove(...)` closes and removes exactly one adapter. `close_all()` is the CLI shutdown path.
- `DatabaseAdapter` defines the contract shared by all engines. Adapters are responsible for connection establishment, metadata retrieval, safe row shaping, read-only execution and resource cleanup.
- `SecureQueryExecutor.execute(sql)` invokes the supplied validator before the adapter. A rejection is audited but not run. Successful/failed adapter calls are also audited. The caller supplies audit context; the executor never guesses a tenant.
- `ModelRouter.generate(...)` classifies complexity when not specified, chooses capability-compatible models using registry priority/cost, health-checks providers, tries eligible candidates, returns structured output and records selected route metadata. Provider health and model output are not security controls.

## 7. Configuration, persistence and operational behavior

### Saved connections

The connection manager itself is in-memory. Persistence is a CLI feature implemented in `main.py`: successful connections serialize to `DB_SENTINEL_CONNECTIONS_JSON` in the local `.env`; startup parses and restores entries matching the chosen organization ID. This is why a profile can persist between runs without embedding any particular database in the Python code.

The implementation does not contact or verify the database at profile restoration time. A restored profile can therefore be listed even if the host is offline or credentials have expired. Query-time connection errors must be addressed by editing the profile/credentials. Deleting a profile removes it from local configuration and closes the adapter.

Runtime connection defaults are generic (`DB_TYPE`, `DB_HOST`, `DB_DATABASE_NAME`, `DB_USER`, and related variables); database datasets are not bundled into the product. Profiles for a particular database are local user configuration, not hardcoded database behavior.

### Logging

The log formatter adds timestamp, level, logger name, and event text. Query audit entries add organization/workspace/connection identifiers and the SQL text/hash. Result rows and URLs are intentionally excluded. SQL literals may still reveal sensitive filter values, so secure and rotate logs.

### Optional dictionary

Dictionary formats and validation are implemented but the current CLI intentionally does not prompt for loading one. The live schema remains available without a dictionary. Do not bypass metadata checks based on dictionary text.

## 8. Running, tests and debugging

### Normal run

Follow the install and run instructions in the repository README, then launch:

```powershell
python -m app.main
```

Use a read-only database account. Start with database exploration questions (table/column listing), then simple aggregates and filters, and finally multi-step comparisons. Examine status and evidence for every result; a plausible summary is not enough.

### Test commands

```powershell
python -m pytest -q
python -m compileall -q app
git diff --check
```

The test suite validates local code behavior; it does not automatically exercise a remote database, production network, every schema, every model, or every question wording.

### Diagnostic sequence

1. **Connection fails:** inspect scheme, hostname, DNS/network, port, database name, SSL mode, user and password. Use the wizard's field selection to edit only the incorrect value.
2. **No schema found:** confirm the selected connection and organization, then inspect schema discovery and `schema_retriever.py` ranking/fallback. An optional dictionary can help vocabulary matching but is not the source of schema truth.
3. **Generated SQL rejected:** read the attempt and validator reason in `logs/db-sentinel.log`. Determine whether it referenced a non-live field, non-allowed table, write operation, sensitive column, or unsupported dialect construct.
4. **Query executes but is not verified:** distinguish execution success from semantic/logical verification and pagination completeness. Check the original question constraints against SQL and evidence rows.
5. **Wrong metric selected:** review the live schema, dictionary descriptions if available, generated SQL and filters. Ambiguous words such as “cases,” “affected,” or “sales” may map to several legitimate measures; do not present one as uniquely intended without evidence or clarification.
6. **Insight is under review:** inspect `insight_verification.unsupported_findings` and `warnings`. Fix the evidence or claim mismatch; do not suppress the guard.
7. **LLM failure:** inspect provider health, available capabilities, route metadata and retry logs before blaming the database.
8. **Optional analysis fails:** the required answer can remain usable, but confirm the returned limitations disclose the missing comparison/dimension.

## 9. Known limitations and improvement path

- The CLI has no authentication/authorization identity for a human user, and its tenant/workspace identifiers are supplied locally.
- Connection credentials are plaintext in a local environment file; hosted deployment needs a secret manager/encryption and access controls.
- Audit logging is a local file prototype, not durable, tamper-evident, tenant-queryable audit storage.
- Database adapter coverage is not the same as live certification for every engine/dialect/version.
- Model output can still misunderstand ambiguous business language. Verification can detect many unsupported claims, but it cannot always infer the metric a person intended.
- A planner's optional prior-period or segment analysis may be unavailable due to schema, data coverage, missing periods, or unsupported joins. The required answer should be separated from those optional limitations.
- Read-only policy is intentional. If a future product adds mutations, it needs explicit human approval, dry-run/preview, transactional semantics, least-privilege credentials, rollback strategy and a separate threat model.
- Dictionary loading from the current CLI is paused; use the loader API/tests until a user-facing flow is added.

## 10. Explaining the product clearly

### Short technical explanation

“DB-Sentinel is a database-agnostic, evidence-first analysis assistant. A model interprets a question and proposes a plan and query, while live schema metadata, deterministic query guards, read-only bounded execution, and independent evidence checks control what may be run and what may be claimed. Each answer can be traced to its investigation steps, SQL, and returned evidence.”

### Honest investor/product framing

The product differentiator is not merely natural-language-to-SQL. It is the workflow around it: investigation planning, adapter-based database access, schema grounding, read-only query enforcement, explicit evidence collection, verification, and traceable reporting. The current code demonstrates those architectural building blocks in a local prototype. A credible investment pitch should describe the prototype as the current milestone and separately present a roadmap for authentication, hosted tenant isolation, encrypted credentials, durable audit/observability, user research, reliability evaluation and production deployment. Do not claim guaranteed correctness, universal engine support, or production SaaS readiness until those have been measured and delivered.

## 11. Change guide

- Changing a question-analysis behavior? Update planner/investigation/verifier tests, including ambiguity and failure cases.
- Adding a new SQL feature? Review `sql_generator.py`, `sql_pipeline.py`, both verifiers and `security/sql_validator.py`; a generated feature is not supported until safe validation and pagination are covered.
- Adding a database engine? Implement and register an adapter, dialect mapping, metadata retrieval, secure read-only execution and adapter-specific tests.
- Changing connection storage? Preserve multiple profiles and organizations, do not print URLs, update `DB_SENTINEL_CONNECTIONS_JSON` atomically where possible, and test restart/restore/delete.
- Changing insight/report output? Update the schema, grounding verifier, CLI renderer and `reporting/artifacts.py` together.
- Updating this guide? Run `python scripts\\build_architecture_docx.py` so the Word file stays in sync.
