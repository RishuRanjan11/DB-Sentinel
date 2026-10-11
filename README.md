# DB-Sentinel

DB-Sentinel is an autonomous, conversational data-analyst platform. It is designed as a SaaS system in which an organization can connect one or many independent databases and ask continuous analytical questions. The initial product has one subscription tier and the same capabilities for every customer.

## Product boundary

The system is not a text-to-SQL chatbot. SQL is an internal analytical tool. The product flow is:

`question → intent → investigation → evidence → verified insight → dashboard/report → follow-up`

The analyst should dynamically understand the connected database rather than contain customer-specific schemas, table names, columns, entities, or test-question assumptions.

## Tenant model

One organization can own any number of database connections:

`Organization A → DB 1, DB 2, DB 3, ... DB N`

Another organization may own a different number of databases. Database connections, credentials, schema metadata, conversations, evidence, reports, and dashboards must remain organization-scoped. Cross-organization database access is rejected by the database manager when an organization context is supplied.

## Pluggability

Database access is adapter-based. PostgreSQL is the current implemented adapter; the adapter registry is the extension point for MySQL, MongoDB, BigQuery, Snowflake, and other engines.

LLM access is already provider/model registry based. Additional providers can be attached without embedding provider-specific routing throughout the investigation engine.

## Current analytical layers

- investigation planning
- dynamic schema retrieval
- deterministic SQL security
- read-only execution
- semantic verification
- logical verification
- evidence collection
- grounded insight generation and verification
- database exploration
- optional, connection-scoped user data dictionaries
- conversational session state
- dashboard/report artifact specifications

## Security principle

The LLM proposes. Deterministic infrastructure decides. The LLM is never the tenant boundary or the security boundary.

## Optional data dictionary

Users may provide a UTF-8 JSON data dictionary for a connected database. It is
optional: DB-Sentinel continues to retrieve live schema metadata and can answer
questions without one. When supplied, descriptions help match business terms
to real tables and columns and provide context for explanations. The loader
checks every documented schema, table, and column against the connected
database. A dictionary cannot add database objects or bypass SQL validation.

In the CLI, select a database and enter `/dictionary`, then provide the JSON
file path. Enter `/dictionary clear` to remove the reference for that
connection. Alternatively, set `DB_SENTINEL_DATA_DICTIONARY` to a JSON file
path before starting the app. A dictionary is held only in the current process
and is associated with the selected connection.

Supported format:

```json
{
  "version": 1,
  "description": "Optional overview of the database.",
  "schemas": [
    {
      "name": "public",
      "description": "Optional explanation of this schema.",
      "tables": [
        {
          "name": "gene",
          "description": "One row represents a curated gene.",
          "columns": [
            {
              "name": "symbol",
              "description": "The approved gene symbol."
            }
          ]
        }
      ]
    }
  ]
}
```

Descriptions are user-provided and unverified. They are treated as semantic
hints, not as evidence or instructions. Final numerical and factual claims
must still be supported by live query results and the normal verification
pipeline; when evidence is insufficient, DB-Sentinel should report that
limitation rather than claim certainty.

## Investigation logs

Investigation logs are written to `logs/db-sentinel.log` by default. The log
records investigation identifiers, intent classification, plan/step/attempt
transitions, schema selection counts, SQL fingerprints, security and execution
outcomes, verifier decisions, evidence counts, and final status. It does not
write result rows or database credentials. Set `DB_SENTINEL_LOG_DIR` to choose
another log directory.
