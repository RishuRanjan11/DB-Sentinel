from dataclasses import dataclass
import hashlib
import json
from typing import Any

from app.database.adapters.base import DatabaseAdapter
from app.security.sql_validator import SQLSecurityValidator
from app.core.trace import logger


@dataclass(frozen=True)
class QueryExecutionResult:
    success: bool
    rows: list[dict[str, Any]]
    row_count: int
    error: str | None = None
    security_allowed: bool = False


class SecureQueryExecutor:

    DEFAULT_TIMEOUT_MS = 10_000
    MAX_ROWS = 100

    def __init__(
        self,
        adapter: DatabaseAdapter,
        validator: SQLSecurityValidator,
        statement_timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_rows: int = MAX_ROWS,
        audit_context: dict[str, str] | None = None,
    ):
        if statement_timeout_ms <= 0:
            raise ValueError(
                "statement_timeout_ms must be greater than zero."
            )

        if max_rows <= 0 or max_rows > self.MAX_ROWS:
            raise ValueError(
                f"max_rows must be between 1 and "
                f"{self.MAX_ROWS}."
            )

        self.adapter = adapter
        self.validator = validator
        self.statement_timeout_ms = statement_timeout_ms
        self.max_rows = max_rows
        self.audit_context = {
            key: value
            for key, value in (audit_context or {}).items()
            if value
        }

    def execute(
        self,
        sql: Any,
        parameters: dict[str, Any] | None = None,
    ) -> QueryExecutionResult:

        # -----------------------------------------------------
        # 1. Deterministic security validation
        # -----------------------------------------------------

        validation = self.validator.validate(sql)

        if not validation.allowed:
            self._audit(
                event="QUERY_REJECTED",
                sql=sql,
                reason=validation.reason,
                audit_context=self.audit_context,
            )

            return QueryExecutionResult(
                success=False,
                rows=[],
                row_count=0,
                error=validation.reason,
                security_allowed=False,
            )

        # -----------------------------------------------------
        # 2. Secure database execution
        # -----------------------------------------------------

        try:
            rows = self.adapter.execute_read_only(
                query=sql,
                parameters=parameters,
                statement_timeout_ms=self.statement_timeout_ms,
                max_rows=self.max_rows,
            )

            self._audit(
                event="QUERY_EXECUTED",
                sql=sql,
                reason="Query executed successfully.",
                audit_context=self.audit_context,
            )

            return QueryExecutionResult(
                success=True,
                rows=rows,
                row_count=len(rows),
                error=None,
                security_allowed=True,
            )

        except Exception as error:

            self._audit(
                event="QUERY_FAILED",
                sql=sql,
                reason=str(error),
                audit_context=self.audit_context,
            )

            return QueryExecutionResult(
                success=False,
                rows=[],
                row_count=0,
                error=str(error),
                security_allowed=True,
            )

    @staticmethod
    def _audit(
        event: str,
        sql: Any,
        reason: str,
        audit_context: dict[str, str] | None = None,
    ) -> None:

        # Prototype audit trail.
        # Persistent tenant-aware audit storage will be added
        # when the SaaS persistence layer is implemented.

        query_text = (
            sql
            if isinstance(sql, str)
            else json.dumps(sql, ensure_ascii=True, sort_keys=True)
        )
        query_text = " ".join(query_text.split())
        query_hash = hashlib.sha256(
            query_text.strip().encode("utf-8")
        ).hexdigest()[:16]
        logger.info(
            "AUDIT event=%s organization_id=%s workspace_id=%s "
            "connection_id=%s reason=%s query_sha256=%s query=%s",
            event,
            (audit_context or {}).get("organization_id", "unspecified"),
            (audit_context or {}).get("workspace_id", "unspecified"),
            (audit_context or {}).get("connection_id", "unspecified"),
            reason,
            query_hash,
            query_text,
        )