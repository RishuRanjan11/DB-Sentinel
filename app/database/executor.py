from dataclasses import dataclass
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

    def execute(
        self,
        sql: str,
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
        sql: str,
        reason: str,
    ) -> None:

        # Prototype audit trail.
        # Persistent tenant-aware audit storage will be added
        # when the SaaS persistence layer is implemented.

        logger.info(
            "AUDIT event=%s reason=%s sql=%s",
            event,
            reason,
            sql.strip(),
        )