from dataclasses import dataclass


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str


class DatabasePermissions:
    """
    Database-level operation permissions.

    DB-Sentinel currently operates as a read-only analyst.
    """

    READ_ONLY_OPERATIONS = frozenset(
        {
            "SELECT",
            "READ",
            "SHOW",
            "DESCRIBE",
            "EXPLAIN",
            "COUNT",
            "COMPARE",
            "ANALYZE",
            "FIND",
            "LIST",
            "GET",
        }
    )

    def check(
        self,
        operation: str,
    ) -> PermissionDecision:

        normalized = (
            operation.strip().upper()
            if operation
            else ""
        )

        if normalized in self.READ_ONLY_OPERATIONS:
            return PermissionDecision(
                allowed=True,
                reason="Read-only operation permitted.",
            )

        return PermissionDecision(
            allowed=False,
            reason=(
                f"Operation '{normalized or 'UNKNOWN'}' "
                "is not permitted by the database "
                "permission policy."
            ),
        )