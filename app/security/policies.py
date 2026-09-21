from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    operation: str


class RequestPolicy:
    """
    Deterministic request-level security policy.

    The LLM must never reinterpret a prohibited write/DDL request
    into a safe SELECT request.
    """

    READ_OPERATIONS = {
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

    WRITE_OPERATIONS = {
        "INSERT",
        "UPDATE",
        "DELETE",
        "MERGE",
        "UPSERT",
    }

    DDL_OPERATIONS = {
        "CREATE",
        "ALTER",
        "DROP",
        "TRUNCATE",
        "RENAME",
    }

    PRIVILEGE_OPERATIONS = {
        "GRANT",
        "REVOKE",
    }

    TRANSACTION_OPERATIONS = {
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
    }


    SENSITIVE_REQUEST_PATTERNS = (
        r"\bprivate\b",
        r"\bsensitive\b",
        r"\bpersonal information\b",
        r"\bpersonally identifiable\b",
        r"\bpii\b",
        r"\bpasswords?\b",
        r"\bcredentials?\b",
        r"\bsecrets?\b",
        r"\bapi keys?\b",
        r"\baccess tokens?\b",
        r"\bprivate keys?\b",
        r"\bssn\b",
        r"\bsocial security\b",
    )

    def evaluate(self, question: str) -> PolicyDecision:
        if not question or not question.strip():
            return PolicyDecision(
                allowed=False,
                reason="Empty investigation request.",
                operation="UNKNOWN",
            )

        text = question.strip().lower()

        if any(
            re.search(pattern, text)
            for pattern in self.SENSITIVE_REQUEST_PATTERNS
        ):
            return PolicyDecision(
                allowed=False,
                reason=(
                    "Requests for private or sensitive database "
                    "information are blocked by the data-access policy."
                ),
                operation="SENSITIVE_READ",
            )

        operation = self._detect_operation(text)

        if operation in self.WRITE_OPERATIONS:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Operation '{operation}' is a destructive "
                    "write operation. DB-Sentinel is configured "
                    "for read-only database investigations."
                ),
                operation=operation,
            )

        if operation in self.DDL_OPERATIONS:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Operation '{operation}' is a schema-changing "
                    "DDL operation and is not permitted."
                ),
                operation=operation,
            )

        if operation in self.PRIVILEGE_OPERATIONS:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Operation '{operation}' changes database "
                    "privileges and is not permitted."
                ),
                operation=operation,
            )

        if operation in self.TRANSACTION_OPERATIONS:
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"Transaction operation '{operation}' is "
                    "not permitted."
                ),
                operation=operation,
            )

        return PolicyDecision(
            allowed=True,
            reason="Request is compatible with read-only investigation.",
            operation=operation,
        )

    def _detect_operation(self, text: str) -> str:
        patterns = [
            (
                "DELETE",
                r"\b(delete|remove|erase|destroy)\b",
            ),
            (
                "INSERT",
                r"\b(insert|add\s+(?:a\s+)?(?:new\s+)?record|create\s+(?:a\s+)?record)\b",
            ),
            (
                "UPDATE",
                r"\b(update|modify|change)\s+(?:the\s+)?(?:table|column|schema|database|record|row|data)\b|\bset\s+[A-Za-z_][A-Za-z0-9_.]*\s*="
            ),
            (
                "ALTER",
                r"\b(alter|modify)\b.*\b(table|column|schema|database)\b",
            ),
            (
                "DROP",
                r"\b(drop)\b.*\b(table|column|schema|database|index)\b",
            ),
            (
                "TRUNCATE",
                r"\b(truncate)\b",
            ),
            (
                "RENAME",
                r"\b(rename)\b.*\b(table|column|schema|database)\b",
            ),
            (
                "CREATE",
                r"\b(create)\b.*\b(table|schema|database|index|view)\b",
            ),
            (
                "GRANT",
                r"\b(grant)\b",
            ),
            (
                "REVOKE",
                r"\b(revoke)\b",
            ),
            (
                "COMMIT",
                r"\b(commit)\b",
            ),
            (
                "ROLLBACK",
                r"\b(rollback)\b",
            ),
            (
                "SAVEPOINT",
                r"\b(savepoint)\b",
            ),
        ]

        for operation, pattern in patterns:
            if re.search(pattern, text):
                return operation

        return "READ"