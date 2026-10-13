import re
from typing import Any


class QueryRepairAnalyzer:
    """Turns failed query checks into focused, bounded retry instructions."""

    @classmethod
    def create_feedback(
        cls,
        question: str,
        query_language: str,
        failure_stage: str,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
        previous_feedback: str | None = None,
    ) -> str:
        category, guidance = cls._diagnose(
            failure_stage,
            reason,
            diagnostics or {},
        )
        parts = []
        if previous_feedback:
            parts.append(
                "Earlier repair constraints that still apply:\n"
                f"{previous_feedback}"
            )
        parts.extend(
            [
                "Repair diagnosis:",
                f"- Query language: {query_language}.",
                f"- Failure stage: {failure_stage}.",
                f"- Failure category: {category}.",
                f"- Observed failure: {reason or 'No reason was provided.'}",
                f"- Recommended repair focus: {guidance}",
                f"- Original request to preserve: {question}",
                (
                    "Revise only the query step that failed. Keep the "
                    "requested measures, entities, filters, time periods, "
                    "and output shape. Use only the supplied live schema. "
                    "Do not weaken security, fabricate values, or change "
                    "the question to make a query pass. Revalidate the new "
                    "query before execution."
                ),
            ]
        )
        for name, value in (diagnostics or {}).items():
            if value not in (None, "", [], {}):
                parts.append(f"- {name}: {value}")
        return "\n".join(parts)

    @staticmethod
    def _diagnose(
        failure_stage: str,
        reason: str,
        diagnostics: dict[str, Any],
    ) -> tuple[str, str]:
        text = f"{failure_stage} {reason}".lower()
        if re.search(r"repeat|identical to a previous", text):
            return (
                "repeated query",
                "change the failed query's actual construction, not just "
                "whitespace or aliases",
            )
        if failure_stage == "security/schema validation":
            return (
                "security or schema validation",
                "remove the prohibited construct or replace unknown objects "
                "with exact allowed live-schema identifiers; never bypass "
                "the validator",
            )
        if failure_stage == "query execution":
            if re.search(
                r"syntax|parse|unknown (?:column|field|operator)|"
                r"does not exist|ambiguous|type mismatch|cannot cast",
                text,
            ):
                return (
                    "query syntax, identifier, or type error",
                    "correct the failing expression using the reported "
                    "database error and live schema; retain the requested "
                    "logic",
                )
            if re.search(
                r"timeout|timed out|connection|network|unavailable|"
                r"temporar|deadlock",
                text,
            ):
                return (
                    "database or transient execution failure",
                    "determine whether the error identifies an expensive "
                    "query construct; do not alter the requested semantics "
                    "to mask a connection or transient service problem",
                )
            return (
                "database execution failure",
                "use the exact execution error to isolate the invalid or "
                "unsupported construct, without inventing schema fields",
            )
        if failure_stage == "result verification":
            return (
                "result-shape or completeness verification",
                "satisfy the failed deterministic checks and return the "
                "complete requested evidence within the enforced bounds",
            )
        if failure_stage == "pagination":
            return (
                "incomplete or unstable pagination",
                "use deterministic ordering on the complete unique result "
                "key while preserving the requested filters and grouping",
            )
        if failure_stage == "logical verification":
            return (
                "evidence does not satisfy the question",
                "return the explicitly missing measures, entities, periods, "
                "or breakdown in a query result shape that can be verified",
            )
        return (
            "query generation or investigation failure",
            "inspect the reported cause, preserve verified constraints, and "
            "make a specific correction only when the query caused the failure",
        )
