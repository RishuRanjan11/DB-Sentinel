from dataclasses import dataclass
from typing import Any

from app.database.adapters.base import DatabaseAdapter


@dataclass(frozen=True)
class VerificationResult:
    verified: bool
    checks: list[str]
    warnings: list[str]


class ResultVerifier:

    def verify(
        self,
        adapter: DatabaseAdapter,
        sql: str,
        rows: list[dict[str, Any]],
    ) -> VerificationResult:

        checks = []
        warnings = []

        if not rows:
            checks.append(
                "Query executed successfully but returned no rows."
            )

            return VerificationResult(
                verified=True,
                checks=checks,
                warnings=warnings,
            )

        checks.append(
            "Query execution succeeded."
        )

        checks.append(
            f"Returned {len(rows)} rows."
        )

        column_names = set()

        for row in rows:
            column_names.update(
                str(key)
                for key in row.keys()
            )

        if column_names:
            checks.append(
                "Result rows have consistent column metadata."
            )
        else:
            warnings.append(
                "Result rows contain no columns."
            )

        # --------------------------------------------------
        # Safe duplicate detection.
        #
        # Database values can contain lists, dictionaries,
        # JSONB objects, arrays, etc. They cannot be placed
        # directly inside a set.
        # --------------------------------------------------

        canonical_rows = {
            self._canonicalize(row)
            for row in rows
        }

        duplicate_rows = (
            len(rows) != len(canonical_rows)
        )

        if duplicate_rows:
            # Duplicate rows are evidence about the result shape, not
            # automatically a correctness failure. Whether duplicates
            # change the requested meaning depends on the SQL objective
            # (for example, relationship membership can legitimately
            # produce repeated entity rows before a later DISTINCT or
            # aggregation). Semantic verification owns that decision.
            warnings.append(
                "Duplicate result rows detected; semantic verification "
                "must determine whether duplication changes the requested meaning."
            )
        else:
            checks.append(
                "No duplicate result rows detected."
            )

        # Structural checks above are deterministic. Duplicate detection
        # remains visible as a warning but must not turn an otherwise
        # executable result into a hard failure by itself.
        return VerificationResult(
            verified=True,
            checks=checks,
            warnings=warnings,
        )

    @classmethod
    def _canonicalize(cls, value: Any):
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    sorted(
                        (
                            str(key),
                            cls._canonicalize(item),
                        )
                        for key, item in value.items()
                    )
                ),
            )

        if isinstance(value, (list, tuple)):
            return (
                "sequence",
                tuple(
                    cls._canonicalize(item)
                    for item in value
                ),
            )

        if isinstance(value, set):
            return (
                "set",
                tuple(
                    sorted(
                        cls._canonicalize(item)
                        for item in value
                    )
                ),
            )

        try:
            hash(value)
            return (
                "value",
                value,
            )
        except TypeError:
            return (
                "repr",
                repr(value),
            )