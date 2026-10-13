import json
from collections.abc import Mapping
from typing import Any

from app.security.sql_validator import SQLValidationResult


class MongoQuerySecurityValidator:
    MAX_LIMIT = 100
    ALLOWED_STAGES = {
        "$addFields",
        "$bucket",
        "$bucketAuto",
        "$count",
        "$group",
        "$limit",
        "$lookup",
        "$match",
        "$project",
        "$replaceRoot",
        "$replaceWith",
        "$set",
        "$skip",
        "$sort",
        "$sortByCount",
        "$unset",
        "$unwind",
    }
    BLOCKED_FIELD_TOKENS = (
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "private_key",
        "access_key",
        "credential",
        "ssn",
        "social_security",
        "credit_card",
        "card_number",
        "security_answer",
        "private_note",
    )
    BLOCKED_OPERATORS = {
        "$accumulator",
        "$currentop",
        "$function",
        "$getfield",
        "$graphlookup",
        "$listlocalsessions",
        "$listsessions",
        "$out",
        "$merge",
        "$setfield",
        "$unionwith",
        "$where",
    }

    def __init__(
        self,
        allowed_collections: set[tuple[str, str]],
        schema_context: list[dict[str, Any]],
    ):
        self.allowed_collections = {
            (str(database).lower(), str(collection).lower())
            for database, collection in allowed_collections
        }
        self.schema_context = schema_context
        self.collection_fields = {
            str(table.get("table", "")).lower(): {
                str(
                    column.get("name", column.get("column", ""))
                    if isinstance(column, Mapping)
                    else column
                ).lower()
                for column in table.get("columns", [])
                if (
                    column.get("name", column.get("column", ""))
                    if isinstance(column, Mapping)
                    else column
                )
            }
            for table in schema_context
        }

    def validate(self, query: Any):
        try:
            payload = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError as error:
            return SQLValidationResult(
                False,
                f"MongoDB query JSON is invalid: {error}",
            )
        if not isinstance(payload, dict):
            return SQLValidationResult(False, "MongoDB query must be an object.")
        collection = payload.get("collection")
        pipeline = payload.get("pipeline")
        if not isinstance(collection, str) or not collection:
            return SQLValidationResult(
                False,
                "MongoDB query must specify one collection.",
            )
        if not isinstance(pipeline, list) or not pipeline:
            return SQLValidationResult(
                False,
                "MongoDB query must contain an aggregation pipeline.",
            )
        allowed = any(
            table.lower() == collection.lower()
            for _, table in self.allowed_collections
        )
        if not allowed:
            return SQLValidationResult(
                False,
                f"Collection '{collection}' is not permitted.",
            )

        limit_stages = []
        error = self._validate_stages(pipeline, limit_stages, top_level=True)
        if error:
            return SQLValidationResult(False, error)
        blocked_operator = self._blocked_operator(payload)
        if blocked_operator:
            return SQLValidationResult(
                False,
                f"MongoDB operator '{blocked_operator}' is not permitted.",
            )
        if not limit_stages:
            return SQLValidationResult(
                False,
                "A top-level $limit is required.",
            )
        if len(limit_stages) != 1 or limit_stages[0] != pipeline[-1]:
            return SQLValidationResult(
                False,
                "Exactly one final top-level $limit is required.",
            )
        limit = limit_stages[0]["$limit"]
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > self.MAX_LIMIT
        ):
            return SQLValidationResult(
                False,
                f"$limit must be an integer between 1 and {self.MAX_LIMIT}.",
            )

        field_error = self._validate_pipeline_fields(
            collection.lower(),
            pipeline,
        )
        if field_error:
            return SQLValidationResult(False, field_error)

        sensitive = self._sensitive_reference(payload)
        if sensitive:
            return SQLValidationResult(
                False,
                f"Sensitive field '{sensitive}' is not permitted.",
            )
        return SQLValidationResult(
            True,
            "MongoDB aggregation passed read-only security validation.",
        )

    def _validate_pipeline_fields(
        self,
        collection: str,
        pipeline: list,
        known_fields: set[str] | None = None,
    ) -> str | None:
        available = set(
            known_fields
            if known_fields is not None
            else self.collection_fields.get(collection, set())
        )
        if not available:
            return (
                f"No live field metadata is available for collection "
                f"'{collection}'."
            )
        available.add("_id")

        for stage in pipeline:
            operator, specification = next(iter(stage.items()))
            if operator == "$lookup":
                error = self._validate_lookup_fields(
                    specification,
                    available,
                )
                if error:
                    return error
                foreign_collection = str(
                    specification.get("from", "")
                ).lower()
                nested = specification.get("pipeline")
                if isinstance(nested, list):
                    foreign_fields = self.collection_fields.get(
                        foreign_collection,
                        set(),
                    )
                    error = self._validate_pipeline_fields(
                        foreign_collection,
                        nested,
                        known_fields=foreign_fields,
                    )
                    if error:
                        return error
                alias = specification.get("as")
                if isinstance(alias, str) and alias:
                    available.add(alias.split(".", 1)[0].lower())
                continue

            error = self._validate_stage_fields(
                operator,
                specification,
                available,
            )
            if error:
                return error

            if operator == "$project" and isinstance(specification, Mapping):
                excluded = {
                    str(name).split(".", 1)[0].lower()
                    for name, include in specification.items()
                    if include in (0, False)
                }
                is_exclusion = all(
                    include in (0, False)
                    for name, include in specification.items()
                    if str(name).lower() != "_id"
                )
                if is_exclusion:
                    available.difference_update(excluded)
                else:
                    available = {
                        str(name).split(".", 1)[0].lower()
                        for name, include in specification.items()
                        if include not in (0, False)
                    }
                if specification.get("_id", 1) in (0, False):
                    available.discard("_id")
                else:
                    available.add("_id")
            elif operator == "$group" and isinstance(specification, Mapping):
                available = {
                    str(name).lower()
                    for name in specification
                }
            elif operator in {"$addFields", "$set"} and isinstance(
                specification,
                Mapping,
            ):
                available.update(
                    str(name).split(".", 1)[0].lower()
                    for name in specification
                )
            elif operator == "$count" and isinstance(specification, str):
                available = {specification.lower()}
            elif operator == "$unset":
                fields = (
                    specification
                    if isinstance(specification, list)
                    else [specification]
                )
                for field in fields:
                    if isinstance(field, str):
                        available.discard(
                            field.split(".", 1)[0].lower()
                        )

        return None

    def _validate_lookup_fields(
        self,
        specification: Mapping,
        local_fields: set[str],
    ) -> str | None:
        foreign_collection = str(
            specification.get("from", "")
        ).lower()
        foreign_fields = self.collection_fields.get(
            foreign_collection,
            set(),
        )
        for key, fields in (
            ("localField", local_fields),
            ("foreignField", foreign_fields),
        ):
            field = specification.get(key)
            if field is not None:
                error = self._require_field(field, fields)
                if error:
                    return f"$lookup {key}: {error}"
        let_values = specification.get("let", {})
        if isinstance(let_values, Mapping):
            for expression in let_values.values():
                error = self._validate_expression_fields(
                    expression,
                    local_fields,
                )
                if error:
                    return f"$lookup let: {error}"
        return None

    def _validate_stage_fields(
        self,
        operator: str,
        specification: Any,
        available: set[str],
    ) -> str | None:
        if operator == "$match":
            return self._validate_match_fields(specification, available)
        if operator == "$sort":
            if not isinstance(specification, Mapping):
                return "$sort must be an object."
            for field in specification:
                error = self._require_field(field, available)
                if error:
                    return error
            return None
        if operator == "$project" and isinstance(specification, Mapping):
            for field, expression in specification.items():
                if expression in (1, True):
                    error = self._require_field(field, available)
                else:
                    error = self._validate_expression_fields(
                        expression,
                        available,
                    )
                if error:
                    return error
            return None
        if operator == "$unset":
            fields = (
                specification
                if isinstance(specification, list)
                else [specification]
            )
            for field in fields:
                error = self._require_field(field, available)
                if error:
                    return error
            return None
        if operator == "$unwind":
            path = (
                specification.get("path")
                if isinstance(specification, Mapping)
                else specification
            )
            return self._require_field(path, available)
        if operator == "$sortByCount":
            return self._validate_expression_fields(
                specification,
                available,
            )
        if operator == "$bucket" and isinstance(specification, Mapping):
            error = self._validate_expression_fields(
                specification.get("groupBy"),
                available,
            )
            if error:
                return error
        return self._validate_expression_fields(
            specification,
            available,
        )

    def _validate_match_fields(
        self,
        specification: Any,
        available: set[str],
    ) -> str | None:
        if isinstance(specification, list):
            for item in specification:
                error = self._validate_match_fields(item, available)
                if error:
                    return error
            return None
        if not isinstance(specification, Mapping):
            return None
        for key, value in specification.items():
            if key == "$expr":
                error = self._validate_expression_fields(value, available)
            elif str(key).startswith("$"):
                error = self._validate_match_fields(value, available)
            else:
                error = self._require_field(key, available)
            if error:
                return error
        return None

    def _validate_expression_fields(
        self,
        value: Any,
        available: set[str],
    ) -> str | None:
        if isinstance(value, Mapping):
            if "$literal" in value or "$const" in value:
                return None
            for child in value.values():
                error = self._validate_expression_fields(child, available)
                if error:
                    return error
        elif isinstance(value, list):
            for child in value:
                error = self._validate_expression_fields(child, available)
                if error:
                    return error
        elif isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
            return self._require_field(value[1:], available)
        return None

    @staticmethod
    def _require_field(
        field: Any,
        available: set[str],
    ) -> str | None:
        if not isinstance(field, str):
            return f"Field reference '{field}' is invalid."
        normalized = field.lstrip("$").lower()
        root = normalized.split(".", 1)[0]
        if not root or root not in available:
            return (
                f"Field '{field}' is not present in the live collection "
                "schema or an earlier pipeline stage."
            )
        return None

    def _validate_stages(
        self,
        pipeline: list,
        top_level_limits: list,
        top_level: bool,
    ) -> str | None:
        for stage in pipeline:
            if not isinstance(stage, dict) or len(stage) != 1:
                return "Every aggregation stage must have exactly one operator."
            operator, specification = next(iter(stage.items()))
            if operator not in self.ALLOWED_STAGES:
                return f"Aggregation stage '{operator}' is not permitted."
            if operator == "$limit":
                if (
                    isinstance(specification, bool)
                    or not isinstance(specification, int)
                    or specification < 1
                    or specification > self.MAX_LIMIT
                ):
                    return (
                        f"$limit must be an integer between 1 and "
                        f"{self.MAX_LIMIT}."
                    )
                if top_level:
                    top_level_limits.append(stage)
            if operator == "$lookup":
                error = self._validate_lookup(specification)
                if error:
                    return error
            nested = self._nested_pipeline(specification)
            if nested:
                error = self._validate_stages(
                    nested,
                    [],
                    top_level=False,
                )
                if error:
                    return error
        return None

    def _validate_lookup(self, specification: Any) -> str | None:
        if not isinstance(specification, dict):
            return "$lookup must be an object."
        foreign_collection = specification.get("from")
        if not isinstance(foreign_collection, str):
            return "$lookup must use a known collection name."
        if not any(
            table.lower() == foreign_collection.lower()
            for _, table in self.allowed_collections
        ):
            return (
                f"$lookup collection '{foreign_collection}' "
                "is not permitted."
            )
        if "db" in specification:
            return "Cross-database $lookup is not permitted."
        return None

    @staticmethod
    def _nested_pipeline(value: Any) -> list | None:
        if isinstance(value, dict):
            pipeline = value.get("pipeline")
            if isinstance(pipeline, list):
                return pipeline
        return None

    def _sensitive_reference(self, value: Any) -> str | None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower()
                if key_text.startswith("$"):
                    token = key_text[1:]
                    if token in self.BLOCKED_FIELD_TOKENS:
                        return token
                else:
                    field = key_text.split(".", 1)[0]
                    if any(token in field for token in self.BLOCKED_FIELD_TOKENS):
                        return field
                result = self._sensitive_reference(child)
                if result:
                    return result
        elif isinstance(value, list):
            for item in value:
                result = self._sensitive_reference(item)
                if result:
                    return result
        elif isinstance(value, str) and value.startswith("$"):
            if value.upper() in {"$$ROOT", "$$CURRENT"}:
                return "whole-document reference"
            field = value.lstrip("$").split(".", 1)[0].lower()
            if any(token in field for token in self.BLOCKED_FIELD_TOKENS):
                return field
        return None

    def _blocked_operator(self, value: Any) -> str | None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = str(key).lower()
                if normalized_key in self.BLOCKED_OPERATORS:
                    return normalized_key
                result = self._blocked_operator(child)
                if result:
                    return result
        elif isinstance(value, list):
            for item in value:
                result = self._blocked_operator(item)
                if result:
                    return result
        return None
