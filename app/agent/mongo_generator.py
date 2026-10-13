from copy import deepcopy
import json
import re

from app.core.trace import logger
from app.llm.model_router import ModelRouter, create_default_model_router
from app.models.mongo import GeneratedMongoQuery, GeneratedMongoStage
from app.agent.question_normalizer import QuestionNormalizer
from app.agent.query_intent import AnalyticalQueryIntent


class MetricFormulaEnforcementError(ValueError):
    """Raised when a generated MongoDB query cannot honor its metric contract."""


class MongoQueryGenerator:
    def __init__(self, model_router: ModelRouter | None = None):
        self.model_router = model_router or create_default_model_router()

    def generate(
        self,
        question: str,
        schema_context: list[dict],
        complexity: str | None = None,
        feedback: str | None = None,
        evidence_context: list[dict] | None = None,
        previous_queries: list[str] | None = None,
        attempt_number: int = 1,
        query_intent: AnalyticalQueryIntent | dict | None = None,
    ) -> GeneratedMongoQuery:
        available_schema = [
            {
                "schema": table.get("schema"),
                "collection": table.get("table"),
                "fields": table.get("columns", []),
                "primary_key": table.get("primary_key", ["_id"]),
            }
            for table in schema_context
        ]
        system_instruction = """
You generate one read-only MongoDB aggregation query for DB-Sentinel.
Use only the database and collections and fields in the supplied live
schema. Return a collection name and an aggregation pipeline.

Security and correctness requirements:
- Never generate insert, update, delete, replace, or administrative operations.
- Use only the supplied collection names; use $lookup only for a supplied
  collection in the same database.
- Never use $out, $merge, $where, $function, $accumulator, $unionWith,
  $graphLookup, or unrecognized stages.
- Always end the top-level pipeline with a numeric $limit from 1 to 100.
- Represent each pipeline stage as an object with exactly these two fields:
  "operator" (one permitted stage name, including its leading $) and
  "specification_json" (a JSON-encoded string containing that stage's
  complete argument). For example, a limit stage is represented as
  {"operator":"$limit","specification_json":"10"}.
  Do not return aggregation stages as arbitrary-key JSON objects.
- For ranking, include $sort on the requested metric and use a stable
  secondary sort on the entity key.
- When the canonical metric intent includes a formula, calculate that exact
  expression for each source record before applying its aggregation.
- For an unanchored relative-year comparison, sort primarily by grouped
  calendar year descending; this takes precedence over metric ranking.
- Use the exact field names in the supplied schema. Do not invent fields.
- When counting unique entities, use $addToSet or an equivalent distinct
  aggregation rather than counting duplicated joins.
- Treat user-provided dictionary descriptions as unverified semantic hints.
- Return no explanation, only the requested structured output.
"""
        user_prompt = f"""
Question:
{question}

Live schema:
{available_schema}

Verified prerequisite evidence:
{evidence_context or []}

Previous query attempts:
{previous_queries or []}

Previous validation/execution feedback:
{feedback or "None"}

Canonical analytical query intent:
{json.dumps(
    query_intent.model_dump(mode="json")
    if isinstance(query_intent, AnalyticalQueryIntent)
    else query_intent or {},
    ensure_ascii=True,
    sort_keys=True,
)}

This is attempt {attempt_number}. Return one MongoDB aggregation query that
preserves the user's exact intent and uses only the live schema.
"""
        duration = QuestionNormalizer.relative_comparison_year_count(question)
        if (
            duration
            and QuestionNormalizer.relative_calendar_years(question) is None
        ):
            user_prompt += f"""
Relative-year requirement:
The user requested the latest {duration} years without specifying an ending
year. Interpret this as the latest {duration} distinct calendar years present
in the stored data, not a rolling date interval relative to today's date.
Do not use $$NOW, $dateSubtract, or a $match date-range filter. Use the
supplied live schema to identify an existing year field or date field, group
the requested raw metric by calendar year, and sort grouped years newest
first. If the question requests an entity/dimension breakdown, group by that
entity and calendar year in exactly one $group stage. Do not add another
$group to discover the latest years. Preserve one row per entity/year, sort
by year descending and _id as the stable tie-breaker, and use a final $limit
of 100 so the grouped rows can be paginated completely; the verified
analysis selects the latest {duration} distinct years from the complete
rows. Otherwise return one raw metric row per year with a final $limit of
{duration}. Use only field paths explicitly shown in the live schema; do not
assume unlisted nested array paths or fields. Do not calculate growth inside the
pipeline.
"""
        generated = self.model_router.generate(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=GeneratedMongoQuery,
            complexity=complexity,
            classification_text=question,
            attempt_number=attempt_number,
        )
        duration = QuestionNormalizer.relative_comparison_year_count(
            question
        )
        if (
            duration
            and QuestionNormalizer.relative_calendar_years(question) is None
        ):
            generated = self._enforce_latest_year_order(
                generated,
                duration,
            )
        generated = self._enforce_metric_formulas(
            generated,
            query_intent=query_intent,
        )
        generated = self._enforce_group_ranking(
            generated,
            question,
            query_intent=query_intent,
        )
        return generated

    @classmethod
    def _enforce_metric_formulas(
        cls,
        query: GeneratedMongoQuery,
        query_intent: AnalyticalQueryIntent | dict | None = None,
    ) -> GeneratedMongoQuery:
        intent_data = (
            query_intent.model_dump(mode="json")
            if isinstance(query_intent, AnalyticalQueryIntent)
            else query_intent or {}
        )
        metrics = intent_data.get("metrics", [])
        formula_metrics = [
            metric
            for metric in metrics
            if isinstance(metric, dict)
            and isinstance(metric.get("formula"), dict)
            and metric.get("operation") in {
                "sum",
                "average",
                "minimum",
                "maximum",
            }
        ] if isinstance(metrics, list) else []
        if not formula_metrics:
            return query

        pipeline = [
            {stage.operator: json.loads(stage.specification_json)}
            for stage in query.pipeline
        ]
        group_indexes = [
            index for index, stage in enumerate(pipeline) if "$group" in stage
        ]
        if not group_indexes:
            raise MetricFormulaEnforcementError(
                "The query intent requires a metric formula, but the generated "
                "MongoDB pipeline does not contain a $group stage."
            )

        changed = False
        accumulator_operators = {
            "sum": "$sum",
            "average": "$avg",
            "minimum": "$min",
            "maximum": "$max",
        }
        for metric in formula_metrics:
            alias = str(metric.get("alias", ""))
            target_group = None
            target = None
            for group_index in group_indexes:
                candidate_group = pipeline[group_index]["$group"]
                aggregate_fields = [
                    str(name)
                    for name, expression in candidate_group.items()
                    if name != "_id" and cls._contains_aggregate(expression)
                ]
                candidate_target = next(
                    (
                        field
                        for field in aggregate_fields
                        if field.lower() == alias.lower()
                    ),
                    None,
                )
                if (
                    candidate_target is None
                    and len(formula_metrics) == 1
                    and len(aggregate_fields) == 1
                    and group_index == group_indexes[0]
                ):
                    candidate_target = aggregate_fields[0]
                if candidate_target is not None:
                    target_group = candidate_group
                    target = candidate_target
                    break
            formula = metric["formula"]
            operator = formula.get("operator")
            fields = formula.get("fields")
            accumulator = accumulator_operators.get(metric["operation"])
            if (
                target is None
                or accumulator is None
                or operator not in {"multiply", "add", "subtract", "divide"}
                or not isinstance(fields, list)
                or not 2 <= len(fields) <= 4
                or not all(isinstance(field, str) and field for field in fields)
            ):
                raise MetricFormulaEnforcementError(
                    f"The generated MongoDB query cannot map the formula for "
                    f"metric '{alias}' to a unique aggregate field."
                )
            expected = {
                accumulator: {
                    f"${operator}": [f"${field}" for field in fields]
                }
            }
            if target_group.get(target) != expected:
                target_group[target] = expected
                changed = True

        if not changed:
            return query
        logger.info(
            "MongoDB metric optimizer enforced canonical metric formula(s)"
        )
        return GeneratedMongoQuery(
            collection=query.collection,
            pipeline=[
                GeneratedMongoStage(
                    operator=operator,
                    specification_json=json.dumps(
                        specification,
                        ensure_ascii=True,
                    ),
                )
                for stage in pipeline
                for operator, specification in stage.items()
            ],
            fields_used=query.fields_used,
        )

    @classmethod
    def _enforce_group_ranking(
        cls,
        query: GeneratedMongoQuery,
        question: str,
        query_intent: AnalyticalQueryIntent | dict | None = None,
    ) -> GeneratedMongoQuery:
        if not re.search(
            r"\b(?:most|highest|largest|maximum|max|top|rank|ranking|"
            r"least|lowest|smallest|minimum|min)\b",
            question,
            re.IGNORECASE,
        ):
            return query

        pipeline = [
            {stage.operator: json.loads(stage.specification_json)}
            for stage in query.pipeline
        ]
        group_indexes = [
            index for index, stage in enumerate(pipeline) if "$group" in stage
        ]
        if len(group_indexes) != 1:
            return query

        group_index = group_indexes[0]
        group = pipeline[group_index]["$group"]
        metric_fields = [
            str(name)
            for name, expression in group.items()
            if name != "_id" and cls._contains_aggregate(expression)
        ]
        if not metric_fields:
            return query

        intent_data = (
            query_intent.model_dump(mode="json")
            if isinstance(query_intent, AnalyticalQueryIntent)
            else query_intent or {}
        )
        intent_metrics = intent_data.get("metrics", [])
        intent_sorts = intent_data.get("sort", [])
        intent_metric_names = {
            str(value).lower()
            for metric in intent_metrics
            if isinstance(metric, dict)
            for value in (metric.get("alias"), metric.get("field"))
            if value
        } if isinstance(intent_metrics, list) else set()
        sort_intent = next(
            (
                item
                for item in intent_sorts
                if isinstance(item, dict)
                and item.get("field")
                and item.get("direction") in {"asc", "desc"}
                and (
                    not intent_metric_names
                    or str(item.get("field", "")).lower()
                    in intent_metric_names
                )
            ),
            None,
        )
        metric_field = None
        if sort_intent:
            metric_field = next(
                (
                    field
                    for field in metric_fields
                    if any(
                        isinstance(metric, dict)
                        and metric.get("alias", "").lower() == field.lower()
                        and metric.get("alias", "").lower()
                        == str(sort_intent["field"]).lower()
                        for metric in intent_metrics
                    )
                ),
                None,
            )
            if (
                metric_field is None
                and len(metric_fields) == 1
                and len(intent_metrics) == 1
            ):
                metric_field = metric_fields[0]
        if metric_field is None:
            metric_field = cls._requested_metric_field(
                metric_fields,
                group,
                question,
            )
        if metric_field is None:
            return query

        trailing_stages = pipeline[group_index + 1:]
        if any(
            next(iter(stage), None) not in {
                "$addFields",
                "$project",
                "$set",
                "$sort",
                "$limit",
                "$unset",
            }
            for stage in trailing_stages
        ):
            return query

        sort_indexes = [
            index
            for index in range(group_index + 1, len(pipeline))
            if "$sort" in pipeline[index]
        ]
        if len(sort_indexes) != 1:
            return query

        projected_metric = cls._projected_alias(
            f"${metric_field}",
            trailing_stages,
        )
        entity_field = cls._projected_alias(
            "$_id",
            trailing_stages,
        )
        if projected_metric is None or entity_field is None:
            return query

        sort_index = sort_indexes[0]
        sort_spec = pipeline[sort_index]["$sort"]
        if not isinstance(sort_spec, dict):
            return query

        if sort_intent and len(metric_fields) == 1:
            descending = sort_intent["direction"] == "desc"
        else:
            direction_text = cls._primary_request_text(question)
            descending = not bool(
                re.search(
                    r"\b(?:least|lowest|smallest|minimum|min)\b",
                    direction_text,
                    re.IGNORECASE,
                )
            )
        expected_sort = {
            projected_metric: -1 if descending else 1,
            entity_field: 1,
        }
        if list(sort_spec.items()) == list(expected_sort.items()):
            return query

        repaired_pipeline = list(pipeline)
        repaired_pipeline[sort_index] = {"$sort": expected_sort}
        logger.info(
            "MongoDB ranking optimizer enforced metric ordering "
            "metric=%s direction=%s entity=%s",
            projected_metric,
            "descending" if descending else "ascending",
            entity_field,
        )
        return GeneratedMongoQuery(
            collection=query.collection,
            pipeline=[
                GeneratedMongoStage(
                    operator=operator,
                    specification_json=json.dumps(
                        specification,
                        ensure_ascii=True,
                    ),
                )
                for stage in repaired_pipeline
                for operator, specification in stage.items()
            ],
            fields_used=query.fields_used,
        )

    @staticmethod
    def _contains_aggregate(value) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).lower()
                in {
                    "$sum",
                    "$avg",
                    "$min",
                    "$max",
                    "$first",
                    "$last",
                    "$count",
                    "$stddevpop",
                    "$stddevsamp",
                }
                or MongoQueryGenerator._contains_aggregate(nested)
                for key, nested in value.items()
            )
        if isinstance(value, list):
            return any(
                MongoQueryGenerator._contains_aggregate(item)
                for item in value
            )
        return False

    @classmethod
    def _requested_metric_field(
        cls,
        metric_fields: list[str],
        group: dict,
        question: str,
    ) -> str | None:
        if len(metric_fields) == 1:
            return metric_fields[0]

        question_tokens = set(
            re.findall(r"[a-z0-9]+", question.lower())
        )
        aliases = {
            "count": {"count", "number", "quantity"},
            "sum": {"sum", "total", "sales", "revenue", "amount"},
            "average": {"average", "avg", "mean"},
        }
        ranked = []
        for field in metric_fields:
            tokens = set(
                re.findall(
                    r"[a-z]+|[A-Z][a-z0-9]*",
                    field,
                )
            )
            normalized_tokens = {token.lower() for token in tokens}
            expression = group[field]
            operators = cls._aggregate_operators(expression)
            for operator in operators:
                normalized_tokens.update(aliases.get(operator, set()))
            score = len(normalized_tokens & question_tokens)
            ranked.append((score, field))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if not ranked or ranked[0][0] == 0:
            return None
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            return None
        return ranked[0][1]

    @classmethod
    def _aggregate_operators(cls, value) -> set[str]:
        if isinstance(value, dict):
            return {
                key[1:].lower()
                for key, nested in value.items()
                if key.lower() in {
                    "$sum",
                    "$avg",
                    "$min",
                    "$max",
                    "$first",
                    "$last",
                    "$count",
                    "$stddevpop",
                    "$stddevsamp",
                }
            } | {
                operator
                for nested in value.values()
                for operator in cls._aggregate_operators(nested)
            }
        if isinstance(value, list):
            return {
                operator
                for nested in value
                for operator in cls._aggregate_operators(nested)
            }
        return set()

    @staticmethod
    def _projected_alias(field_reference: str, stages: list[dict]) -> str | None:
        current_reference = field_reference
        for stage in stages:
            projection = stage.get("$project")
            if isinstance(projection, dict):
                alias = next(
                    (
                        str(name)
                        for name, value in projection.items()
                        if value == current_reference
                    ),
                    None,
                )
                if alias is not None:
                    current_reference = f"${alias}"
                elif (
                    current_reference.startswith("$")
                    and current_reference[1:] in projection
                    and projection[current_reference[1:]] == 1
                ):
                    pass
                elif projection.get("_id", 1) == 0:
                    return None
            for operator in ("$addFields", "$set"):
                assignments = stage.get(operator)
                if not isinstance(assignments, dict):
                    continue
                alias = next(
                    (
                        str(name)
                        for name, value in assignments.items()
                        if value == current_reference
                    ),
                    None,
                )
                if alias is not None:
                    current_reference = f"${alias}"
        return current_reference[1:] if current_reference.startswith("$") else None

    @staticmethod
    def _primary_request_text(question: str) -> str:
        return next(
            (
                line.strip()
                for line in str(question or "").splitlines()
                if line.strip()
            ),
            "",
        )

    @classmethod
    def _enforce_latest_year_order(
        cls,
        query: GeneratedMongoQuery,
        year_count: int,
    ) -> GeneratedMongoQuery:
        pipeline = [
            {stage.operator: json.loads(stage.specification_json)}
            for stage in query.pipeline
        ]
        group_indexes = [
            index
            for index, stage in enumerate(pipeline)
            if "$group" in stage
        ]
        if len(group_indexes) != 1:
            return query

        group_index = group_indexes[0]
        if any("$limit" in stage for stage in pipeline[:group_index]):
            return query
        group_key = pipeline[group_index]["$group"].get("_id")
        group_alias = cls._year_group_alias(group_key)
        if group_alias is None:
            return query
        composite_year_group = (
            isinstance(group_key, dict)
            and len(group_key) > 1
            and group_alias.startswith("_id.")
        )

        trailing_stages = pipeline[group_index + 1:]
        if any(
            next(iter(stage), None) not in {
                "$addFields",
                "$project",
                "$set",
                "$sort",
                "$limit",
            }
            for stage in trailing_stages
        ):
            return query

        output_alias = cls._projected_year_alias(
            group_alias,
            trailing_stages,
        )
        sort_field = output_alias or group_alias
        repaired_pipeline = deepcopy(pipeline[:group_index + 1]) + [
            deepcopy(stage)
            for stage in trailing_stages
            if "$sort" not in stage and "$limit" not in stage
        ]
        if composite_year_group:
            for stage in repaired_pipeline[group_index + 1:]:
                projection = stage.get("$project")
                if (
                    not isinstance(projection, dict)
                    or projection.get("_id", 1) not in (0, False)
                ):
                    continue
                has_inclusions = any(
                    value not in (0, False)
                    for key, value in projection.items()
                    if key != "_id"
                )
                if has_inclusions:
                    projection["_id"] = 1
                else:
                    projection.pop("_id")
        sort_spec = {sort_field: -1}
        if composite_year_group:
            sort_spec["_id"] = 1
        repaired_pipeline.append({"$sort": sort_spec})
        repaired_pipeline.append(
            {"$limit": 100 if composite_year_group else year_count}
        )
        if repaired_pipeline != pipeline:
            logger.info(
                "MongoDB relative-period optimizer enforced newest-year "
                "ordering field=%s count=%s grouped_by_entity=%s",
                sort_field,
                100 if composite_year_group else year_count,
                composite_year_group,
            )
            return GeneratedMongoQuery(
                collection=query.collection,
                pipeline=[
                    GeneratedMongoStage(
                        operator=operator,
                        specification_json=json.dumps(
                            specification,
                            ensure_ascii=True,
                        ),
                    )
                    for stage in repaired_pipeline
                    for operator, specification in stage.items()
                ],
                fields_used=query.fields_used,
            )
        return query

    @classmethod
    def _year_group_alias(cls, group_key) -> str | None:
        if isinstance(group_key, dict):
            if "$year" in group_key:
                return "_id"
            year_keys = [
                str(key)
                for key, expression in group_key.items()
                if re.search(r"year", str(key), re.IGNORECASE)
                or cls._contains_year_expression(expression)
            ]
            if len(year_keys) == 1:
                return f"_id.{year_keys[0]}"
            if year_keys:
                return None
            if (
                len(group_key) == 1
                and cls._contains_year_expression(group_key)
            ):
                return "_id"
            return None
        if cls._contains_year_expression(group_key):
            return "_id"
        if isinstance(group_key, str) and re.search(
            r"(?:year|date|time)",
            group_key,
            re.IGNORECASE,
        ):
            return "_id"
        return None

    @classmethod
    def _contains_year_expression(cls, value) -> bool:
        if isinstance(value, dict):
            return "$year" in value or any(
                cls._contains_year_expression(nested)
                for nested in value.values()
            )
        if isinstance(value, list):
            return any(cls._contains_year_expression(nested) for nested in value)
        return False

    @classmethod
    def _projected_year_alias(
        cls,
        group_alias: str,
        stages: list[dict],
    ) -> str | None:
        current_alias = group_alias
        for stage in stages:
            projection = stage.get("$project")
            if not isinstance(projection, dict):
                continue
            matching = [
                name
                for name, expression in projection.items()
                if expression == f"${current_alias}"
            ]
            if matching:
                current_alias = str(matching[0])
        return current_alias if current_alias != group_alias else None

    @staticmethod
    def serialize(query: GeneratedMongoQuery) -> str:
        return json.dumps(
            {
                "collection": query.collection,
                "pipeline": [
                    {
                        stage.operator: json.loads(
                            stage.specification_json
                        )
                    }
                    for stage in query.pipeline
                ],
            },
            ensure_ascii=True,
        )
