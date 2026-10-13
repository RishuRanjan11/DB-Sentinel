import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.llm.model_router import ModelRouter, create_default_model_router


MetricOperation = Literal[
    "raw",
    "count",
    "distinct_count",
    "sum",
    "average",
    "minimum",
    "maximum",
]
FilterOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "contains",
    "starts_with",
    "ends_with",
    "exists",
    "is_null",
    "before",
    "after",
]
SortDirection = Literal["asc", "desc"]
IntentKind = Literal[
    "lookup",
    "aggregate",
    "comparison",
    "trend",
    "ranking",
    "explanation",
]


class IntentMetricFormula(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: Literal["multiply", "add", "subtract", "divide"]
    fields: list[str] = Field(min_length=2, max_length=4)


class IntentMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: MetricOperation
    field: str | None = None
    formula: IntentMetricFormula | None = None
    alias: str

    @model_validator(mode="after")
    def validate_metric_expression(self):
        if self.formula is not None:
            if self.field is not None:
                raise ValueError(
                    "A metric must use either a source field or a formula, not both."
                )
            if self.operation not in {"sum", "average", "minimum", "maximum"}:
                raise ValueError(
                    "Metric formulas require a numeric aggregation operation."
                )
            if (
                self.formula.operator in {"subtract", "divide"}
                and len(self.formula.fields) != 2
            ):
                raise ValueError(
                    "Subtraction and division formulas require exactly two fields."
                )
        elif self.operation not in {"count"} and self.field is None:
            raise ValueError(
                "This metric operation requires a source field or formula."
            )
        return self


class IntentFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    operator: FilterOperator
    value: Any = None


class IntentSort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: SortDirection


class AnalyticalQueryIntent(BaseModel):
    """Schema-grounded, database-neutral contract for one analytical step."""

    model_config = ConfigDict(extra="forbid")

    kind: IntentKind
    metrics: list[IntentMetric] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[IntentFilter] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    sort: list[IntentSort] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, le=100)
    include_all_extreme_ties: bool = False
    requires_causal_evidence: bool = False
    result_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_query_shape(self):
        if self.kind in {"aggregate", "comparison", "trend", "ranking", "explanation"}:
            if not self.metrics:
                raise ValueError(
                    "Analytical intents must specify at least one metric."
                )
        if self.kind in {"comparison", "trend"} and not self.periods:
            raise ValueError(
                "Comparison and trend intents must specify requested periods."
            )
        if self.kind == "ranking":
            if not self.sort:
                raise ValueError(
                    "Ranking intents must specify a sort direction."
                )
            metric_fields = {
                value.lower()
                for metric in self.metrics
                for value in (metric.alias, metric.field)
                if value
            }
            if not any(
                item.field.lower() in metric_fields
                for item in self.sort
            ):
                raise ValueError(
                    "Ranking intents must sort by a requested metric."
                )
        return self


class QueryIntentPlanner:
    """Extract a typed analytical contract grounded in live schema fields."""

    RANKING_REQUEST = re.compile(
        r"\b(?:most|highest|largest|maximum|max|top|rank|ranking|"
        r"least|lowest|smallest|minimum|min)\b|"
        r"\b(?:sort|order)\s+(?:the\s+)?(?:results?\s+)?by\b",
        re.IGNORECASE,
    )

    def __init__(self, model_router: ModelRouter | None = None):
        self.model_router = model_router or create_default_model_router()

    def create_intent(
        self,
        question: str,
        required_information: list[str],
        schema_context: list[dict[str, Any]],
        feedback: str | None = None,
        attempt_number: int = 1,
        previous_intent: AnalyticalQueryIntent | dict[str, Any] | None = None,
    ) -> AnalyticalQueryIntent:
        schema = [
            {
                "schema": table.get("schema"),
                "table_or_collection": table.get("table"),
                "columns": table.get("columns", []),
                "primary_key": table.get("primary_key", []),
                "foreign_keys": table.get("foreign_keys", []),
            }
            for table in schema_context
        ]
        intent = self.model_router.generate(
            system_instruction="""
You translate one database investigation step into a domain-neutral
analytical query contract. You do not write SQL or MongoDB.

Use only fields and relationships in the supplied live schema. Represent
the requested operation, measures, grouping dimensions, literal filters,
periods, sort direction, result limit, and evidence requirements exactly.
Do not infer a cause from correlation. Mark requires_causal_evidence true
only when the user explicitly asks for a cause. Preserve ambiguity as a
result requirement instead of guessing. The query is read-only; never
represent a write operation.

Metric operations: raw (return field values), count (count rows/events),
distinct_count (count unique field entities), sum, average, minimum,
maximum. Use a null metric field only for row count. For ranking, sort by
the metric alias in the requested direction and then by a dimension as a
stable tie-break. An implicit maximum/minimum requests every entity tied at
the extreme. A Top-N request retains its requested limit.

For sales or revenue totals over line items, calculate each line's extended
amount as unit price multiplied by quantity, then sum those amounts. Use this
formula only when the live schema contains fields that represent both unit
price and quantity; choose their exact schema field paths and include both in
the metric formula. Never substitute a sum of unit prices for revenue.

Return only the structured intent. Do not invent schema fields, filters,
period boundaries, or values beyond the revenue calculation above.
""",
            user_prompt=f"""
Investigation step:
{question}

Required information:
{required_information}

Live schema:
{schema}

Repair feedback for this intent, if any:
{feedback or "None"}

Previous canonical intent to preserve unless the repair evidence shows
that a specific part is wrong:
{json.dumps(
    previous_intent.model_dump(mode="json")
    if isinstance(previous_intent, AnalyticalQueryIntent)
    else previous_intent or {},
    ensure_ascii=True,
    sort_keys=True,
)}

Produce the smallest complete query intent that captures the user's exact
request. Distinguish row count from summing a numeric measure. Keep requested
period values as explicit labels. Every result requirement must be
checkable against query output or verified evidence. On a retry, change only
the intent details implicated by the repair feedback; preserve every other
valid measure, entity, filter, period, direction, and output constraint.

When the step contrasts multiple measures, include every requested measure
for each requested dimension in the same result rows. Preserve the measure
used for the high/low ranking, and return all grouped rows needed for the
relative comparison unless the user explicitly requested a bounded Top-N.
""",
            response_model=AnalyticalQueryIntent,
            complexity="fast",
            classification_text=question,
            attempt_number=attempt_number,
        )
        if (
            intent.kind == "ranking"
            and not self.RANKING_REQUEST.search(question)
        ):
            intent_data = intent.model_dump(mode="json")
            intent_data["kind"] = "aggregate"
            intent_data["sort"] = []
            intent_data["limit"] = None
            intent = AnalyticalQueryIntent.model_validate(intent_data)
        return intent

    @staticmethod
    def invalid_schema_references(
        intent: AnalyticalQueryIntent,
        schema_context: list[dict[str, Any]],
    ) -> list[str]:
        known = {
            str(column.get("name", column.get("column", ""))).lower()
            for table in schema_context
            for column in table.get("columns", [])
            if isinstance(column, dict)
        }
        known.update(
            str(column).lower()
            for table in schema_context
            for column in table.get("columns", [])
            if isinstance(column, str)
        )
        references = [
            *(metric.field for metric in intent.metrics if metric.field),
            *(
                field
                for metric in intent.metrics
                if metric.formula is not None
                for field in metric.formula.fields
            ),
            *intent.dimensions,
            *(item.field for item in intent.filters),
        ]
        output_fields = {
            value.lower()
            for metric in intent.metrics
            for value in (metric.alias, metric.field)
            if value
        }
        references.extend(
            item.field
            for item in intent.sort
            if item.field.lower() not in known
            and item.field.lower() not in output_fields
        )
        invalid = []
        for reference in references:
            normalized = str(reference).lower()
            if normalized not in known:
                invalid.append(str(reference))
        return sorted(set(invalid))
