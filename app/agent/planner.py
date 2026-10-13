import re
from typing import List

from pydantic import BaseModel, Field, ConfigDict

from app.llm.model_router import (
    ModelRouter,
    create_default_model_router,
)
from app.agent.question_normalizer import QuestionNormalizer


class SubQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int

    question: str

    objective: str

    required_information: List[str] = Field(
        default_factory=list
    )

    depends_on: List[int] = Field(
        default_factory=list
    )

    optional: bool = False


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_question: str

    goal: str

    sub_questions: List[SubQuestion]


class QuestionPlanner:

    def __init__(
        self,
        model_router: ModelRouter | None = None,
    ):
        self.model_router = (
            model_router
            or create_default_model_router()
        )

    @staticmethod
    def _direct_ranking_plan(
        question: str,
        original_question: str,
    ) -> InvestigationPlan:
        asks_minimum = bool(
            re.search(
                r"\b(?:least|lowest|smallest|minimum|min)\b",
                original_question,
                re.IGNORECASE,
            )
        )
        extreme = "minimum" if asks_minimum else "maximum"
        direction = (
            "lowest to highest"
            if asks_minimum
            else "highest to lowest"
        )
        top_n = QuestionPlanner._extract_top_n(original_question.lower())
        contrast_match = re.search(
            r"\b(?:but|while|whereas|yet|although|despite)\b",
            original_question,
            re.IGNORECASE,
        )
        contrastive = False
        if contrast_match:
            clauses = (
                original_question[:contrast_match.start()],
                original_question[contrast_match.end():],
            )
            high_pattern = (
                r"\b(?:most|highest|largest|maximum|max|more|greater|high|higher)\b"
            )
            low_pattern = (
                r"\b(?:low|lower|lowest|least|smallest|minimum|min|less|fewer)\b"
            )
            first, second = clauses
            contrastive = bool(
                re.search(high_pattern, first, re.I)
                and re.search(low_pattern, second, re.I)
                or re.search(low_pattern, first, re.I)
                and re.search(high_pattern, second, re.I)
            )
        if contrastive:
            objective = (
                "Directly answer the original multi-measure contrast in one "
                "database operation. For each requested entity, return every "
                "measure explicitly requested in the original question, "
                "keeping each measure associated with the same entity. "
                "Preserve the requested high/low relationship; do not reduce "
                "the request to a ranking by only one measure. Return all "
                "entities needed to compare the measures, and do not impose "
                "an arbitrary result limit. Use the median across the "
                "returned entities as the relative benchmark when the user "
                "has not specified another benchmark. Rank using the measure "
                "the question describes as high."
            )
        else:
            objective = (
                "Directly answer the original ranking request in one database "
                "operation. Rank the requested entities from "
                f"{direction} by the metric requested by the user and return "
                "that metric value for each ranked entity."
            )
        if top_n is not None:
            objective += f" Return exactly the top {top_n} results."
        elif not contrastive and re.search(
            r"\b(?:most|highest|largest|maximum|max|least|lowest|smallest|minimum|min)\b",
            original_question,
            re.IGNORECASE,
        ):
            objective += (
                f" Return all entities tied at the requested {extreme}; "
                "do not select an arbitrary single entity when multiple "
                "entities share the extreme value."
            )

        return InvestigationPlan(
            original_question=original_question,
            goal=original_question,
            sub_questions=[
                SubQuestion(
                    id=1,
                    question=question,
                    objective=objective,
                    required_information=[
                        original_question,
                        (
                            "entity names and every measure required to "
                            "evaluate the requested relative comparison"
                            if contrastive
                            else "entity names and the requested ranking metric"
                        ),
                    ],
                    depends_on=[],
                )
            ],
        )

    def create_plan(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        authoritative_question: str | None = None,
    ) -> InvestigationPlan:

        question = question.strip()
        original_question = (authoritative_question or question).strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if self._is_simple_count_lookup(original_question):
            plan = InvestigationPlan(
                original_question=original_question,
                goal=original_question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=question,
                        objective=(
                            "Return the requested count for the user's "
                            "stated filters as a single aggregate result. "
                            "Do not add comparisons or segment analyses "
                            "unless the user asks for them."
                        ),
                        required_information=[original_question],
                        depends_on=[],
                    )
                ],
            )
            if self._asks_why(original_question):
                self._sanitize_why_plan(
                    original_question,
                    plan,
                )
            self._validate_and_normalize_dependencies(plan)
            return plan

        relative_years = QuestionNormalizer.relative_calendar_years(
            original_question
        )
        if (
            relative_years
            and self._asks_for_period_comparison(original_question)
        ):
            years = relative_years
            plan = InvestigationPlan(
                original_question=original_question,
                goal=original_question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=question,
                        objective=(
                            "Return the requested metric as one raw value "
                            "per calendar year for the inclusive requested "
                            f"range {years[0]} through {years[-1]}. Include "
                            "each year label and its value in separate "
                            "result rows. Do not collapse the yearly rows "
                            "into a calculated summary; derive growth or "
                            "change from the verified annual values."
                        ),
                        required_information=[
                            "The raw requested metric for every calendar "
                            "year in the inclusive requested range",
                            "A separate year label and metric value per row",
                            *(f"Year {year}" for year in years),
                        ],
                        depends_on=[],
                    )
                ],
            )
            if self._asks_why(original_question):
                self._sanitize_why_plan(
                    original_question,
                    plan,
                )
            self._validate_and_normalize_dependencies(plan)
            return plan

        relative_duration = (
            QuestionNormalizer.relative_comparison_year_count(
                original_question
            )
        )
        if (
            relative_duration
            and not relative_years
            and self._asks_for_period_comparison(original_question)
        ):
            plan = InvestigationPlan(
                original_question=original_question,
                goal=original_question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=question,
                        objective=(
                            "Return the requested metric as one raw value "
                            f"for each of the {relative_duration} most recent "
                            "distinct calendar years present in the database. "
                            "Determine the years from the data, group by "
                            "calendar year, sort the year values descending, "
                            f"and return the latest {relative_duration} annual "
                            "rows. Do not use the current system date or "
                            "assume a specific year range; do not filter to "
                            "years that are not present in the database. "
                            "If the original question requests an entity or "
                            "dimension breakdown, group by that dimension and "
                            "calendar year and return one row per "
                            "entity/year; do not limit the result to the "
                            "number of years, since that would omit entities. "
                            "Otherwise return one year label and its raw "
                            "metric value per row. Preserve all groups needed "
                            "to derive the requested trend from verified "
                            "values."
                        ),
                        required_information=[
                            original_question,
                            (
                                f"Raw requested metric values for the "
                                f"{relative_duration} latest distinct years "
                                "available in the database"
                            ),
                            "A separate calendar-year label and raw metric "
                            "value for each returned year",
                        ],
                        depends_on=[],
                    )
                ],
            )
            if self._asks_why(original_question):
                self._sanitize_why_plan(
                    original_question,
                    plan,
                )
            self._validate_and_normalize_dependencies(plan)
            return plan

        if self._has_explicit_year_comparison(original_question):
            years = self._explicit_years(original_question)
            plan = InvestigationPlan(
                original_question=original_question,
                goal=original_question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=question,
                        objective=(
                            "Return the requested metric grouped by each "
                            f"explicitly requested year ({years[0]} and "
                            f"{years[1]}) so the values can be compared. "
                            "Return one result row per year containing the "
                            "year and its raw requested metric value. Do not "
                            "collapse the yearly rows into a single percentage, "
                            "difference, or other calculated summary; derive "
                            "those from the verified yearly values afterward. "
                            "If a year has no matching records, preserve "
                            "that as missing-period evidence; do not replace "
                            "the requested years with the latest available "
                            "years."
                        ),
                        required_information=[
                            "One raw requested-metric value and year label "
                            "for each requested period",
                            "A separate result row for each requested year",
                            f"Year {years[0]}",
                            f"Year {years[1]}",
                        ],
                        depends_on=[],
                    )
                ],
            )
            if self._asks_why(original_question):
                self._sanitize_why_plan(
                    original_question,
                    plan,
                )
            self._validate_and_normalize_dependencies(plan)
            return plan

        if self._is_direct_ranking_lookup(original_question):
            plan = self._direct_ranking_plan(
                question=question,
                original_question=original_question,
            )
            self._validate_and_normalize_dependencies(plan)
            return plan

        system_instruction = """
You are the investigation planner for DB-Sentinel.

Your task is to decompose a user's database-analysis
question into concrete data-analysis steps.

IMPORTANT:

You are creating an INVESTIGATION PLAN, not SQL.

Never ask a step to discover database schema metadata.

Do NOT create steps such as:

- "Identify the tables..."
- "Find the table containing..."
- "Identify columns..."
- "Inspect information_schema..."
- "Determine which tables contain..."

DB-Sentinel has a dedicated schema-retrieval system that handles
database metadata automatically.

Every step must instead describe DATA that must be retrieved
or ANALYSIS that must be performed.

============================================================
CRITICAL: DO NOT INVENT ANALYTICAL DIMENSIONS
============================================================

The planner does NOT know the database schema.

Therefore you MUST NOT assume that the database contains:

- functional categories
- entity classifications
- related-record categories
- quality or status scores
- measured values
- disease categories
- timestamps
- geographic dimensions
- demographic dimensions
- severity levels
- performance metrics
- any other explanatory variable

unless the USER explicitly mentioned that dimension.

For a "why" question, do NOT invent a possible explanation.

Instead, identify measurable evidence that can be used to
compare the entities involved.

The later schema-retrieval and query-generation layers determine
which requested measurements and relationships can actually be answered.

If the database cannot provide evidence for a proposed
explanation, DB-Sentinel should report that limitation instead
of inventing an explanation.

============================================================
ORIGINAL USER INTENT IS AUTHORITATIVE
============================================================

Every sub-question MUST preserve all constraints that materially
affect the final answer.

Never silently remove or weaken requirements.

Preserve:

1. RANKING

If the original question contains:

- most
- highest
- largest
- maximum
- top
- rank
- ranking

the relevant sub-question MUST explicitly preserve ranking.

2. LOWEST / ASCENDING RANKING

If the original asks for:

- least
- lowest
- smallest
- minimum

preserve the requirement to identify the lowest values.

3. TOP-N

If the original explicitly asks:

- top 5
- top 10
- top N

preserve the exact N.

4. COMPARISON

If the original asks to:

- compare
- compare across
- difference
- versus
- vs

preserve the comparison.

5. TREND / CHANGE

If the original asks about:

- increase
- decrease
- change
- trend
- growth
- decline

preserve the relevant comparison or time dimension.

6. WHY / EXPLANATION

If the original asks "why":

- establish the relevant measurable baseline;
- compare relevant measurable dimensions;
- identify evidence-backed differences;
- do NOT invent explanatory variables;
- do NOT claim causation.

A "why" investigation should establish observable differences
before attempting interpretation.

7. CORRELATION / RELATIONSHIP

If the original asks about:

- correlation
- relationship
- association
- linked factors

retrieve evidence relevant to measuring that relationship.

8. CAUSATION

Do not convert:

correlation → causation
association → causation
difference → cause

unless the investigation actually contains evidence supporting
causation.

============================================================
SUB-QUESTION DESIGN
============================================================

Rules:

1. Preserve the user's actual intent.

2. Simple lookup questions normally produce one step.

3. Complex analytical questions may be decomposed into multiple
   meaningful steps.

4. Every sub-question must contribute evidence toward the
   ORIGINAL question.

5. Every sub-question must retain important ranking,
   comparison, filtering, aggregation, or analytical constraints.

6. Ranking steps must explicitly state ranking metric and direction.

7. For "why" questions, establish measurable baseline evidence
   first.

8. For comparison questions, retrieve the comparison dimensions
   explicitly requested or directly implied by the question.

9. For decline/increase questions, identify measurable contributors
   only when those contributors are represented or requested.

10. Do not invent explanatory variables.

11. required_information must describe business concepts,
    metrics, entities, relationships, dimensions, filters,
    or measurements.

12. Never invent database table names or column names.

13. Do not generate SQL.

14. Avoid redundant steps.

15. Prefer 1-6 steps.

16. Dependencies must represent genuine analytical dependencies.

17. Do not create a generic intermediate step that loses the
    final analytical requirement.

18. Treat dates, years, and ranges as filters when they constrain the
     requested records or metric. Do not add trend, prior-period, or
     segment-comparison work unless the user explicitly asks for that
     analysis.

19. Do not assume that a value decreased or that any factor caused a
     change. First compare verified measurements. Describe contributors
     as associations unless the evidence supports causation.

20. Choose useful dimensions from the user's wording and discovered
     schema. Do not assume specific columns or invent database fields.
     If no suitable dimension is available, report that limitation.

21. When the user explicitly asks for period or segment comparisons,
     use only dimensions confirmed by live schema metadata. Treat
     measured differences as associations, not causes.

============================================================
WHY-QUESTION DESIGN
============================================================

For questions beginning with "why", use this strategy:

Step 1:
Establish the baseline metric being discussed.

Step 2:
Compare the relevant entities using directly measurable
dimensions already implied by the question.

Step 3, if useful:
Measure coverage, ratios, counts, overlap, or other direct
relationships that could explain the observed difference.

Do NOT create a step that requires an unspecified scientific,
business, or domain explanation.

Use only dimensions stated in the question or justified by verified
schema metadata. Never invent an entity type, category, measure, or
causal mechanism.

============================================================
IMPORTANT RANKING EXAMPLE
============================================================

For a request such as "Which groups contain the most related records,
and how many are in each?", preserve all three requirements:
identify the requested groups, count the requested related entities per
group, and rank groups by that count.

Valid decomposition:

Step 1:
"Identify the entities and relationships required by the user."

Step 2:
"Retrieve the requested group-to-entity relationships."

Step 3:
"Count the requested entities in each group and rank the groups from
highest to lowest count."

The final analytical requirement must never be reduced to:

"Count records by group."

============================================================
OUTPUT
============================================================

Return structured output only.
"""

        history = conversation_history or []
        history_text = "\n\n".join(
            f"Turn {index}:\nQuestion: {item.get('question', '')}\nResult summary: {item.get('result', {})}"
            for index, item in enumerate(history[-5:], start=1)
        )

        user_prompt = f"""
Create an investigation plan for the following ORIGINAL
USER QUESTION:

{original_question}

Clear, grammar-corrected interpretation for query planning:
{question}

CONVERSATION CONTEXT:
{history_text or 'No previous conversation context.'}

Use previous conversation context only to resolve references such as
'it', 'that region', 'the previous result', or similar follow-up
references. The current user question remains authoritative.

The original question is authoritative.

When a ranking request contrasts measures (for example, one measure is high
while another is low), preserve both measures and their relationship. Do not
reduce it to a ranking on only one measure; request the values needed to make
the comparison explicit. Keep all requested measures in one step when one
grouped result can answer the comparison.

Before generating the plan, identify internally:

- requested entities
- requested metrics
- filters
- aggregations
- ranking requirements
- ranking direction
- Top-N requirements
- comparison requirements
- relationship requirements
- trend/change requirements
- explanatory/why requirements
- any other constraints that affect the final answer

IMPORTANT:

The database schema will be discovered automatically later.

Therefore:

- describe WHAT DATA is needed;
- do not describe WHERE the data is stored;
- do not invent table names;
- do not invent column names;
- do not invent explanatory dimensions;
- do not generate SQL.

For "why" questions, use measurable evidence directly related
to the user's question. Do not invent biological, business,
scientific, or technical explanations.

Every step must contribute evidence toward answering the
original question.

Treat time ranges as filters unless the user explicitly requests
a comparison, trend, or change analysis. Do not add supplemental
period or dimension analyses merely because the question includes
a date or year.
"""

        plan = self.model_router.generate(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=InvestigationPlan,
            complexity="fast",
            classification_text=original_question,
        )

        plan.original_question = original_question
        plan.goal = original_question
        self._remove_unrequested_period_context(original_question, plan)
        self._validate_and_normalize_dependencies(
            plan
        )

        # ==================================================
        # DIRECT LOOKUP NORMALIZATION
        # ==================================================
        # ==================================================
        # DIRECT LOOKUP NORMALIZATION
        # ==================================================

        if (
            self._is_direct_lookup(original_question)
            and not self._has_explicit_time_period(original_question)
        ):

            plan = InvestigationPlan(
                original_question=original_question,
                goal=original_question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=question,
                        objective=(
                            "Directly retrieve the data required "
                            "to answer the original user question."
                        ),
                        required_information=[
                            original_question
                        ],
                        depends_on=[],
                    )
                ],
            )

        # ==================================================
        # CONDITIONAL VERIFICATION REQUEST NORMALIZATION
        # ==================================================
        #
        # Questions such as:
        #
        #   "Find X. If the first result is insufficient or
        #    inconsistent, investigate further, optimize the
        #    query, rerun it, and give the verified answer."
        #
        # contain execution instructions rather than additional
        # database entities/metrics. The planner must not turn those
        # instructions into a second SQL step that consumes the first
        # step's returned rows. The investigation engine already has
        # bounded retry/recovery logic for insufficient SQL attempts.
        #
        # Normalize this into the actual analytical question while
        # preserving the user's request for verification/retry in the
        # objective.
        # ==================================================

        verification_core = self._extract_verification_core(original_question)

        if verification_core is not None:

            plan = InvestigationPlan(
                original_question=original_question,
                goal=original_question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=verification_core,
                        objective=(
                            "Directly answer the requested question. "
                            "If generation, execution, or deterministic "
                            "verification is insufficient, use the bounded "
                            "retry mechanism to repair or materially "
                            "change the SQL strategy before accepting the "
                            "result. The final evidence must be sufficient "
                            "to support the verified answer."
                        ),
                        required_information=[
                            verification_core
                        ],
                        depends_on=[],
                    )
                ],
            )

        # ==================================================
        # WHY-QUESTION SANITIZATION
        # ==================================================
        #
        # The planner does not know the schema.
        #
        # Therefore a model-generated "why" plan must never
        # introduce speculative explanatory dimensions.
        #
        # This deterministic layer removes that failure mode.
        # ==================================================

        if self._asks_why(
            original_question
        ):

            plan = self._sanitize_why_plan(
                question=original_question,
                plan=plan,
            )

        # ==================================================
        # RANKING INTENT PRESERVATION
        # ==================================================

        original_lower = original_question.lower()

        ranking_terms = {
            "most",
            "highest",
            "largest",
            "maximum",
            "top",
            "rank",
            "ranking",
        }

        lowest_terms = {
            "least",
            "lowest",
            "smallest",
            "minimum",
        }

        asks_ranking = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                original_lower,
            )
            for term in ranking_terms
        )

        asks_lowest = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                original_lower,
            )
            for term in lowest_terms
        )

        top_n = self._extract_top_n(original_lower)

        if asks_ranking or asks_lowest:

            required_direction = (
                "highest to lowest"
                if asks_ranking
                else "lowest to highest"
            )

            ranking_constraint = (
                "The original user request requires ranking. "
                f"Results must be ordered from {required_direction}."
            )

            if top_n is not None:
                ranking_constraint += (
                    f" The original request specifically requires "
                    f"the top {top_n} results."
                )

            # Always enforce ranking on the FINAL analytical step.
            # A prerequisite step may retrieve supporting data, but it
            # cannot satisfy a ranking request by itself.
            target_step = plan.sub_questions[-1] if plan.sub_questions else None

            if target_step is not None:

                combined = f"{target_step.question} {target_step.objective}".lower()

                if not any(
                    re.search(
                        rf"\b{re.escape(term)}\b",
                        combined,
                    )
                    for term in (ranking_terms | lowest_terms)
                ):
                    target_step.question = (
                        f"{target_step.question} Rank the results from "
                        f"{required_direction}."
                    )

                if top_n is not None and f"top {top_n}" not in target_step.question.lower():
                    target_step.question += (
                        f" Return the top {top_n} results."
                    )

                if ranking_constraint.lower() not in combined:
                    target_step.objective = (
                        f"{target_step.objective} {ranking_constraint}"
                    )

                if top_n is not None and f"top {top_n}" not in target_step.objective.lower():
                    target_step.objective += (
                        f" Return only the top {top_n} ranked results."
                    )

        plan = self._normalize_period_scoped_plan(
            original_question,
            plan,
        )

        self._validate_and_normalize_dependencies(
            plan
        )

        return plan

    @classmethod
    def _normalize_period_scoped_plan(
        cls,
        question: str,
        plan: InvestigationPlan,
    ) -> InvestigationPlan:
        if (
            not cls._has_explicit_time_period(question)
            or not cls._asks_for_period_comparison(question)
        ):
            return plan

        measure_text = " ".join(
            [
                question,
                *(f"{step.question} {step.objective}" for step in plan.sub_questions),
            ]
        )
        if not re.search(
            r"\b(?:metric|measure|total|sum|average|mean|amount|quantity|"
            r"volume|count|number|value|rate|sales|revenue|calculate|"
            r"aggregate|how many)\b",
            measure_text,
            re.IGNORECASE,
        ):
            return plan

        required_steps = [
            step for step in plan.sub_questions if not step.optional
        ]
        if not required_steps:
            return plan
        baseline = required_steps[0]

        comparison = next(
            (
                step
                for step in plan.sub_questions
                if step.optional
                and re.search(
                    r"\b(?:previous|prior|preceding|compare|comparison)\b",
                    f"{step.question} {step.objective}",
                    re.IGNORECASE,
                )
            ),
            None,
        )
        if comparison is None:
            comparison = SubQuestion(
                id=max(step.id for step in plan.sub_questions) + 1,
                question=(
                    "Compare the requested metric for the requested "
                    "period with the immediately preceding comparable "
                    "period."
                ),
                objective=(
                    "Measure the period-to-period difference, comparing "
                    "corresponding calendar or fiscal periods when the "
                    "data supports that comparison."
                ),
                depends_on=[baseline.id],
                optional=True,
            )
            plan.sub_questions.append(comparison)
        else:
            comparison.depends_on = [baseline.id]
            comparison.optional = True
            comparison.objective = (
                f"{comparison.objective} Compare corresponding calendar "
                "or fiscal periods when the data supports that comparison."
            )

        segment_steps = [
            step
            for step in plan.sub_questions
            if step.optional
            and step.id != comparison.id
            and re.search(
                r"\b(?:breakdown|broken down|segment|category|categories|"
                r"product|geograph|region|customer|channel|dimension)\b",
                f"{step.question} {step.objective}",
                re.IGNORECASE,
            )
        ]
        if not segment_steps:
            segment_steps = [
                SubQuestion(
                    id=max(step.id for step in plan.sub_questions) + 1,
                    question=(
                        "Break down the requested metric by relevant "
                        "dimensions available in the database."
                    ),
                    objective=(
                        "Identify measurable segments associated with "
                        "the period-to-period difference, using only "
                        "dimensions present in live schema metadata."
                    ),
                    optional=True,
                )
            ]
            plan.sub_questions.extend(segment_steps)

        for step in segment_steps:
            if not re.search(
                r"\b(?:previous|prior|preceding|compare|comparison)\b",
                step.question,
                re.IGNORECASE,
            ):
                step.question = (
                    f"{step.question.rstrip('.')} and compare the same "
                    "breakdown with the immediately preceding comparable "
                    "period."
                )
            step.objective = (
                f"{step.objective} Identify segments with the largest "
                "verified differences across corresponding periods; "
                "describe association, not causation."
            )
            step.depends_on = [baseline.id, comparison.id]
            step.optional = True

        return plan

    @classmethod
    def _remove_unrequested_period_context(
        cls,
        question: str,
        plan: InvestigationPlan,
    ) -> None:
        if cls._asks_for_period_comparison(question):
            return

        period_context_patterns = (
            r"\b(?:previous|prior|preceding)\s+(?:comparable\s+)?period\b",
            r"\bperiod[- ]to[- ]period\b",
            r"\bcompare\s+(?:the\s+)?same\s+breakdown\b",
            r"\bcorresponding\s+(?:calendar|fiscal)\s+periods?\b",
            r"\bsegments?\s+with\s+the\s+largest\s+"
            r"(?:verified|measured)\s+differences\b",
        )
        removed_ids = {
            step.id
            for step in plan.sub_questions
            if step.optional
            and any(
                re.search(
                    pattern,
                    f"{step.question} {step.objective}",
                    re.IGNORECASE,
                )
                for pattern in period_context_patterns
            )
        }
        if not removed_ids:
            return

        retained = [
            step
            for step in plan.sub_questions
            if step.id not in removed_ids
        ]
        for step in retained:
            step.depends_on = [
                dependency
                for dependency in step.depends_on
                if dependency not in removed_ids
            ]
        plan.sub_questions = retained

    @staticmethod
    def _asks_for_period_comparison(question: str) -> bool:
        return bool(
            re.search(
                r"\b(?:compare|compared|comparison|versus|vs\.?|"
                r"difference|trend|change|increase|increased|"
                r"decrease|decreased|growth|decline)\b",
                question or "",
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _extract_verification_core(question: str) -> str | None:
        """Extract the analytical question from conditional verification prose."""

        patterns = [
            r"^(.*?)(?:\.|;)\s*if\s+(?:the\s+)?first\s+result\b",
            r"^(.*?)(?:\.|;)\s*if\s+the\s+first\s+result\b",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                question.strip(),
                flags=re.IGNORECASE | re.DOTALL,
            )

            if match:
                core = match.group(1).strip().rstrip(".;")
                if core:
                    return core

        return None

    # ==========================================================
    # WHY HELPERS
    # ==========================================================

    @staticmethod
    def _asks_why(
        question: str,
    ) -> bool:

        return bool(
            re.search(
                r"\b(?:why|reason|reasons|explain|explanation|cause|"
                r"causes|caused|driver|drivers|factor|factors|"
                r"contributor|contributors|contributed)\b",
                question,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _explicit_years(question: str) -> list[int]:
        return sorted({
            int(year)
            for year in re.findall(r"\b(?:19|20)\d{2}\b", question or "")
        })

    @classmethod
    def _has_explicit_year_comparison(cls, question: str) -> bool:
        return (
            len(cls._explicit_years(question)) == 2
            and bool(
                re.search(
                    r"\b(?:compare|comparison|versus|vs\.?|against|"
                    r"increase|increased|increasing|decrease|decreased|"
                    r"decreasing|higher|lower|change|changed|changing|"
                    r"difference|delta|trend|growth|grow|decline|declined)\b",
                    question or "",
                    re.IGNORECASE,
                )
            )
        )

    @staticmethod
    def _has_explicit_time_period(question: str) -> bool:
        return bool(
            re.search(
                r"\b(?:"
                r"january|february|march|april|may|june|july|august|"
                r"september|october|november|december|"
                r"q[1-4]|quarter|"
                r"today|yesterday|week|month|quarter|year|"
                r"last|previous|prior|current|this|past|next"
                r")\b|\b\d{4}(?:-\d{2}(?:-\d{2})?)?\b",
                question or "",
                re.IGNORECASE,
            )
        )

    @classmethod
    def _sanitize_why_plan(
        cls,
        question: str,
        plan: InvestigationPlan,
    ) -> InvestigationPlan:

        """
        Prevent the planner from introducing unsupported explanatory
        dimensions into a why-investigation.

        The planner has no database schema, so it is allowed to ask
        for measurable dimensions directly implied by the question,
        but not speculative domain-specific explanations.
        """

        unsupported_causal_patterns = [
            r"\bcauses?\b",
            r"\bcaused by\b",
            r"\bdrives?\b",
            r"\bdriven by\b",
            r"\bleads? to\b",
            r"\bresponsible for\b",
        ]

        cleaned_steps: list[
            SubQuestion
        ] = []

        for step in plan.sub_questions:

            combined = (
                f"{step.question} "
                f"{step.objective} "
                f"{' '.join(step.required_information)}"
            )

            unsupported_causal_claim = any(
                re.search(
                    pattern,
                    combined,
                    re.IGNORECASE,
                )
                for pattern in unsupported_causal_patterns
            )

            if unsupported_causal_claim:
                step.question = (
                    "Compare the measurements, entities, and dimensions "
                    "requested by the user to identify evidence-supported "
                    "differences associated with the outcome."
                )
                step.objective = (
                    "Report measurable associations relevant to the user's "
                    "question without asserting an unsupported causal "
                    "relationship."
                )
                step.required_information = [question]
                step.optional = True

            cleaned_steps.append(
                step
            )

        plan.sub_questions = (
            cleaned_steps
        )

        # --------------------------------------------------
        # Ensure a why-question has an evidence baseline.
        # --------------------------------------------------

        baseline_exists = any(
            cls._looks_like_baseline_step(
                step
            )
            for step in plan.sub_questions
        )

        if not baseline_exists:

            next_id = (
                max(
                    (
                        step.id
                        for step in plan.sub_questions
                    ),
                    default=0,
                )
                + 1
            )

            baseline = SubQuestion(
                id=next_id,
                question=(
                    "Measure the metric and entities named in the "
                    "original question, preserving its filters, to "
                    "establish an evidence baseline."
                ),
                objective=(
                    "Establish the requested measurable baseline before "
                    "comparing observed outcomes or associated dimensions."
                ),
                required_information=[
                    question,
                ],
                depends_on=[],
            )

            # The baseline should come first.
            for step in plan.sub_questions:
                step.depends_on = [
                    baseline.id
                    if dependency == baseline.id
                    else dependency
                    for dependency in step.depends_on
                ]

            plan.sub_questions.insert(
                0,
                baseline,
            )

            # Re-map IDs to maintain a simple sequential plan.
            id_mapping = {}

            for index, step in enumerate(
                plan.sub_questions,
                start=1,
            ):

                id_mapping[
                    step.id
                ] = index

            for step in plan.sub_questions:

                old_id = step.id

                step.id = id_mapping[
                    old_id
                ]

                step.depends_on = [
                    id_mapping[
                        dependency
                    ]
                    for dependency
                    in step.depends_on
                    if dependency in id_mapping
                ]

        if not cls._asks_for_specific_breakdown(question):
            baseline_step = next(
                (
                    step
                    for step in plan.sub_questions
                    if cls._looks_like_baseline_step(step)
                ),
                None,
            )
            if baseline_step is not None:
                baseline_step.optional = False
            for step in plan.sub_questions:
                if baseline_step is None or step.id != baseline_step.id:
                    step.optional = True

        baseline_step = next(
            (
                step
                for step in plan.sub_questions
                if cls._looks_like_baseline_step(step)
            ),
            None,
        )
        if baseline_step is not None:
            baseline_step.optional = False

        if cls._asks_for_specific_breakdown(question):
            for step in plan.sub_questions:
                if baseline_step is not None and step.id == baseline_step.id:
                    continue
                step.optional = not cls._is_relevant_breakdown_step(
                    question,
                    step,
                )

        breakdown_step_exists = any(
            cls._is_relevant_breakdown_step(question, step)
            for step in plan.sub_questions
            if baseline_step is None or step.id != baseline_step.id
        )
        if not breakdown_step_exists:
            next_id = max(
                (step.id for step in plan.sub_questions),
                default=0,
            ) + 1
            requested_breakdown = cls._asks_for_specific_breakdown(question)
            plan.sub_questions.append(
                SubQuestion(
                    id=next_id,
                    question=(
                        (
                            "Compare the measured outcome across "
                            f"{cls._requested_breakdown_label(question)} values."
                        )
                        if requested_breakdown
                        and cls._requested_breakdown_label(question)
                        else (
                            "Compare the measured outcome across the "
                            "explicitly requested dimensions."
                            if requested_breakdown
                            else
                            "Compare the measured outcome across relevant "
                            "dimensions available in the database."
                        )
                    ),
                    objective=(
                        "Use only dimensions present in live schema "
                        "metadata. Identify measured associations with the "
                        "outcome, without asserting that a segment caused it."
                    ),
                    required_information=[question],
                    depends_on=(
                        [baseline_step.id] if baseline_step is not None else []
                    ),
                    optional=not requested_breakdown,
                )
            )

        if not cls._asks_for_specific_breakdown(question):
            for step in plan.sub_questions:
                if baseline_step is None or step.id != baseline_step.id:
                    step.optional = True

        # --------------------------------------------------
        # Remove duplicate analytical steps.
        # --------------------------------------------------

        unique_steps = []
        fingerprints = set()

        for step in plan.sub_questions:

            fingerprint = re.sub(
                r"\s+",
                " ",
                (
                    f"{step.question} "
                    f"{step.objective}"
                ).strip().lower(),
            )

            if fingerprint in fingerprints:
                continue

            fingerprints.add(
                fingerprint
            )

            unique_steps.append(
                step
            )

        plan.sub_questions = (
            unique_steps
        )

        # --------------------------------------------------
        # Normalize IDs after removal/insertion.
        # --------------------------------------------------

        old_to_new = {}

        for index, step in enumerate(
            plan.sub_questions,
            start=1,
        ):

            old_to_new[
                step.id
            ] = index

        for index, step in enumerate(
            plan.sub_questions,
            start=1,
        ):

            step.id = index

            step.depends_on = [
                old_to_new[
                    dependency
                ]
                for dependency
                in step.depends_on
                if dependency in old_to_new
            ]

        # A why investigation should not contain a giant number
        # of speculative steps.
        plan.sub_questions = (
            plan.sub_questions[:6]
        )

        return plan

    @classmethod
    def _is_relevant_breakdown_step(
        cls,
        question: str,
        step: SubQuestion,
    ) -> bool:
        text = f"{step.question} {step.objective}".lower()
        dimension_terms = (
            "product",
            "categor",
            "region",
            "countr",
            "store",
            "location",
            "customer",
            "client",
            "segment",
            "channel",
            "method",
            "type",
            "department",
        )
        if cls._asks_for_specific_breakdown(question):
            requested = [
                term
                for term in dimension_terms
                if re.search(
                    rf"\b{re.escape(term)}\w*\b",
                    question,
                    re.IGNORECASE,
                )
            ]
            return bool(requested) and any(term in text for term in requested)
        return bool(
            re.search(
                r"\b(?:breakdown|broken down|segment|category|categories|"
                r"dimension|by\s+\w+|across\s+\w+)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _asks_for_specific_breakdown(question: str) -> bool:
        return QuestionPlanner._requested_breakdown_label(question) is not None

    @staticmethod
    def _requested_breakdown_label(question: str) -> str | None:
        patterns = (
            (r"\bproduct\s+categor(?:y|ies)\b", "product category"),
            (r"\bproducts?\b", "product"),
            (r"\bcategor(?:y|ies)\b", "category"),
            (r"\bregions?\b", "region"),
            (r"\bcountries\b|\bcountry\b", "country"),
            (r"\bstores?\b", "store"),
            (r"\blocations?\b", "location"),
            (r"\bcustomers?\b", "customer"),
            (r"\bclients?\b", "client"),
            (r"\bsegments?\b", "segment"),
            (r"\bchannels?\b", "channel"),
            (r"\bmethods?\b", "method"),
            (r"\btypes?\b", "type"),
            (r"\bdepartments?\b", "department"),
        )
        dimension_pattern = "|".join(
            f"(?:{pattern})"
            for pattern, _ in patterns
        )
        explicitly_requested = bool(
            re.search(
                rf"\b(?:by|per|across)\s+(?:the\s+)?(?:{dimension_pattern})\b|"
                rf"\bwhich\s+(?:(?:top|most|least|highest|lowest)\s+)?"
                rf"(?:{dimension_pattern})\b",
                question or "",
                re.IGNORECASE,
            )
        )
        if not explicitly_requested:
            return None
        for pattern, label in patterns:
            if re.search(pattern, question or "", re.IGNORECASE):
                return label
        return None

    @staticmethod
    def _looks_like_baseline_step(
        step: SubQuestion,
    ) -> bool:

        text = (
            f"{step.question} "
            f"{step.objective}"
        ).lower()

        baseline_terms = [
            "measure",
            "count",
            "number",
            "baseline",
            "metric",
            "value",
            "total",
            "rate",
            "compare",
        ]

        return (
            sum(
                term in text
                for term in baseline_terms
            )
            >= 2
        )

    # ==========================================================
    # DIRECT RANKING LOOKUP
    # ==========================================================

    @staticmethod
    def _is_direct_ranking_lookup(
        question: str,
    ) -> bool:
        """Return True for a self-contained ranking/Top-N analysis.

        A comparison request that asks for a metric across entities and
        then asks for the highest/lowest/top-N entities is still one
        relational analytical operation. It should not be decomposed into
        entity discovery -> relationship discovery -> aggregation.

        Only genuinely multi-stage analytical intents such as why, trend,
        correlation, causation, or explicit investigation/retry instructions
        remain excluded.
        """

        text = " ".join(
            question.lower().split()
        )

        if not text:
            return False

        # These intents genuinely require investigation beyond a single
        # ranked aggregate operation. "compare" is intentionally NOT here:
        # compare + metric + ranking/Top-N can be answered by one aggregate
        # query with GROUP BY / ORDER BY / LIMIT.
        if any(
            re.search(pattern, text)
            for pattern in (
                r"\bwhy\b",
                r"\btrend\b",
                r"\bover time\b",
                r"\bcorrelation\b",
                r"\brelationship\b",
                r"\bimpact\b",
                r"\beffect\b",
                r"\bcaus(?:e|es|ed|al)\b",
                r"\binvestigate\b",
                r"\binsufficient\b",
                r"\binconsistent\b",
                r"\brerun\b",
                r"\bre-?run\b",
                r"\boptimize\b",
            )
        ):
            return False

        asks_highest = bool(
            re.search(
                r"\b(?:most|highest|largest|maximum|top|rank|ranking)\b",
                text,
            )
        )
        asks_lowest = bool(
            re.search(
                r"\b(?:least|lowest|smallest|minimum|min)\b",
                text,
            )
        )
        has_ranking = asks_highest or asks_lowest

        if not has_ranking:
            return False

        # A ranking request must identify a measurable quantity.
        has_measure = re.search(
            r"\b(?:how many|number of|count|counts|value|values|"
            r"amount|quantity|total|rate|ratio|percentage|percent|"
            r"score|volume|size|metric|measure|relev\w*|strength|"
            r"similarity|affinity|confidence)\b",
            text,
        )
        has_explicit_top_n = QuestionPlanner._extract_top_n(text) is not None
        has_measure_after_rank = bool(
            re.search(
                r"\b(?:most|least|highest|lowest|largest|smallest|"
                r"maximum|minimum|max|min)\s+"
                r"(?:of\s+)?(?:the\s+)?(?:\w+\s+){0,3}\w+",
                text,
            )
        )
        return bool(
            has_measure
            or has_explicit_top_n
            or has_measure_after_rank
        )

    # ==========================================================
    # DIRECT LOOKUP
    # ==========================================================

    @staticmethod
    def _is_direct_lookup(
        question: str,
    ) -> bool:

        text = " ".join(
            question.lower().split()
        )

        if not text:
            return False

        complex_signals = [
            r"\bwhy\b",
            r"\bexplain\b",
            r"\breason(?:s)?\b",
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\btrend\b",
            r"\bover time\b",
            r"\bcorrelation\b",
            r"\bimpact\b",
            r"\beffect\b",
            r"\bcaus(?:e|es|ed|al)\b",
            r"\binvestigate\b",
            r"\binsufficient\b",
            r"\binconsistent\b",
            r"\brerun\b",
            r"\bre-?run\b",
            r"\bretry\b",
            r"\boptimize\b",
            r"\bcross[- ]?check\b",
            r"\btop\s+\d+\b",
            (
                r"\b(?:most|highest|lowest|least|largest|"
                r"smallest|maximum|minimum)\b"
            ),
        ]

        if any(
            re.search(
                pattern,
                text,
            )
            for pattern in complex_signals
        ):

            return False

        direct_signals = [
            r"\bwhich\b",
            r"\bwhat\b",
            r"\bshow\b",
            r"\blist\b",
            r"\bfind\b",
            r"\bget\b",
            r"\bhow many\b",
            r"\bnumber of\b",
            r"\bcount\b",
        ]

        return any(
            re.search(
                pattern,
                text,
            )
            for pattern in direct_signals
        )

    @classmethod
    def _is_simple_count_lookup(
        cls,
        question: str,
    ) -> bool:
        text = " ".join(question.lower().split())
        if not cls._is_direct_lookup(text):
            return False
        if not re.search(
            r"\b(?:how many|number of|count)\b",
            text,
        ):
            return False
        return not re.search(
            r"\b(?:per|each|every|by|across|among|between|compare(?:d)?|"
            r"difference|versus|vs\.?|trend|increase|decrease|change|"
            r"previous|prior|before|after|higher|lower)\b",
            text,
        )

    # ==========================================================
    # TOP-N
    # ==========================================================

    @staticmethod
    def _extract_top_n(
        question: str,
    ) -> int | None:

        match = re.search(
            r"\b(?:top|first|which)\s+(\d+)\b",
            question,
            flags=re.IGNORECASE,
        )

        if not match:
            return None

        return int(
            match.group(1)
        )

    # ==========================================================
    # DEPENDENCY VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_and_normalize_dependencies(
        plan: InvestigationPlan,
    ) -> None:

        steps = plan.sub_questions

        if not steps:

            raise ValueError(
                "Investigation planner returned no sub-questions."
            )

        ids = [
            step.id
            for step in steps
        ]

        if len(ids) != len(
            set(ids)
        ):

            raise ValueError(
                "Investigation plan contains duplicate step IDs."
            )

        known_ids = set(
            ids
        )

        position_by_id = {
            step.id: index
            for index, step in enumerate(
                steps
            )
        }

        for step in steps:

            cleaned = []

            for dependency_id in (
                step.depends_on
            ):

                if (
                    dependency_id
                    == step.id
                ):

                    raise ValueError(
                        f"Step {step.id} cannot depend on itself."
                    )

                if (
                    dependency_id
                    not in known_ids
                ):

                    raise ValueError(
                        f"Step {step.id} depends on unknown "
                        f"step {dependency_id}."
                    )

                if (
                    position_by_id[
                        dependency_id
                    ]
                    >= position_by_id[
                        step.id
                    ]
                ):

                    raise ValueError(
                        f"Step {step.id} depends on step "
                        f"{dependency_id}, which is scheduled later."
                    )

                if (
                    dependency_id
                    not in cleaned
                ):

                    cleaned.append(
                        dependency_id
                    )

            step.depends_on = (
                cleaned
            )

        graph = {
            step.id: list(
                step.depends_on
            )
            for step in steps
        }

        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(
            step_id: int,
        ) -> None:

            if step_id in visiting:

                raise ValueError(
                    "Investigation plan contains a dependency cycle."
                )

            if step_id in visited:
                return

            visiting.add(
                step_id
            )

            for dependency_id in (
                graph[step_id]
            ):

                visit(
                    dependency_id
                )

            visiting.remove(
                step_id
            )

            visited.add(
                step_id
            )

        for step_id in graph:

            visit(
                step_id
            )


if __name__ == "__main__":

    planner = QuestionPlanner()

    question = (
        "Why did sales decrease in August?"
    )

    plan = planner.create_plan(
        question
    )

    print(
        plan.model_dump_json(
            indent=2
        )
    )