import re
from typing import List

from pydantic import BaseModel, Field, ConfigDict

from app.llm.model_router import (
    ModelRouter,
    create_default_model_router,
)


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

    def create_plan(
        self,
        question: str,
    ) -> InvestigationPlan:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

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
- gene types
- biological functions
- conservation scores
- expression levels
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

For example:

User:
"Why do some pathways have more human-ortholog mappings than others?"

BAD:
"Analyze the distribution of gene types or functional categories
within pathways."

BAD:
"Compare evolutionary conservation levels between pathways."

BAD:
"Determine whether pathway function explains the difference."

These are invalid because the planner has no schema knowledge
and the user did not request those dimensions.

GOOD:
"Compare the number of FlyBase genes and the number of
FlyBase genes with human-ortholog mappings across pathways."

GOOD:
"Measure the proportion of pathway-member genes that have
human-ortholog mappings and compare that proportion across
pathways."

GOOD:
"Identify measurable pathway-level differences in gene
membership and human-ortholog coverage."

The later schema-retrieval and SQL-generation layers determine
which of these measurements can actually be answered.

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

For example:

Question:
"Why do some pathways have more human-ortholog mappings than others?"

Preferred plan:

Step 1:
"Measure the number of FlyBase genes and FlyBase genes with
human-ortholog mappings for each pathway."

Step 2:
"Compare human-ortholog coverage across pathways by measuring
the proportion of pathway-member genes that have human-ortholog
mappings."

Step 3:
"Compare pathway membership and human-ortholog mapping patterns
to identify measurable differences associated with the observed
mapping counts."

NOT:

"Analyze gene functional categories."

NOT:

"Analyze evolutionary conservation."

NOT:

"Determine which biological functions cause the difference."

============================================================
IMPORTANT RANKING EXAMPLE
============================================================

User:

"Which pathways contain the most FlyBase genes with human
orthologs, and how many are in each?"

Valid decomposition:

Step 1:
"Identify FlyBase genes that have human orthologs."

Step 2:
"Determine the pathways associated with those FlyBase genes."

Step 3:
"Count the FlyBase genes with human orthologs in each pathway
and rank the pathways from highest to lowest count."

The final analytical requirement must never be reduced to:

"Count genes by pathway."

============================================================
OUTPUT
============================================================

Return structured output only.
"""

        user_prompt = f"""
Create an investigation plan for the following ORIGINAL
USER QUESTION:

{question}

The original question is authoritative.

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
"""

        plan = self.model_router.generate(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=InvestigationPlan,
            complexity="fast",
            classification_text=question,
        )

        self._validate_and_normalize_dependencies(
            plan
        )

        # ==================================================
        # DIRECT RANKING NORMALIZATION
        # ==================================================
        #
        # A request such as:
        #   "Which pathways contain the most X, and how many are in each?"
        # is already a complete analytical request. It does not need
        # a prerequisite "find pathways" step followed by a ranking
        # step. Keeping it as one step prevents intermediate row-level
        # duplicates from becoming a false verification failure.
        #
        # Comparison questions (for example Q5) are intentionally NOT
        # normalized here; they retain their multi-step investigation.
        # ==================================================

        if self._is_direct_ranking_lookup(question):

            top_n = self._extract_top_n(question.lower())
            ranking_text = (
                "Rank the requested entities from highest to lowest by "
                "the metric requested by the user and return the metric "
                "value for each ranked entity."
            )

            if top_n is not None:
                ranking_text += (
                    f" Return exactly the top {top_n} results."
                )

            plan = InvestigationPlan(
                original_question=question,
                goal=question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=question,
                        objective=(
                            "Directly answer the original ranking "
                            "request in one database operation. "
                            + ranking_text
                        ),
                        required_information=[
                            question,
                            "entity names and the requested ranking metric",
                        ],
                        depends_on=[],
                    )
                ],
            )

        # ==================================================
        # DIRECT LOOKUP NORMALIZATION
        # ==================================================
        # ==================================================
        # DIRECT LOOKUP NORMALIZATION
        # ==================================================

        if self._is_direct_lookup(
            question
        ):

            plan = InvestigationPlan(
                original_question=question,
                goal=question,
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question=question,
                        objective=(
                            "Directly retrieve the data required "
                            "to answer the original user question."
                        ),
                        required_information=[
                            question
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

        verification_core = self._extract_verification_core(question)

        if verification_core is not None:

            plan = InvestigationPlan(
                original_question=question,
                goal=question,
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
            question
        ):

            plan = self._sanitize_why_plan(
                question=question,
                plan=plan,
            )

        # ==================================================
        # RANKING INTENT PRESERVATION
        # ==================================================

        original_lower = question.lower()

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

        self._validate_and_normalize_dependencies(
            plan
        )

        return plan

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
                r"\bwhy\b",
                question,
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

        forbidden_patterns = [
            r"\bfunctional categories?\b",
            r"\bfunctional category\b",
            r"\bgene types?\b",
            r"\btypes? of genes\b",
            r"\bevolutionary conservation\b",
            r"\bconservation level\b",
            r"\bexpression levels?\b",
            r"\bdisease categories?\b",
            r"\bbiological functions?\b",
            r"\bbiological pathways?\b",
            r"\bcausal factors?\b",
            r"\bcauses?\b",
            r"\broot causes?\b",
            r"\bexplain(?:s|ed)? by\b",
            r"\bdriv(?:e|es|en)\b",
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

            speculative = any(
                re.search(
                    pattern,
                    combined,
                    re.IGNORECASE,
                )
                for pattern in forbidden_patterns
            )

            if speculative:

                # Replace speculative analytical steps with a
                # schema-agnostic measurable comparison.
                replacement_question = (
                    "Compare measurable pathway-level differences "
                    "directly related to the observed human-ortholog "
                    "mapping counts."
                )

                replacement_objective = (
                    "Identify measurable differences in pathway "
                    "membership and human-ortholog coverage that "
                    "are supported by the available database schema. "
                    "Do not assume an explanatory variable that is "
                    "not represented in the database."
                )

                required_information = [
                    "pathway-level gene membership",
                    "FlyBase gene counts",
                    "human-ortholog mapping counts",
                    "human-ortholog coverage",
                ]

                step.question = (
                    replacement_question
                )

                step.objective = (
                    replacement_objective
                )

                step.required_information = (
                    required_information
                )

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
                    "Measure the relevant pathway-level "
                    "human-ortholog mapping counts and the "
                    "corresponding pathway gene membership."
                ),
                objective=(
                    "Establish the measurable baseline needed "
                    "to compare pathways with different numbers "
                    "of human-ortholog mappings."
                ),
                required_information=[
                    "pathway-level gene membership",
                    "FlyBase gene counts",
                    "human-ortholog mapping counts",
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
            "mapping",
            "baseline",
            "pathway-level",
            "pathway level",
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

        has_ranking = bool(
            re.search(
                r"\b(?:most|highest|largest|maximum|top|rank|ranking)\b",
                text,
            )
        )

        if not has_ranking:
            return False

        # A ranking request must contain a metric/value requirement.
        # This covers "how many", "number of", "count", etc.
        return bool(
            re.search(
                r"\b(?:how many|number of|count|counts|value|values)\b",
                text,
            )
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

    # ==========================================================
    # TOP-N
    # ==========================================================

    @staticmethod
    def _extract_top_n(
        question: str,
    ) -> int | None:

        match = re.search(
            r"\btop\s+(\d+)\b",
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