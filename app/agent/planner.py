from pydantic import BaseModel, Field
from typing import List


class SubQuestion(BaseModel):
    id: int
    question: str
    required_information: List[str] = Field(default_factory=list)


class InvestigationPlan(BaseModel):
    original_question: str
    goal: str
    sub_questions: List[SubQuestion]


class QuestionPlanner:
    """
    Creates an investigation plan from a natural-language question.

    For now this is a deterministic prototype.
    Later, the LLM will generate the plan.
    """

    def create_plan(self, question: str) -> InvestigationPlan:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        # Temporary planning logic.
        # We will replace this with an LLM-based planner later.
        if "ortholog" in question.lower():
            return InvestigationPlan(
                original_question=question,
                goal="Find genes associated with human orthologs",
                sub_questions=[
                    SubQuestion(
                        id=1,
                        question="Which genes are present in FlyBase?",
                        required_information=[
                            "gene identifier",
                            "gene symbol",
                            "gene name",
                        ],
                    ),
                    SubQuestion(
                        id=2,
                        question="Which FlyBase genes have human orthologs?",
                        required_information=[
                            "FlyBase gene",
                            "human ortholog",
                        ],
                    ),
                ],
            )

        # Generic fallback
        return InvestigationPlan(
            original_question=question,
            goal=question,
            sub_questions=[
                SubQuestion(
                    id=1,
                    question=question,
                    required_information=[],
                )
            ],
        )


if __name__ == "__main__":
    planner = QuestionPlanner()

    question = "Which genes have human orthologs?"

    plan = planner.create_plan(question)

    print(plan.model_dump_json(indent=2))