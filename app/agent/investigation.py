from app.agent.planner import QuestionPlanner
from app.database.schema_retriever import get_relevant_schema


class InvestigationEngine:

    def __init__(self):
        self.planner = QuestionPlanner()

    def investigate(self, question: str):

        plan = self.planner.create_plan(question)

        investigation = {
            "question": question,
            "goal": plan.goal,
            "sub_questions": [],
        }

        for sub_question in plan.sub_questions:

            schema_context = get_relevant_schema(
                sub_question.required_information
            )

            investigation["sub_questions"].append({
                "id": sub_question.id,
                "question": sub_question.question,
                "required_information": (
                    sub_question.required_information
                ),
                "schema_context": schema_context,
            })

        return investigation


if __name__ == "__main__":

    engine = InvestigationEngine()

    question = "Which genes have human orthologs?"

    result = engine.investigate(question)

    import json

    print(
        json.dumps(
            result,
            indent=2
        )
    )