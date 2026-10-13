import json
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field

from app.llm.groq import GroqProvider


class InsightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    findings: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


def test_groq_retries_rejected_optional_field_schema_with_json_mode():
    provider = object.__new__(GroqProvider)
    calls = []
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "summary": "Sales increased.",
                        "findings": ["Sales increased between the years."],
                        "supporting_evidence": ["Verified annual totals."],
                        "recommendations": [],
                        "uncertainties": [],
                    })
                )
            )
        ]
    )

    def request(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "invalid JSON schema for response_format: required is missing"
            )
        return response

    provider._request = request

    result = provider.generate(
        model="openai/gpt-oss-20b",
        system_instruction="Return a grounded result.",
        user_prompt="Did sales grow?",
        response_model=InsightResponse,
    )

    assert result.summary == "Sales increased."
    assert len(calls) == 2
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[1]["response_format"] == {"type": "json_object"}
