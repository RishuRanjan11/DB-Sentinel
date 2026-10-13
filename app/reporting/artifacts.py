from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class DashboardWidget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    title: str
    data: Any
    source_step_ids: list[int] = Field(default_factory=list)


class AnalysisDashboard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    widgets: list[DashboardWidget] = Field(default_factory=list)


class BusinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    executive_summary: str
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)


def build_artifacts(question: str, insight: dict | None, evidence: list[dict]) -> tuple[AnalysisDashboard, BusinessReport]:
    widgets: list[DashboardWidget] = []
    for item in evidence:
        rows = item.get("rows") or []
        if rows:
            widgets.append(
                DashboardWidget(
                    type="data_table",
                    title=f"Evidence — Step {item.get('step_id')}",
                    data=rows,
                    source_step_ids=[int(item["step_id"])],
                )
            )

    insight = insight or {}
    findings = list(insight.get("findings") or [])
    uncertainties = list(insight.get("uncertainties") or [])
    summary = str(insight.get("summary") or "Verified evidence was collected for the requested analysis.")

    dashboard = AnalysisDashboard(title=question, widgets=widgets)
    report = BusinessReport(
        title=question,
        executive_summary=summary,
        findings=findings,
        recommendations=list(insight.get("recommendations") or []),
        uncertainties=uncertainties,
        evidence=evidence,
    )
    return dashboard, report
