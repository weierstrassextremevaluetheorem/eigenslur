from __future__ import annotations

from dataclasses import dataclass

from app.prompt_templates import load_prompt_templates


@dataclass(frozen=True)
class LabelerPromptSet:
    usage_prompt: str
    severity_prompt: str
    disambiguation_prompt: str
    drift_prompt: str


def load_labeler_prompts() -> LabelerPromptSet:
    templates = load_prompt_templates()
    return LabelerPromptSet(
        usage_prompt=templates["A_usage_classification"],
        severity_prompt=templates["B_severity_scoring"],
        disambiguation_prompt=templates["C_ambiguity_resolution"],
        drift_prompt=templates["D_drift_detection"],
    )
