from __future__ import annotations

from pathlib import Path

PROMPT_FILES = {
    "A_usage_classification": "prompt_a_usage.md",
    "B_severity_scoring": "prompt_b_severity.md",
    "C_ambiguity_resolution": "prompt_c_disambiguation.md",
    "D_drift_detection": "prompt_d_drift.md",
}


def load_prompt_templates() -> dict[str, str]:
    base_dir = Path(__file__).resolve().parent / "prompts"
    templates: dict[str, str] = {}
    for name, filename in PROMPT_FILES.items():
        path = base_dir / filename
        templates[name] = path.read_text(encoding="utf-8")
    return templates
