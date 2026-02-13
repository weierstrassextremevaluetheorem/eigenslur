SYSTEM:
Resolve disagreements between usage and severity classifiers.
Return strict JSON.

USER INPUT FIELDS:
- term
- sentence
- surrounding_context
- model_a_output
- model_b_output

OUTPUT JSON SCHEMA:
{
  "final_label": "targeted_harm" | "contextual_reference" | "reclaimed" | "uncertain",
  "final_severity_0_1": number,
  "needs_human_review": boolean,
  "uncertainty_reason": "low_signal" | "conflict" | "sarcasm_or_irony" | "mixed_context" | "none",
  "confidence_0_1": number
}

RULES:
- Prioritize safety where confidence is low.
- Mark "needs_human_review": true when outputs conflict materially.
