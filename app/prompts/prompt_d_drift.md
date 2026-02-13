SYSTEM:
Detect semantic drift and new variants of a candidate sensitive term.
Return strict JSON only.

USER INPUT FIELDS:
- term
- baseline_contexts_summary
- recent_contexts_summary
- locale

OUTPUT JSON SCHEMA:
{
  "semantic_shift_0_1": number,
  "new_variant_detected": boolean,
  "variant_examples": [string],
  "confidence_0_1": number
}

RULES:
- "semantic_shift_0_1" should represent meaning change, not only frequency change.
- Keep "variant_examples" short.
