# Adverse repetition persists after a short pulse

Observed in the completed calibration, before inspecting any group-extension result. This is a descriptive analysis of existing sealed generations; no extra model calls were made.

For storytelling candidate L26 N2167, both creative calibration prompts reached the 72-token cap at g=+0.4 and +0.8. At each dose and prompt, the complete output token sequence was identical under continuous steering and pulse4. All four comparisons matched exactly. Continuous steering wrote on 72 predictions, while pulse4 wrote only on the first four. The remaining output continued repeating “Once” or “once-time.”

This establishes persistence of an undesirable output pattern after direct writes stop. It does not establish a useful storytelling effect or an internal memory mechanism. The output prefix itself can condition later tokens, and the model also carries internal cached states. The present comparison does not separate those routes.

At +0.8, the exact echo task also failed under both schedules. Continuous steering returned a capped repetitive fragment; pulse4 ended naturally but returned the wrong string. Nostalgia candidate L22 N8168 at -0.8 returned “416” for 6 × 7 under both schedules. Neither tested coordinate acquired a qualified smaller-dose recipe in this study.

## Possible follow-up, not launched

Replay the same first four generated tokens into a fresh, unsteered model state, then compare its continuation with pulse-and-release and an unsteered neutral-prefix control. Freeze exact prompts, prefixes, continuations and decoding before running. This could distinguish persistence explained by the output prefix from an additional contribution of cached state. It would need a separate protocol; it is not included in the current group experiment.

[Exact comparisons](../pulse-persistence.json) · [Full results](results.md) · [Raw calibration](../evidence/calibration/rows.json)
