# Secondary experiment: generation-first CAPS dose selection

This is a separate, outcome-informed experiment, frozen after the main CAPS calibration revealed a sign conflict and before any secondary calibration/heldout outputs. It does not replace or repair the main experiment retrospectively. The earlier project had found public g=-0.2 useful on some copy prompts; the main experiment's first-token filter selected g=+0.2 and no qualified generation recipe. That motivates testing whether the cheap filter discarded useful doses.

Use three coordinates, fixed from main calibration: public L1/N8396; shared signed/RMS screening winner L30/N271; and the signed method's random-screen winner L31/N1699. Use the exact same ten additive, six scale, and six clamp recipes, now generating at all 22 recipes per coordinate without first-token pruning. Clamp values reuse hashed discovery captures; no new discovery ranking is performed.

Two entirely new calibration sentences and two new protections select a recipe. Require both copies to retain all content modulo case, both to reach 85% uppercase, and both baseline-correct protections to remain correct. Prefer greater mean uppercase fraction, then lower mean realized write norm. Freeze the result before eight new copy sentences and four new protection prompts. If no recipe qualifies, deploy zero and evaluate one explicitly failed challenger descriptively. Public additive g=-0.2, +0.2 and -1 are separately fixed reference conditions, with identical recipes aliased rather than rerun.

The secondary budget is 10,000 forwards and 60 minutes, zero backwards. It starts only after the main process exits successfully. The same greedy model/runtime and all-token intervention scope apply. Selection and all unsuccessful recipes remain in the record. It cannot demonstrate global novelty, comprehensive absence of other useful methods, or independent superiority of signed ranking when signed and RMS chose the same candidate.

[Protocol](../generation-audit/source/protocol.json) · [Source hashes](../generation-audit/source/source-hashes.json) · [Main protocol](protocol.md).
