# Groups versus singletons: a fixed-dose causal extension

Prepared following the user's request to extend the finding that groups can provide more consistent causal effects than some top singletons. This is a separate follow-up after the portable storytelling/nostalgia experiment. It does not alter that experiment's frozen source, dose selection, grading or held-out cohort.

## Why test this

The completed inventory-excluded study found positive donor-restoration effects on 3/8 held-out prompts for the loss-of-language top singleton and 2/8 for the academic singleton. Both top-4 and top-16 groups were positive on 8/8 in those categories. Storytelling and nostalgia singletons already succeeded directionally on 8/8 and serve as strong-singleton comparison cases. The old group-versus-singleton interventions did not equalize total write strength, and every member was not tested separately. That result supports an assay-specific observation, not a general group advantage or synergy claim.

These four categories are chosen because of those prior outcomes. This is a new test of a hypothesis suggested by the old data, not a blind category selection. All coordinate sets retain the original signed-mean PP ordering.

## Fixed inputs, fresh prompts

- Use the same pinned Qwen3.5-4B checkpoint and hybrid FP16 CUDA-transformer / FP32 CPU-vocabulary runtime.
- Keep original PP top-1/4/16 coordinates. Do not rerank, add inventory hints, or replace any coordinate.
- Fix each coordinate's direction coefficient to its mean target-minus-base activation across the original eight discovery pairs. Compute this from the hash-verified original activation archive. No new test prompt supplies a donor activation.
- Use eight new authored requests, disjoint from the earlier discovery/validation requests and the portable-style prompts. Reuse them across the four categories, and retain that clustering limitation.
- Measure the original frozen lexical logit contrast at the final prompt position. Base-prompt edits follow the discovery direction; target-prompt edits reverse it. Positive oriented effects mean the intended direction in either case.

This extension performs single-forward causal measurements. It does not generate stories, certify semantic style, or demonstrate sustained control. The preceding portable-style study addresses those separate questions.

## Equal intervention budgets

For a sparse activation direction d, let W_l contain the selected down-projection columns at layer l, and let r_l be the median residual norm on one frozen neutral passage. The primary budget is:

`B2(d) = sqrt(sum_l ||W_l d_l / r_l||_2^2)`.

Normalize each PP1/4/16 direction to B2=1, then apply g=0.1 or g=0.4. This matches aggregate requested residual-relative write strength, not the downstream effect or the number of edited layers. Edit only the last prompt position, at post-SwiGLU input to down_proj. Record every actual FP16 coordinate write and its projected norm.

This L2 normalization defines the intervention budget only. Coordinate ranking remains the original corrected signed-mean PP ranking; RMS ranking is not introduced.

Three random16 masks have the same ordered layer counts as PP16. The first mask is copied from the prior frozen random list; two additional disjoint masks are sampled with frozen seeds. Map PP discovery coefficients to the random coordinates by rank, then match each layer's requested write norm to the PP16 profile. No random mask is selected based on its effect.

As a sensitivity check at g=0.4, repeat PP1/4/16 and the first random16 under the coordinate-cost budget:

`B1(d) = sum_j |d_j| ||W[:,j]||_2 / r_layer(j)`.

This helps distinguish a conclusion that depends on within-layer cancellation or on how intervention strength is defined. Neither budget makes unequal layer locations causally interchangeable; report both definitions plainly.

## Separate additivity, dominance and deletion tolerance

At both B2 doses and in both directions, measure all 16 PP members separately at their **full-group component amplitudes**, without individually renormalizing them. Compare the group effect with the sum of these effects. Report positive or negative nonadditivity and actual FP16 rounding; do not equate positive nonadditivity with a useful semantic circuit.

At g=0.4, remove every member in turn and rescale the remaining group to the same B2 budget. This tests whether the remaining coordinates can retain the effect under a fixed rescaling rule. It is not an unscaled necessity ablation. Also test rank-1 removal without any rescaling at both doses; that arm isolates removal of the top component at its assigned amplitude. Removing a component can increase net B2 when contributions cancel, so retain its measured norm rather than assuming it always reduces total strength.

These measurements can distinguish a broad group effect from one dominated by a single member. They do not by themselves identify causal edges between the neurons.

## Protection diagnostics and outcomes

On eight frozen balanced A/B questions, test baseline and PP1/4/16 plus the first random16 under both signs and doses. Record greedy first-token correctness, correct-letter margin and distribution-wide KL. These are next-token diagnostics, not complete-answer competence tests. Retain incorrect baselines rather than silently excluding them.

For each category, direction and dose, report all paired effects, mean, median, minimum, counts above zero and counts above 0.05 logits. Report PP16 minus PP1 and PP16 minus the mean of the three random masks on each same prompt. Report the full individual-member effects, nonadditivity, and every deletion result. No validation-based arm or dose selection occurs. There is no preregistered universal-winner threshold, and no outcome-based retry or extension.

A compelling result would be greater cross-prompt consistency at a matched budget, survival of single-coordinate deletion, and limited disruption of the A/B diagnostics. A larger average effect alone is weaker evidence. Failure at these two doses does not prove that no useful group or dose exists.

## Scope and execution

The fixed total is **4,899 forward calls**, including three reference/identity checks; hard cap 5,200 calls and 25 minutes of compute, with zero backwards. The source verifies a completion-and-review receipt from the preceding portable-style experiment before launch. Both experiments use the shared lp GPU sequentially and never terminate unrelated work. A resource-only timeout remains pending; a partially executed scientific failure is preserved and is not automatically replayed.

Before launch: validate source syntax, prompt disjointness, inherited coordinate/source hashes, equal-budget geometry, random layer profiles, component amplitudes, every deletion and the exact call count. At execution: verify a multi-layer zero intervention preserves logits exactly, record actual inference PID and CUDA telemetry, seal all outputs. After completion: replay source/result hashes, condition coverage, all FP16 writes, readout arithmetic, budget accounting and baseline repeatability offline, then generate the report and figures. Full-vocabulary logits are not retained; KL is a runtime diagnostic, while target/control logits are retained for readout replay.

This study tests whether PP can identify a more consistent **group direction**, not whether it has found a new intrinsic neuron identity. A link to control-window research would require later held-out generation tests of group dose and timing. The present two-point causal assay alone does not establish a window or its boundaries.
