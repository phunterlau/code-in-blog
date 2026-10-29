# PP group controls — measured results

**The group-consistency claim survives a stricter test for the loss-of-language readout. It is not a universal advantage across all four categories.** The experiment completed on lp with 4,899 forward calls, zero backwards, 675 seconds of compute, and all integrity checks passing. It retains the original signed-mean PP ranking. L2 and L1 below define intervention cost only; neither is a replacement ranking rule.

Eight fresh requests were tested in each category using fixed directions estimated from the original discovery pairs. No new test request supplied donor activations. At matched aggregate block-L2 write strength, the loss-of-language singleton cleared the frozen 0.05-logit threshold on 0/8 base prompts at dose 0.1 and 1/8 at dose 0.4; top-4 and top-16 cleared it on 8/8 at both doses. Reversing the edit on target-style prompts gave singleton counts of 3/8 and 6/8, versus 8/8 for both groups. The group advantage also survives the separate coordinate-L1 cost check in this category.

The other three categories already have 8/8 threshold-positive singleton effects under this intervention. They show differences in effect magnitude, not an improvement in this particular consistency count. The academic singleton's earlier weakness does not reproduce here. The old experiment used donor restoration; this one changes coefficients, normalization and prompts together, so that change cannot be attributed to dose alone.

The new findings are:

- **The highest PP rank need not be the best control member.** For loss of language, deleting rank 1 (L26 N1) without rescaling increases the base-prompt mean from +2.114 to +2.238 logits at nominal dose 0.4, while reducing its actual requested B2 budget from 0.400 to 0.316. This is an observed mean improvement; it is not a newly validated optimized recipe. Rank 3 (L31 N66) has the largest mean effect among the separately tested members at their group-assigned amplitudes. PP provides candidates; causal validation still matters.
- **Every group tolerates single-member deletion in this assay.** After removing each of the 16 members and rescaling the remainder to B2=0.4, every category and direction retains effects above 0.05 on 8/8 prompts. Median retention across prompts and deletions is 96–98%; the weakest individual retained effect is 66.7% of its full-group value. This tests retained control capacity under rescaling, not unscaled necessity or an identified circuit. Even the separate unscaled rank-1 deletions remain above threshold on 8/8 throughout.
- **PP selection matters beyond group size.** PP16 exceeds the mean of three layer-matched random groups by more than 0.05 logits on every paired prompt in every tested category/direction/dose. Random groups themselves sometimes produce consistent effects, so positive direction alone is insufficient evidence of specificity. The random controls match requested per-layer write norms under B2.
- **The useful group size depends on the intervention budget.** For storytelling base prompts at numerical dose 0.4, B2 matching gives PP16 +4.107 versus PP1 +2.523. Under coordinate-L1 matching, PP16 gives +1.630 versus PP1 +2.523. For nostalgia under L1, PP4 is strongest among the tested PP arms in both directions: +3.336 inducing and +5.219 suppressing, versus singleton +2.140 and +3.846. Equal numerical doses under these two budgets are different physical interventions. Neither definition equalizes downstream gain across layers.
- **Interaction effects depend on direction and dose.** At B2=0.4, storytelling's group-minus-component-sum is −0.629 when inducing and +1.067 when suppressing its readout; nostalgia gives −0.474 and +0.905. At smaller doses the deviations are generally smaller. These are finite-intervention nonadditivity measurements, not proof of useful semantic synergy or causal connections between members.

PP16 leaves every individual A/B protection prediction unchanged at both signs and doses in all four categories, retaining the baseline's 7/8 accuracy. Academic PP4 at −0.4 adds one incorrect answer; one random control corrects an existing error and another adds one. These small next-token diagnostics do not establish full-answer competence. [Exact changes, budget comparisons and component diagnostics](diagnostics.md) are retained alongside all predefined arms below.

![Matched-budget group effects](../figures/group-effects.png)

![Retention after individual deletions](../figures/deletion-retention.png)

The broader value is evidence that PP can supply sparse groups with reliable, distributed control over a chosen readout, even when its top singleton is weak. It also supplies counterexamples to a universal largest-group or top-ranked-neuron rule. These results do not demonstrate coherent loss of language, academic prose, storytelling or nostalgia: no text was generated in this extension. The preceding [portable generation study](../../qwen-pp-portable-styles/notes/results.md) found no qualified storytelling or nostalgia singleton recipe in its frozen grid. Strong first-token effects and portable semantic control must remain separate findings.

For the control-window research, the next hypothesis is that useful dose ranges depend jointly on group composition, intervention schedule and prompt family. A follow-up would need held-out generation with semantic and quality scoring, matched random controls, multiple doses and protection tests, including explicit style prompts as assay controls. The two dose points here do not establish a useful semantic window or its boundaries. That follow-up is a proposal; no additional experiment was launched.

The tables below were produced by the frozen analyzer. Supplementary descriptions and figures use only those audited results; their offline source is retained separately from the launch-sealed code.

Every condition was fixed before these model outputs. Effects below are oriented toward the intended change: positive for a base-prompt increase and positive for a target-prompt decrease. These are single-position lexical readouts, not complete response quality.

| Category | Source | Dose | PP1 > .05 | PP4 > .05 | PP16 > .05 | Mean PP16 minus PP1 | Mean nonadditivity |
|---|---|---:|---:|---:|---:|---:|---:|
| losing_language | base | 0.1 | 0/8 | 8/8 | 8/8 | +0.5167 | +0.0005 |
| losing_language | base | 0.4 | 1/8 | 8/8 | 8/8 | +2.1996 | -0.0286 |
| losing_language | target | 0.1 | 3/8 | 8/8 | 8/8 | +0.3313 | +0.0127 |
| losing_language | target | 0.4 | 6/8 | 8/8 | 8/8 | +1.5221 | +0.1018 |
| academic | base | 0.1 | 8/8 | 8/8 | 8/8 | +0.0399 | -0.0089 |
| academic | base | 0.4 | 8/8 | 8/8 | 8/8 | +0.1860 | -0.1116 |
| academic | target | 0.1 | 8/8 | 8/8 | 8/8 | -0.3900 | +0.0349 |
| academic | target | 0.4 | 8/8 | 8/8 | 8/8 | +0.8195 | -0.0551 |
| storyteller | base | 0.1 | 8/8 | 8/8 | 8/8 | -0.0817 | +0.0274 |
| storyteller | base | 0.4 | 8/8 | 8/8 | 8/8 | +1.5838 | -0.6289 |
| storyteller | target | 0.1 | 8/8 | 8/8 | 8/8 | +0.4231 | +0.0748 |
| storyteller | target | 0.4 | 8/8 | 8/8 | 8/8 | +1.8030 | +1.0670 |
| nostalgic | base | 0.1 | 8/8 | 8/8 | 8/8 | +0.1831 | +0.0016 |
| nostalgic | base | 0.4 | 8/8 | 8/8 | 8/8 | +2.7314 | -0.4738 |
| nostalgic | target | 0.1 | 8/8 | 8/8 | 8/8 | +0.2502 | +0.1114 |
| nostalgic | target | 0.4 | 8/8 | 8/8 | 8/8 | +1.9763 | +0.9050 |

PP1/4/16 receive the same aggregate block-L2 write budget at each dose. This does not equalize all downstream effects or coordinate-L1 cost. Three random masks match PP16 layer counts and requested per-layer write norms. The separate coordinate-L1 arm checks sensitivity to the definition of intervention strength. All masks and individual-component measurements are retained in analysis.json.

## Random-group and budget-definition checks

| Category | Source | Dose | PP16 minus mean random effect | Paired advantages > .05 |
|---|---|---:|---:|---:|
| losing_language | base | 0.1 | +0.4267 | 8/8 |
| losing_language | base | 0.4 | +1.6741 | 8/8 |
| losing_language | target | 0.1 | +0.3406 | 8/8 |
| losing_language | target | 0.4 | +1.3567 | 8/8 |
| academic | base | 0.1 | +1.1405 | 8/8 |
| academic | base | 0.4 | +4.1367 | 8/8 |
| academic | target | 0.1 | +0.7263 | 8/8 |
| academic | target | 0.4 | +2.8172 | 8/8 |
| storyteller | base | 0.1 | +1.0281 | 8/8 |
| storyteller | base | 0.4 | +3.8613 | 8/8 |
| storyteller | target | 0.1 | +0.7814 | 8/8 |
| storyteller | target | 0.4 | +4.8157 | 8/8 |
| nostalgic | base | 0.1 | +1.4040 | 8/8 |
| nostalgic | base | 0.4 | +5.2675 | 8/8 |
| nostalgic | target | 0.1 | +1.2153 | 8/8 |
| nostalgic | target | 0.4 | +5.7178 | 8/8 |

Equal coordinate-L1 budget at g=0.4:

| Category | Source | PP1 > .05 | PP4 > .05 | PP16 > .05 | Random16 > .05 |
|---|---|---:|---:|---:|---:|
| losing_language | base | 1/8 | 8/8 | 8/8 | 8/8 |
| losing_language | target | 6/8 | 8/8 | 8/8 | 3/8 |
| academic | base | 8/8 | 8/8 | 8/8 | 8/8 |
| academic | target | 8/8 | 8/8 | 8/8 | 8/8 |
| storyteller | base | 8/8 | 8/8 | 8/8 | 8/8 |
| storyteller | target | 8/8 | 8/8 | 8/8 | 7/8 |
| nostalgic | base | 8/8 | 8/8 | 8/8 | 0/8 |
| nostalgic | target | 8/8 | 8/8 | 8/8 | 0/8 |

## Deletion robustness at dose 0.4

Each deleted group is rescaled to the original aggregate budget. This tests whether the remaining coordinates can retain the effect with a predetermined rescaling; it is not a pure necessity ablation. The additional unscaled rank-1 deletion is available separately.

| Category | Source | Worst deletion: effects > .05 | Median retention across eligible cases/deletions |
|---|---|---:|---:|
| losing_language | base | 8/8 | 0.960 |
| losing_language | target | 8/8 | 0.967 |
| academic | base | 8/8 | 0.973 |
| academic | target | 8/8 | 0.962 |
| storyteller | base | 8/8 | 0.977 |
| storyteller | target | 8/8 | 0.978 |
| nostalgic | base | 8/8 | 0.964 |
| nostalgic | target | 8/8 | 0.965 |

## Protection diagnostics

These are greedy first-token choices and margins for eight balanced A/B tasks, not full-answer competence certification. No generation was performed in this extension.

| Category | Condition | Correct first token | Mean correct-letter margin |
|---|---|---:|---:|
| losing_language | baseline:+0 | 7/8 | +4.7443 |
| losing_language | pp1:+0.1 | 7/8 | +4.7331 |
| losing_language | pp4:+0.1 | 7/8 | +4.7398 |
| losing_language | pp16:+0.1 | 7/8 | +4.7296 |
| losing_language | random16_0:+0.1 | 7/8 | +4.6929 |
| losing_language | pp1:-0.1 | 7/8 | +4.7601 |
| losing_language | pp4:-0.1 | 7/8 | +4.7415 |
| losing_language | pp16:-0.1 | 7/8 | +4.7392 |
| losing_language | random16_0:-0.1 | 7/8 | +4.7791 |
| losing_language | pp1:+0.4 | 7/8 | +4.6955 |
| losing_language | pp4:+0.4 | 7/8 | +4.6712 |
| losing_language | pp16:+0.4 | 7/8 | +4.5792 |
| losing_language | random16_0:+0.4 | 7/8 | +4.4729 |
| losing_language | pp1:-0.4 | 7/8 | +4.8149 |
| losing_language | pp4:-0.4 | 7/8 | +4.6752 |
| losing_language | pp16:-0.4 | 7/8 | +4.5945 |
| losing_language | random16_0:-0.4 | 7/8 | +4.7784 |
| academic | baseline:+0 | 7/8 | +4.7443 |
| academic | pp1:+0.1 | 7/8 | +4.7694 |
| academic | pp4:+0.1 | 7/8 | +4.7199 |
| academic | pp16:+0.1 | 7/8 | +4.7572 |
| academic | random16_0:+0.1 | 7/8 | +4.7470 |
| academic | pp1:-0.1 | 7/8 | +4.7393 |
| academic | pp4:-0.1 | 7/8 | +4.7832 |
| academic | pp16:-0.1 | 7/8 | +4.7231 |
| academic | random16_0:-0.1 | 7/8 | +4.7399 |
| academic | pp1:+0.4 | 7/8 | +4.9409 |
| academic | pp4:+0.4 | 7/8 | +4.7170 |
| academic | pp16:+0.4 | 7/8 | +4.7650 |
| academic | random16_0:+0.4 | 7/8 | +4.7214 |
| academic | pp1:-0.4 | 7/8 | +4.8033 |
| academic | pp4:-0.4 | 6/8 | +4.8713 |
| academic | pp16:-0.4 | 7/8 | +4.6282 |
| academic | random16_0:-0.4 | 7/8 | +4.7245 |
| storyteller | baseline:+0 | 7/8 | +4.7443 |
| storyteller | pp1:+0.1 | 7/8 | +4.6615 |
| storyteller | pp4:+0.1 | 7/8 | +4.7126 |
| storyteller | pp16:+0.1 | 7/8 | +4.6925 |
| storyteller | random16_0:+0.1 | 7/8 | +4.7327 |
| storyteller | pp1:-0.1 | 7/8 | +4.8284 |
| storyteller | pp4:-0.1 | 7/8 | +4.7639 |
| storyteller | pp16:-0.1 | 7/8 | +4.7885 |
| storyteller | random16_0:-0.1 | 7/8 | +4.7423 |
| storyteller | pp1:+0.4 | 7/8 | +4.4585 |
| storyteller | pp4:+0.4 | 7/8 | +4.5178 |
| storyteller | pp16:+0.4 | 7/8 | +4.4653 |
| storyteller | random16_0:+0.4 | 8/8 | +4.6287 |
| storyteller | pp1:-0.4 | 7/8 | +5.0840 |
| storyteller | pp4:-0.4 | 7/8 | +4.7884 |
| storyteller | pp16:-0.4 | 7/8 | +4.8777 |
| storyteller | random16_0:-0.4 | 7/8 | +4.6355 |
| nostalgic | baseline:+0 | 7/8 | +4.7443 |
| nostalgic | pp1:+0.1 | 7/8 | +4.7506 |
| nostalgic | pp4:+0.1 | 7/8 | +4.7478 |
| nostalgic | pp16:+0.1 | 7/8 | +4.7708 |
| nostalgic | random16_0:+0.1 | 7/8 | +4.7406 |
| nostalgic | pp1:-0.1 | 7/8 | +4.7482 |
| nostalgic | pp4:-0.1 | 7/8 | +4.7533 |
| nostalgic | pp16:-0.1 | 7/8 | +4.7179 |
| nostalgic | random16_0:-0.1 | 7/8 | +4.7400 |
| nostalgic | pp1:+0.4 | 7/8 | +4.8208 |
| nostalgic | pp4:+0.4 | 7/8 | +4.7222 |
| nostalgic | pp16:+0.4 | 7/8 | +4.8730 |
| nostalgic | random16_0:+0.4 | 7/8 | +4.6669 |
| nostalgic | pp1:-0.4 | 7/8 | +4.7640 |
| nostalgic | pp4:-0.4 | 7/8 | +4.7773 |
| nostalgic | pp16:-0.4 | 7/8 | +4.6397 |
| nostalgic | random16_0:-0.4 | 6/8 | +4.6819 |

## Interpretation limits

Nonadditivity compares the combined group with the sum of sixteen separately measured components at their group-assigned nominal amplitudes. Report its sign and size; a positive deviation alone is not proof of a reusable semantic circuit or useful synergy. Actual FP16 writes and rounding are audited. The same eight authored requests are reused across four categories, so the 32 pairs are not 32 independent task families. Categories were selected after the earlier results to include weak and strong singleton cases; this is a new test of a post-hoc hypothesis. Two dose points cannot establish the boundaries of a control window.

Readouts are recomputed from retained target/control logits. Full vocabulary logits are not retained, so distribution-wide KL scalars are runtime diagnostics rather than independently replayed evidence.

Runtime: 4899 forward calls, zero backwards; CUDA PID 1335697; 67 correlated telemetry samples. All integrity checks passed. See [protocol](protocol.md), [analysis](../analysis.json), and [verification](../verification.json).

