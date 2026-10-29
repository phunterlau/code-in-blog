# Additional diagnostics from the frozen group run

These are descriptive summaries after the run. They do not select a new neuron, group, or recipe for a confirmatory test. All comparisons use the already audited outputs.

## Effect magnitude under two budget definitions

Positive means the intended direction. Both budgets use the numerical dose 0.4, but measure different physical costs; they are not interchangeable doses.

| Category | Direction | Budget | PP1 mean | PP4 mean | PP16 mean | Random16 mean |
|---|---|---|---:|---:|---:|---:|
| losing_language | base | B2 | -0.086 | +0.846 | +2.114 | +0.678 |
| losing_language | base | B1 | -0.086 | +0.532 | +0.678 | +0.169 |
| losing_language | target | B2 | +0.159 | +0.741 | +1.681 | +0.305 |
| losing_language | target | B1 | +0.159 | +0.463 | +0.513 | +0.034 |
| academic | base | B2 | +4.406 | +5.564 | +4.592 | +0.726 |
| academic | base | B1 | +4.406 | +2.907 | +1.518 | +0.203 |
| academic | target | B2 | +3.136 | +4.658 | +3.955 | +1.218 |
| academic | target | B1 | +3.136 | +3.148 | +1.181 | +0.340 |
| storyteller | base | B2 | +2.523 | +3.003 | +4.107 | +0.397 |
| storyteller | base | B1 | +2.523 | +2.265 | +1.630 | +0.145 |
| storyteller | target | B2 | +3.092 | +3.838 | +4.895 | +0.321 |
| storyteller | target | B1 | +3.092 | +2.706 | +1.288 | +0.082 |
| nostalgic | base | B2 | +2.140 | +4.158 | +4.872 | -0.516 |
| nostalgic | base | B1 | +2.140 | +3.336 | +1.703 | -0.144 |
| nostalgic | target | B2 | +3.846 | +8.105 | +5.822 | +0.022 |
| nostalgic | target | B1 | +3.846 | +5.219 | +1.559 | -0.069 |

## Removing the top-ranked member without rescaling

| Category | Direction | Dose | Full PP16 mean | Without rank 1 mean | Change | Remaining B2 | Effects > .05 |
|---|---|---:|---:|---:|---:|---:|---:|
| losing_language | base | 0.1 | +0.4929 | +0.5116 | +0.0186 | 0.0790 | 8/8 |
| losing_language | base | 0.4 | +2.1140 | +2.2382 | +0.1242 | 0.3162 | 8/8 |
| losing_language | target | 0.1 | +0.3710 | +0.3428 | -0.0282 | 0.0790 | 8/8 |
| losing_language | target | 0.4 | +1.6811 | +1.5322 | -0.1489 | 0.3162 | 8/8 |
| academic | base | 0.1 | +1.2282 | +0.9436 | -0.2846 | 0.0972 | 8/8 |
| academic | base | 0.4 | +4.5918 | +3.5278 | -1.0640 | 0.3887 | 8/8 |
| academic | target | 0.1 | +0.9439 | +0.6166 | -0.3273 | 0.0972 | 8/8 |
| academic | target | 0.4 | +3.9554 | +2.7610 | -1.1944 | 0.3887 | 8/8 |
| storyteller | base | 0.1 | +1.0920 | +0.6557 | -0.4362 | 0.0924 | 8/8 |
| storyteller | base | 0.4 | +4.1067 | +3.1336 | -0.9731 | 0.3695 | 8/8 |
| storyteller | target | 0.1 | +0.7876 | +0.6343 | -0.1533 | 0.0924 | 8/8 |
| storyteller | target | 0.4 | +4.8951 | +3.4281 | -1.4670 | 0.3695 | 8/8 |
| nostalgic | base | 0.1 | +1.3158 | +1.0292 | -0.2866 | 0.0958 | 8/8 |
| nostalgic | base | 0.4 | +4.8715 | +4.1046 | -0.7669 | 0.3831 | 8/8 |
| nostalgic | target | 0.1 | +1.1508 | +0.9006 | -0.2502 | 0.0958 | 8/8 |
| nostalgic | target | 0.4 | +5.8225 | +4.3102 | -1.5123 | 0.3831 | 8/8 |

## Individual components at group-assigned amplitudes

The largest member below is identified descriptively after seeing these outcomes; it is not independently validated as an optimal singleton. Members do not receive a full normalized group dose.

| Category | Direction / dose | Largest mean component | Mean component sum | Mean group effect | Nonadditivity |
|---|---|---|---:|---:|---:|
| losing_language | base:0.1 | rank 3: L31 N66 | +0.4924 | +0.4929 | +0.0005 |
| losing_language | base:0.4 | rank 3: L31 N66 | +2.1426 | +2.1140 | -0.0286 |
| losing_language | target:0.1 | rank 3: L31 N66 | +0.3583 | +0.3710 | +0.0127 |
| losing_language | target:0.4 | rank 3: L31 N66 | +1.5793 | +1.6811 | +0.1018 |
| academic | base:0.1 | rank 1: L25 N3530 | +1.2371 | +1.2282 | -0.0089 |
| academic | base:0.4 | rank 1: L25 N3530 | +4.7034 | +4.5918 | -0.1116 |
| academic | target:0.1 | rank 1: L25 N3530 | +0.9090 | +0.9439 | +0.0349 |
| academic | target:0.4 | rank 1: L25 N3530 | +4.0105 | +3.9554 | -0.0551 |
| storyteller | base:0.1 | rank 1: L26 N2167 | +1.0646 | +1.0920 | +0.0274 |
| storyteller | base:0.4 | rank 1: L26 N2167 | +4.7356 | +4.1067 | -0.6289 |
| storyteller | target:0.1 | rank 3: L21 N6889 | +0.7128 | +0.7876 | +0.0748 |
| storyteller | target:0.4 | rank 3: L21 N6889 | +3.8281 | +4.8951 | +1.0670 |
| nostalgic | base:0.1 | rank 1: L22 N8168 | +1.3142 | +1.3158 | +0.0016 |
| nostalgic | base:0.4 | rank 1: L22 N8168 | +5.3453 | +4.8715 | -0.4738 |
| nostalgic | target:0.1 | rank 1: L22 N8168 | +1.0394 | +1.1508 | +0.1114 |
| nostalgic | target:0.4 | rank 1: L22 N8168 | +4.9175 | +5.8225 | +0.9050 |

## Exact first-token protection changes

Every retained protection argmax is A or B, verified against the saved tokenizer IDs. There are no hidden format-token substitutions in these diagnostics. Baseline is 7/8: it incorrectly chooses A (zebra) instead of B (apple) on the alphabetical-order question.

| Category | Arm / signed dose | Question | Before → after | Result |
|---|---|---|---|---|
| academic | pp4 / -0.4 | A box contains exactly three coins. A means two coins; B means three coins. Output only the true letter. | B → A | new error |
| storyteller | random16_0 / +0.4 | Choose the word that comes first alphabetically: A means zebra; B means apple. Output only the letter. | A → B | corrected |
| nostalgic | random16_0 / -0.4 | A box contains exactly three coins. A means two coins; B means three coins. Output only the true letter. | B → A | new error |

PP16 leaves every individual A/B prediction unchanged in all four categories at both signs and doses. Academic PP4 at −0.4 adds one error. Random groups are not inert: storytelling random16 at +0.4 corrects the baseline alphabetical error, while nostalgia random16 at −0.4 adds one error. These are eight-question next-token diagnostics only.

![Budget sensitivity](../figures/budget-sensitivity.png)

![Deletion retention](../figures/deletion-retention.png)

See [all frozen-arm results](results.md), [machine-readable supplementary summaries](../secondary-analysis.json), and [the protocol](protocol.md).
