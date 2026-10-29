# Dose rescue, changed phenotypes, and control-window limits

The experiments apply Hongliang Liu's [control-window framework](https://arxiv.org/abs/2606.19831) to the Qwen demonstrations. The earlier report now links that framework explicitly. The source explorer already uses residual-relative doses and several strengths; its remaining limitation is shared sparse calibration, including an acknowledged unusable early-layer setting.

## A middle dose helps, but does not transfer to every prompt

On four fresh identity prompts and twelve elementary answer-only questions:

| Alien L22/N2742 dose g | Added activation offset | Frozen coherent literal identity hits | Exact answers |
| --- | ---: | ---: | ---: |
| Baseline | 0 | 0/4 | 12/12 |
| +0.25 | +15.221 | 0/4 | 12/12 |
| +0.50 | +30.442 | 2/4 | 12/12 |
| +1.00 | +60.884 | 1/4 | 4/12 |

The new half-dose result is 2/4, whereas the earlier pair of identity prompts gave 2/2. The dose is useful under some introductions and leaves an ordinary AI identity under others. At dose 1 one introduction is a clean extraterrestrial claim, while other outputs contain invented-word loops or repeated fragments. A broader lexical detector would count more alien-like wording; the frozen literal screen and the semantic examples are retained separately.

This demonstrates an operational dose window and prompt-dependent triggers. It does not show a universal per-neuron operating point. The twelve answer-only questions are a small retention screen; they cannot establish general capability preservation.

![Sampled dose responses](../dose-windows.png)

## The three other advertised identities were not cleanly recovered

Human L7/N6297, building L3/N3066, and corporation L4/N4783 received the same six-point search in reference-budget units: magnitudes 0.1, 0.25, 0.4, 0.55, 0.7, and 1, using each coordinate's public sign. Two random neurons in each target's layer received the identical search budget. Selection used two calibration introductions and three exact answers; a qualifying dose required a coherent literal target hit and all three exact answers.

None of the nine target/control searches qualified. Their declared null fallback was 0.1, which gave ordinary AI introductions on all four held-out prompts and 12/12 exact answers. This fallback is a null check, not a selected controller.

The qualitative cases show why all the gates matter:

- Human +0.55 produces a 22-year-old female identity on one prompt, followed by an impossible, repetitive biography. The primary detector misses the age/female phrase. At +0.7 a repeated student claim passes the word-bigram screen but fails the exact-answer gate. These are human-role claims, not validated coherent persona control.
- Corporation -0.4 produces an AI claim followed by company/corporation claims and a list of petroleum companies. It retains only 1/3 exact answers. Fluent surface form does not establish a consistent corporation persona.
- Building -0.4 produces heraldic descriptions rather than building self-identity. At -0.55 one calibration prompt yields heraldic prose and the other loops over thinking tags; -0.7 and -1 terminate immediately.

The [semantic review](../semantic-review.json) records those specific cases and the primary screen's false positives and misses. The negative conclusion is scoped to this protocol. A sparse dose grid, short rollout, different precision, or imperfect target detector can miss a valid window; these results do not prove that any coordinate has a genuinely closed window.

## Actual scalar values

These are additive post-SwiGLU activation offsets, applied at every prompt and generated-token position. The reference budget uses the median residual norm on the frozen mixed prose/code passage, divided by the selected down-projection column norm.

| Coordinate | Reference B | Examples of g and additive offset t = g B |
| --- | ---: | --- |
| L7/N6297 human | 19.63356 | +0.55 -> +10.79846; +0.70 -> +13.74349 |
| L3/N3066 building | 10.72111 | -0.40 -> -4.28845; -0.55 -> -5.89661; -1 -> -10.72111 |
| L4/N4783 corporation | 12.15155 | -0.40 -> -4.86062; -1 -> -12.15155 |
| L22/N2742 alien | 60.88392 | +0.50 -> +30.44196; +1 -> +60.88392 |

These values are neither weight edits nor absolute activation clamps. A donor clamp uses the measured donor value; an additive write adds the listed amount to the current activation. The [earlier methods note](../../qwen-neuron-interventions/notes/methods.md) retains the donor, ablation, singleton, and combination equations and values.

## Confirmed repair of the source-reported early-layer setting

The explorer's [October 19 log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md#L192) reports unusable first layers at +0.8 and leaves recalibration untested. We sampled two new neurons in each of layers 0, 1, and 2, then tested +0.1, +0.3, and +0.8 on an introduction, a story, and three elementary answers. These are fresh sampled coordinates and our FP16 runtime, so this reproduces the failure type rather than their exact neuron cohort.

| Layer | Reference median residual PR | At +0.3: coherence / 4 | At +0.3: exact answers / 6 | At +0.8: coherence / 4 | At +0.8: exact answers / 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.44 | 4/4 | 6/6 | 0/4 | 0/6 |
| 1 | 2.04 | 4/4 | 6/6 | 2/4 | 2/6 |
| 2 | 10.06 | 4/4 | 6/6 | 3/4 | 6/6 |

Dose +0.1 also passes all four continuation screens and all six answers in each layer. Layer-0 coordinates are N5925 and N1525. Their activation offsets change from +4.270/+4.378 at g=0.8 to +1.601/+1.642 at g=0.3. All four layer-0 high-dose continuations consist of immediate `<|im_end|>`; at the smaller dose they become ordinary AI introductions or stories.

This is a confirmed operating-setting repair. It does not establish useful semantic controllers among those random neurons. Empty output cannot support a conclusion that an early layer lacks meaningful neurons.

The participation ratio is `(sum r_j^2)^2 / sum r_j^4`, measured over the same generic passage, excluding its first token. Its small early-layer values suggest an interesting application of the paper's near-rank-one fragility argument **within Qwen across depth**, in addition to its published cross-architecture comparison. There are only three layers, two sampled directions each, and one reference passage here. No causal concentration manipulation or predictive fit of a per-layer collapse coefficient was performed. The association is exploratory.

## The geometric drive works; suppressing EOS does not repair the large dose

A separately frozen 642-forward diagnostic tests L0/N5925 at g=0, 0.3, and 0.8 on both prompts, under ordinary decoding and EOS suppression. It saves the full first-token top probabilities, raw residual vectors, and all continuations.

At g=0.8, EOS probability is 0.9204 for the introduction and 0.9479 for the story. Preventing EOS at every decode step changes immediate stopping into repeated thinking tags; neither output passes the coherence screen. Both baseline and g=0.3 remain fluent under either decoding policy. Thus EOS masking alone does not repair this failed setting.

The reference g=0.8 corresponds to context-specific couplings 0.830 and 0.842 at the first generated position, so the generic-reference difference is modest in this exemplar. Measured cosine alignments are 0.6639 and 0.6703. The full c0-aware geometric prediction agrees within **0.000238** across all six prompt/dose points. An independent local audit recomputes these quantities from the saved float32 vectors, including the approximately linear residual write despite FP16 rounding.

The result supports separating the deterministic drive from the empirical usable ceiling. It does not independently fit or refute the paper's model-average collapse coefficient. This deliberately early-layer, near-rank-one case, different coherence gate, and termination/tag-loop outcomes call for layer- and outcome-specific calibration. A large-dose failure should be typed before assigning a geometric mechanism.

## Building-coordinate follow-up: vocabulary intrusion is not a recovered skill

The calibration heraldry effect was tested on six new prompts: a garden, missing-key story, town, introduction, desk organization, and sky-color explanation. At g=-0.4, the target has heraldic vocabulary on 4/6 prompts; at -0.55, on 3/6. Baseline, target -0.25, target +0.4, and both matched random neurons at each tested negative dose have 0/6 under the same expanded marker readout.

That expanded readout is **post hoc and descriptive**. The frozen rule was too restrictive: it requires two listed terms unless a narrow explicit phrase is present, missing single occurrences of “heraldic” or “fess.” It gives 1/6 at -0.4 and 0/6 at -0.55. Both measures and every output remain in [heraldry-secondary.json](../heraldry-secondary.json). The primary protocol has not been retroactively changed.

Qualitative inspection rejects the vocabulary shift as useful domain control. Several outputs redefine the requested garden, story, or town as a heraldic charge instead of performing the task. One describes “Or” as a red cross, whereas Or denotes gold/yellow in the [National Records of Scotland terminology](https://www.nrscotland.gov.uk/learning-and-events/research-guides/coat-of-arms/?section=terminology). At -0.55 there is also an incomplete box diagram, an empty answer, and a thinking-tag loop. At -0.4 the exact-answer score is 2/3, with the third answer factually correct but in the wrong format; at -0.55 it is 0/3.

This is evidence of direction- and dose-specific lexical/topic intrusion. It recovers fluent surface text from a setting that previously terminated, but it recovers neither building self-identity nor accurate, on-task heraldry. It illustrates why an inventory title and a fluency score are insufficient behavior labels.

![Early-layer and frozen heraldry screens](../followup-windows.png)

## Bounded human refinement and scoring limits

The additional human doses +0.60 and +0.65 also received the same two-point search for both random controls. No condition qualified, so the protocol launched no selected-dose held-out human trial. The target keeps an AI identity on one calibration prompt and produces repetitive name/age/nationality claims on the other.

The exact-answer scores are 1/3 and 0/3, but all three expected answers remain present at both doses: these particular losses are answer-format/compliance failures, not demonstrations of lost factual knowledge. By contrast, alien g=1 gives genuinely wrong held-out answers such as `18 - 7 = 5`, `8 times 4 -> 54`, `capital of Italy -> Alien`, and `100 - 25 -> 85`. Keeping this distinction avoids using a format gate as a knowledge-loss claim.

The word-bigram gate also admits some repetitive or contradictory biographies. Human age/female claims can fall outside the frozen literal detector. The raw outputs and qualitative review are therefore essential; neither a zero marker score nor a positive lexical-coherence score settles semantic control.

## What this adds and what it does not

The control-window explanation, off-axis finite-dose gates, rollout horizons, and the need to calibrate combinations are already in Liu's paper. The new measurements here are the early-layer operating-setting repair, the first-token/decoding diagnostic with a quantitative drive check, fresh prompt dependence of the alien effect, and the different phenotype exposed by the building coordinate at intermediate negative doses.

The practical method is to define a behavior and its horizon, measure reference budgets and concentration, bracket coherent doses separately on each signed coordinate, select against task and competence gates on calibration data, and then test fresh prompts. Give random controls the same dose-search budget. For combined edits, check each signed component and the combination: the earlier alien triplet already shows that an antagonist can reduce transfer, and similar scalar readouts need not produce similar continuations. A singleton's operating window cannot simply be copied to a joint intervention.

The source's early-layer setting is a confirmed calibration problem under our reproduction. Its other advertised persona results are not established to be wrong-dose failures by this study. The Persona Library's 0.5 lower limit is a plausible source of missed windows, but no specific Qwen3-8B library entry was rerun here. A model-wide inventory false-negative rate and a general closed-window claim remain unmeasured.

## Reproducibility

Three sequential detached jobs completed with exit code 0 on `lp`: 8,391 main forwards, 5,735 follow-up forwards, and 642 diagnostic forwards, for **14,768 forwards and 1,778.6 seconds of model-job time**. The local bundle includes 284 continuations and 432 exact-answer trials. No model weights were downloaded and no paid API was called.

All 32 transformer layers use CUDA FP16; the tied embedding/output vocabulary boundary uses CPU FP32 promoted from the FP16 copy. This is the same pinned runtime as the earlier experiments, with Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It differs from public BF16/vLLM demonstrations. Correct reference PyTorch kernel notices remain in the reviewed stderr.

The [offline audit](../audit.json) verifies protocol/source hashes, equal search budgets, calibration-only selections, exact-answer arithmetic, condition denominators, geometry, exit codes, and 99/67/8 PID-correlated CUDA telemetry samples. It does not provide an independent blinded semantic judge. The small control cohorts, imperfect lexical screens, 64/96-token horizons, and precision/runtime difference limit the claims.

- [Theory and source audit](theory-and-source-audit.md)
- [Primary frozen protocol](../source/protocol.json)
- [Follow-up frozen protocol](../source/followups/protocol.json)
- [EOS diagnostic frozen protocol](../source/eos/protocol.json)
- [Dose results CSV](../dose-results.csv)
- [Early-layer summary](../early-layer-summary.json)
- [Qualitative review and screening limitations](../semantic-review.json)
- [Every main continuation](../evidence/main-v1/results/generations.json)
- [Every follow-up continuation](../evidence/followups-v1/results/generations.json)
- [EOS distributions and continuations](../evidence/eos-v1/results/distributions.json)
