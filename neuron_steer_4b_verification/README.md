# Signed-mean PP: zero, egg and verification neurons in Qwen3.5-4B

The experiments investigate three behavior labels. **Zero** means a tendency to answer `0` on arithmetic prompts, including prompts whose correct answer is nonzero. **Egg** means introducing egg-related words into a requested sentence while retaining the user’s specified word. **Verification** means assessing a proposed arithmetic claim, including whether it is correct and what the answer should be. These are experimental targets, not claims that a single neuron exclusively controls each behavior.

For each target, **public** identifies a previously reported Neuron Explorer coordinate, **PP** identifies a candidate nominated by perturbation probing, and **sweep** identifies a candidate selected from a random sample of neurons. The experiment tests whether these candidates produce useful effects at different doses and timings while preserving other task answers.

This study applies [Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs](https://arxiv.org/abs/2604.27401). It uses **signed-mean PP, ranked by descending signed value**. RMS is recorded only as a diagnostic comparison and never selects PP candidates. The signed-mean formulation follows the author's correction of the RMS transcription error in v1 of the PP preprint.

The completed main study used 6,761 forward calls, zero backward passes and 880.55 seconds on lp, my gaming laptop with an NVIDIA GeForce RTX 2070 Max-Q and 8 GB of GPU memory. It rediscovered the published verification neuron near the top of the ranking and found additional causal candidates. It did not establish reliable arithmetic repair or a generally useful verification control. Separate exploratory donor-patching and egg-dose refinement results are linked below when completed.

## Method and scope

For paired base and target prompts, the ranking score is

$$s_{\ell n}=\frac{1}{K}\sum_{k=1}^{K}c_{\ell n}(a^{\mathrm{target}}_{k\ell n}-a^{\mathrm{base}}_{k\ell n}),\qquad c_{\ell n}=d^\top W_{\mathrm{down},\ell}[:,n].$$

Here $d$ is the difference between mean output-embedding vectors for the specified target and control token sets. The score retains direction and cancellation. The saved RMS score measures magnitude and may rank a consistently opposing contribution highly. This direct readout coupling is a nomination proxy; downstream propagation can alter the causal sign or effect size.

Each task used eight authored contrast pairs. All three rankings were sealed before the program read public coordinates. The investigator already knew the published examples, so the appropriate claim is coordinate-excluded algorithmic selection. The screen tested the top four positive signed PP candidates and the published coordinate at eight signed doses, plus 32 fixed random coordinates at two doses. This is not an equal-budget benchmark or a full replication of Neuron Explorer's sweep-and-rank pipeline.

Additive edits act on post-SwiGLU MLP coordinates. The activation offset is

$$\Delta a_{\ell n}=g\,\frac{\mathrm{median}_t\|h_{\ell,t}\|_2}{\|W_{\mathrm{down},\ell}[:,n]\|_2}.$$

The residual scale comes from a fixed reference passage. These relative doses cannot be read as raw activation values or imported unchanged into another runtime. Main generation calibration tested the winning screen dose and half that dose with continuous steering, a four-prediction pulse, or prefill-only steering. Every schedule edits all prefill positions. Releasing removes subsequent edits but does not reset the intervened cache state.

Choices were sealed before eight held-out target prompts and eight protection prompts. No edit remained valid. This is an empirical dose/timing study; it does not estimate the control-window theorem's cosine geometry or prove the theory's quantitative bounds.

The model is checkpoint revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with CUDA FP16 transformer layers and a CPU FP32 tied vocabulary boundary, greedy decoding and thinking disabled. These results are specific to that runtime. Token caps were 48 for prose and verification, 8 for zero tasks, and 12 for protection tasks. Capped responses remain failures; none were retried.

## Rediscovery

| Public label | Coordinate | Signed-mean rank / 294,912 | RMS diagnostic rank |
|---|---|---:|---:|
| Zero | L26 N4405 | 205 | 529 |
| Egg | L23 N7748 | 294,901 | 11 |
| Verification | L22 N4114 | 3 | 6 |

The verification result is direct rediscovery using the preferred signed-mean algorithm. Zero is positive but outside the tested top-four PP pool. The egg coordinate is 12th from the bottom by signed mean, with six negative and two positive pair contributions. Its low descending rank does not establish irrelevance. It expresses an opposing direct contribution on these particular contrasts. The pairs also differ in food context, which limits interpretation as a clean egg-only feature.

PP's screened candidates were L28 N596 for zero, L25 N7481 for egg and L22 N4114 for verification. The verification PP and public arms are the same coordinate and are not independent confirmations.

## Zero: causal bias with inadequate arithmetic specificity

| Arm | Selected recipe | Correct zero cases | Correct nonzero cases | All target accuracy | Protection accuracy |
|---|---|---:|---:|---:|---:|
| Baseline | No edit | 0/4 | 0/4 | 0/8 | 7/8 |
| PP | L28 N596, −0.2, prefill only | 3/4 | 0/4 | 3/8 | 7/8 |
| Public | L26 N4405, +0.2, prefill only | 4/4 | 0/4 | 4/8 | 7/8 |
| Random sweep | L12 N5082, +0.2, pulse4 | 1/4 | 0/4 | 1/8 | 7/8 |

For `A previous draft gives 7. Independently compute (46+27)-(35+38). Return only the integer answer.`, baseline answered `7`; both PP and public interventions answered `0`.

The public intervention answered `0` on all eight cases, including all four nonzero expressions. PP answered `0` on four cases, including the case whose correct answer was `3`. Both preserve the seven baseline-correct protection answers, but neither establishes arithmetic repair. Baseline target accuracy was already 0/8, making target-correctness retention vacuous. The frozen automated eligibility label therefore overstates suitability for a practical arithmetic controller; absolute accuracy and nonzero specificity must be examined alongside retention.

Suppression of PP's L28 N596 is directionally consistent with its negative activation change and negative coupling. A positive PP score does not mean the intervention should always boost activation.

## Egg: token leverage and a dose/timing boundary

PP's L25 N7481 is rank 1 by signed mean. Both it and the public L23 N7748 have strong positive effects on the egg-token readout when boosted. At +0.4 continuous steering, neither calibration prompt became egg-themed. At +0.8 continuous steering, one of two prompts devolved into repeated `egg`, losing the requested word and reaching the token cap. Releasing after four predictions prevented the observed repetition but also removed the desired theme. No arm passed the main generation calibration; all selected no edit.

On the eight main held-out prose prompts, no edit and each separately reported failed challenger produced 0/8 joint egg/content successes. Required words were retained in 8/8. The automatic content criterion checks a required word, minimum length and bigram diversity; it is not a complete semantic quality metric.

This motivated the separately frozen intermediate-dose follow-up, whose new prompts and full results are in `window/`. It does not change the main-study result or retrospectively convert failed recipes into selected recipes.

## Verification and a sweep candidate

| Arm | Diagnostic recipe | Correct final ANSWER | Correct answer plus verification marker | Protection accuracy |
|---|---|---:|---:|---:|
| Baseline | No edit | 4/8 | 0/8 | 7/8 |
| PP / public | L22 N4114, +0.4 continuous | 5/8 | 0/8 | 8/8 |
| Random sweep | L12 N7836, +0.2 prefill only | 6/8 | 0/8 | 7/8 |

Both intervention rows are failed calibration challengers, not selected policies. All selected policies for verification were no edit. The PP/public challenger removed two capped baseline responses, but this is a small exploratory observation rather than an established reasoning gain. The extra correct number for `9*8=72` still came with the contradictory statement: `The draft is incorrect because 9 multiplied by 8 equals 72, which is actually the correct result. ANSWER=72`. Final-number correctness is therefore insufficient to establish correct verification.

The random-sweep L12 N7836 is worth investigating independently: it improved final-answer counts without the target vocabulary. At +0.4 in calibration it also endorsed the false `17+24=42` claim. Its effects plausibly involve agreement or response framing, a hypothesis requiring fresh balanced true/false claim tests and explicit verdict scoring. It is not yet a verified “truth” or “verification” neuron.

## Novelty, attribution and limitations

The tested PP zero and egg candidates, and all three random-sweep winners, were absent from the 180 unique coordinates in the archived inventory and source-label checks. This is absence from checked sources, not a claim of worldwide priority. Source filenames and hashes are in `analysis.json`. Top-four donor groups are tested only on discovery contrasts, so they cannot support claims of general group-control superiority.

Published coordinates and inspiration are credited to [Neuron Explorer, model-agnostic branch](https://github.com/sambhavnoobcoder/neuron-explorer/tree/0f3484031bccb061081207559183517942bccc76), the archived Neuronary recipes and the [PEFT discussion](https://github.com/huggingface/peft/pull/4066#issuecomment-6577425151). Prompts in this study are authored contrasts and evaluations, not verbatim source recipe replications. The referenced comment discusses additional models and recipes; this 4B study is not an exact replication of those newer claims. Source tooling already supports contextual interpretation and dose exploration, so these experiments do not establish that its authors missed the importance of dose.

The useful conclusion is a separation of candidate ranking, causal token influence and successful task-preserving behavior. Signed mean successfully nominated a published verification coordinate and additional causal candidates. Several interventions with large readout effects failed generation or specificity checks. Small denominators, prompt-dependent effects, weak baseline competence on the distractor arithmetic cohort and unmeasured semantic verdict accuracy limit stronger claims.

## Reproducible evidence

[Raw prompts, outputs and main calibration grid](notes/raw-examples.md), [frozen protocol](notes/protocol.md), [audit results](analysis.json), [main runner](source/run_study.py), [main raw JSONs](evidence/results/heldout.json), and [note index](notes/index.md) are local artifacts. The offline audit recomputes signed means from saved tensors, verifies ranking order, selection commitments, source hashes, metrics, actual float32-recorded FP16 write deltas, GPU PID evidence and exact forward-call accounting. A rounding assertion in the offline auditor was corrected to reproduce the runtime's float32 subtraction; no model-run source or result was changed.

Run `python3 source/analyze.py` and `python3 source/report.py` from this bundle's directory to regenerate the audit and raw-example report. The novelty check uses sibling archived inventories when present, and thus requires those files to reproduce its exact count. No commit, push or public issue comment was made in this task.

## Completed exploratory follow-ups

The independently queued donor patch completed 352 forwards. On the eight discovery pairs, replacing the base activation with the target activation for L25 N7481 increased the egg-token gap in 8/8 pairs (mean +7.03). For the public L23 N7748 it decreased that gap on average (−0.084, positive in 2/8), consistent with its negative signed score in these contrasts. For the verification coordinate L22 N4114 the same patch increased its gap in 8/8 (mean +0.907). These are within-discovery, first-token causal effects, not independent generation validation or proof of a unique feature.

The subsequent egg refinement completed 914 forwards. It was designed after observing main calibration and is exploratory. Its selection was sealed before four fresh breakfast-related confirmation prompts. L25 N7481 at **+0.6 continuous** passed the frozen egg/required-word lexical screen on 4/4, versus 0/4 without an edit, and retained 7/7 baseline-correct protections (7/8 absolute in both arms). The output quality varied: one response combined a sunny-side egg with a telescope, while another compared a cafe to a “perfect egg.” Thus the 4/4 is lexical, not four demonstrations of natural or useful prose.

The published L23 N7748 at **+0.8 pulse12** passed the same screen on 3/4, retained 6/7 baseline-correct protections (6/8 absolute), and produced a capped repetitive fourth answer. Both recipes scored 1/2 in calibration. These tiny, breakfast-heavy sets suggest a promising dose-dependent egg bias and a cost to stronger steering; they do not establish a general-purpose control window or a fair superiority comparison between methods. Exact prompts, outputs, protection cases and write logs are in [the refinement JSON](window/results/confirmation.json), with the [calibration grid](window/results/calibration.json) and [patching data](followup/results/patches.json). Both follow-ups passed source-hash, completion-hash, exit-code and PID-correlated GPU checks.
