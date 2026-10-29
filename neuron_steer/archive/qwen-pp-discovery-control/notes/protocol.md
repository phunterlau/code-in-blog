# PP rediscovery, causal testing and intervention calibration

Frozen before model outcomes on 2026-10-29 UTC (2026-10-28 local). The user authorized designing, running and analyzing experiments on lp.

Three targets: CAPS L1/N8396; extraterrestrial identity L22/N2742; condolence language L10/N6295. The last is an isolated component of the published two-neuron PEFT preset, so it is not an exact replication of that combined preset. All use the already audited Qwen3.5-4B checkpoint and hybrid loader: FP16 transformer CUDA, tied embedding/head FP32 CPU, no quantization, greedy decoding. This is a bounded assay, not a replication of every public neuron or the entire PP paper.

## Questions and frozen distinctions

1. **Exact recovery:** rank the public coordinate globally out of 294,912 with signed mean and signed ordering. Top 50 is a recovery; negative importance is not a supporting candidate. RMS is a separate comparator. Each behavior has one fixed token-set direction across prompts, unlike the prior example-dependent CAPS/policy direction.
2. **New causal candidates:** screen the leading 50 positive signed candidates and 50 RMS candidates with natural activation restoration on four separate pairs. Give each method its own 50 layer-matched random candidates and the identical screening budget. Select one winner per method, one public reference, and the strongest signed candidate absent from the frozen 160-coordinate public index. Deduplicate identical coordinates for execution while retaining method aliases.
3. **Fresh local causal evidence:** evaluate selected singletons on eight unused paired prompts, with target-to-base restoration, base-to-target reverse restoration, and zero ablation at the final prompt token. Retain failed lexical qualification and individual signed effects. These donor-assisted tests are not deployable controllers.
4. **Dose and method:** for each selected coordinate, compare ten residual-relative additive doses, six activation multipliers, and six clamps at base mean plus alpha times the discovery activation difference. All continuous generation edits apply to prompt and decode positions. Methods have equal search budgets across candidate-selection arms but do not imply equal physical intervention strength; actual writes are recorded.
5. **Fresh behavior:** commit recipes before eight unused generation prompts and four protection prompts. Run the calibration-qualified recipe, or one explicitly labeled failed-calibration challenger. A challenger is diagnostic and cannot be relabeled as a prequalified success. Also test the public coordinate at the predeclared signed normalized strength of one. This is a source-strength reference under our runtime, not a claim of exact source calibration equivalence.

## Selection and measurement

Discovery uses eight pairs. Four different pairs screen candidates; eight further pairs test restoration. The two generation-calibration prompts differ from the eight heldout generation prompts. First-token dose screening uses those two calibration prompts plus two protections: arithmetic and a case-sensitive identifier. Require positive fixed-observable movement, preservation of all baseline-correct protection argmaxes and average KL(base || edit) no greater than 0.5. For each method, generate at its best surviving dose. Final qualification requires both target examples to meet the trait AND content rule and both baseline-correct protections to remain correct. The heldout data never retune selection.

CAPS requires at least 85% uppercase letters and an otherwise exact copy modulo case. Alien identity uses an explicit lexical extraterrestrial marker plus retention of an unrelated required word. Condolence uses a sympathy/apology marker plus that content check. Its generation contexts are deliberately positive news: inducing condolences measures semantic overreach, not improved empathy. Semantic coherence beyond these deterministic tests will be reviewed descriptively from saved text; no posthoc semantic label replaces a failed frozen score.

The fixed readouts are explicit proxies. In particular, a single-token alien/AI distinction and sorry/glad distinction do not establish general persona or empathy mechanisms. The CAPS fixed token set also differs from individual heldout word logits. Report natural contrast qualification, restoration and generated behavior separately.

## Budgets and evidence

Hard limits: 22,000 forward calls and 90 minutes for the main process; zero backward passes. A small separate infrastructure smoke check tests identity residual/scale edits, clamp execution and generation before launch. No outcome-driven retries or prompt/grid changes. The remote job runs detached with PID-correlated GPU telemetry. Preserve source/protocol hashes, model/runtime metadata, exact generation tokens, all candidate screen results, all tested doses, selections, failed gates and completion status locally.

No superiority claim follows merely from one successful candidate. Search-matched controls and fresh data support a scoped comparison; this sample does not establish population-wide superiority or global coordinate novelty. Inventory absence means absence from the saved same-model Neuronary/PEFT/prior-source index only.

[Machine-readable protocol](../source/protocol.json) · [Source hashes](../source/source-hashes.json).
