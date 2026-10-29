# PP rediscovery and dose calibration: a new apology-language control

**The strongest result is L22/N4945: PP found it, and a smaller, brief intervention passed the fresh-prompt control test after a strong continuous push failed.** It is absent from the checked same-model inventory and targeted source searches. This is a new candidate relative to those sources, not a proven first discovery worldwide. RMS also found it, so the result does not establish signed PP superiority.

The full experiment completed on lp: **14,499 scientific forward calls, zero backwards, 30.1 minutes of model-run time**, plus six infrastructure-smoke forwards. All three jobs exited zero. Source/ranking/selection audits passed; CUDA execution was confirmed in 180 PID-matched telemetry samples. The GPU was idle afterward. Raw outputs, prompts, failed conditions, schedules and sources are retained in this bundle.

## 1. Can PP rediscover the published neurons?

We used eight discovery pairs per behavior, one fixed token-set observable per behavior, and the corrected rule `mean(c * delta_activation)`, ranked by signed value. Supporting candidates must have positive scores. Each method nominated 50 coordinates, rather than the earlier two-candidate CAPS shortlist. Each had a separate layer-matched random shortlist with the same candidate-screening budget. Four additional pairs selected a singleton; eight fresh pairs tested restoration, reverse restoration and ablation.

| Published target | Signed PP global rank / 294,912 | RMS rank | Exact top-50 recovery |
| --- | ---: | ---: | --- |
| CAPS L1/N8396 | 185,784 | 234,160 | No |
| Alien L22/N2742 | **11** | 23 | **Yes, both methods** |
| Condolence component L10/N6295 | 171,773 | 268,742 | No |

The fixed lexical contrast classified both sides correctly on all eight fresh CAPS and alien pairs. Condolence classified all eight base cases and seven of eight target cases; all eight had a positive natural target-minus-base gap. These are constrained readouts, not full semantic-behavior qualifications. L10/N6295 is tested alone; this is not the two-neuron PEFT condolences preset, which also includes L15/N3920.

The [source explorer log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md) already describes the alien coordinate and some downstream neighbors. Its recovery is corroboration, not a new neuron claim. The experiment required 16 discovery forwards per behavior; the much larger validation and generation budget must not be hidden behind the two-pass discovery description.

## 2. Which additional neurons did it find?

| Candidate | Signed rank; RMS rank | Fresh natural restoration | What subsequent generation established |
| --- | --- | --- | --- |
| L30/N271, casing readout | 14; 20 | +0.584 logits, positive on 8/8 pairs | A local contributor; no qualifying CAPS controller in the tested additive, scale or clamp grids |
| L31/N1196, alien/AI readout | 3; 9 | +0.153 logits, positive on 8/8 | A local contributor; no qualifying free-form alien controller |
| **L22/N4945, sorry/glad readout** | **1; 2** | **+0.592 logits, positive on 8/8** | **A dose/timing recipe passed the fresh generation and protection checks** |

Reverse restoration moved all three readouts in the opposite direction on all eight pairs. Zeroing L22/N4945 reduced the target gap by 0.615 logits on average, with negative effects on 8/8. The matched random-screen winner for this task averaged −0.006 logits under restoration. These interventions support a local causal contribution to the measured contrast.

Signed PP and RMS selected the same strongest CAPS and condolence candidates. Their alien winners differed—L31/N1196 versus L29/N228—but fresh effects were +0.153 versus +0.148 logits; the paired bootstrap interval for their difference includes zero. Thus this batch supports useful PP candidate generation, with **no demonstrated advantage over RMS** on these outcomes.

![Restoration comparison](../restoration-comparison.png)

The checked index has 160 same-model coordinates, assembled from the saved Neuronary snapshot, PEFT presets and prior source coordinates. L22/N4945, L30/N271 and L31/N1196 are absent. Targeted repository and web searches found no matching published coordinate claims. These checks have bounded coverage: use “unlisted in the checked sources,” not “totally new to science.” See the [novelty audit](../novelty-audit.json).

## 3. The strongest dose-and-method result

The main experiment initially preferred a continuous g=+1 push on L22/N4945. It elicited apology language but changed a calibration answer from 7+4=11 to 10. We therefore froze a separate follow-up with new calibration and heldout cases, five positive doses and three timing schedules. It selected **g=+0.4 for the first four output predictions** before seeing the follow-up test set.

Here, `g` is a residual-relative write:

`h[L22] += g * R[L22] * W_down[:,4945] / norm(W_down[:,4945])`.

The generic reference passage gave `R[L22] = 26.813528`, so g=0.4 requested a write norm of about **10.725411**. The edit is active during prefill and the next three decode forwards, then removed. Prefill edits every prompt position; this is not only four isolated activation positions. The model/checkpoint, reference passage and exact implementation are in the frozen protocol.

| Recipe on the same fresh follow-up cases | Marker + required word / 8 | Protected answers / 4 | Interpretation |
| --- | ---: | ---: | --- |
| No edit | 0 | 4 | Normal positive-news responses |
| **L22/N4945, +0.4, first four predictions** | **8** | **4** | Calibration-qualified recipe; no target completion hit the 32-token cap |
| L22/N4945, +1, continuous | 0 | 2 | Repeated apologies, lost required words; all eight target outputs capped |
| Public L10/N6295, +1, first four | 8 by the lexical proxy | 0 | Mostly task refusals; seven target outputs capped; not a successful condolence controller |
| Matched random winner L31/N5385, chosen pulse | 0 | 4 | No target effect |

All four protected answers received the selected pulse on every prediction: the arithmetic outputs had three forwards each, and the identifiers had four and three. The successful protection result is not caused by withholding the intervention from these checks.

Calibration mapped a finite region: continuous and first-four interventions passed at +0.4 and +0.6, but not at +0.2 or the larger doses under the full gate. Delayed intervention needed larger doses to produce the marker. **Its short protection cases finished before steering began**, however, so their passes cannot establish preserved capability under exposure. The figure marks that distinction.

![Dose and timing calibration](../dose-timing-rescue.png)

The selected pulse uses fewer active forwards than continuous steering. But continuous +0.4 also passed calibration and was not separately confirmed on fresh prompts. We have established a successful combined recipe, **not that pulsing is necessary or superior at the same dose**. Persistence after release can also arise from conditioning on the generated opening; this experiment does not isolate latent-state memory from text continuation.

## 4. What the new control actually does

For a friend who recovered an orchid keepsake, the baseline said:

> That is wonderful news that your friend has finally recovered the orchid!

The L22/N4945 pulse said:

> I am so sorry to hear that your friend found the orchid!

For actual loss, it produced a coherent response such as:

> I am so deeply sorry for the loss of your irreplaceable family keepsake, coral.

The same pulse produced no apology markers in four neutral material-description prompts and retained their requested words. This suggests task-dependent expression in this small set, rather than an identical prefix on every input. It does not establish general selectivity. Baseline loss responses already expressed sympathy using “heartbroken”; the frozen sorry/sympathy-marker regex does not count that word. An increase in this marker is not an increase in empathy.

The public L10/N6295 pulse often began “I'm sorry, but I can't write a sentence…” and refused harmless arithmetic and identifiers. Its nominal 8/8 marker score is a clear proxy failure: refusal apologies and condolences share vocabulary. The protection gate and raw-text review prevent that result being mistaken for a useful controller.

L22/N4945 is therefore best described provisionally as an **apology/negative-valence language control**. It can impose the wrong emotional frame on good news. A plausible application is an explicitly requested fictional voice or an assay of inappropriate apologies; these experiments do not demonstrate a better empathetic assistant or a deployable production feature.

## 5. CAPS: a right sign helps, but no reliable window was established

The cheap fixed-readout dose filter chose public g=+0.2, opposite to the earlier useful sign. A separately frozen generation-first audit tested all 22 recipes for the public neuron, shared PP/RMS candidate and matched random winner, with new calibration and heldout prompts.

Public g=−0.2 produced the CAPS marker on 4/8 fresh copies, with **3/8 satisfying both casing and exact content**, while preserving 4/4 protections. It failed the strict calibration gate, so it remains a diagnostic challenger. The opposite +0.2 gave 0/8 joint successes; −1 destroyed all eight contents and all four protections. No tested recipe established a reliably transferable CAPS control.

The new L30/N271 candidate illustrates why casing alone is insufficient. At additive +1 it reached an uppercase fraction of one by outputting strings such as `CMCMCMCM...` and `RSVPRSVP...`; content retention was zero. Natural local importance and the ability to force uppercase characters do not establish useful text transformation.

![CAPS dose comparison](../generation-dose-window.png)

## 6. What this adds, and what remains unproved

The measured contribution is a complete small-scale chain: fixed-observable signed PP nominated an unlisted coordinate, fresh donor interventions supported its local contribution, and a separately frozen dose/timing policy transferred its language effect while preserving the tested short answers. The strong continuous push would have caused us to reject that candidate prematurely.

That result is consistent with [control-window research](https://arxiv.org/abs/2606.19831): target onset, collateral failure and timing must be measured separately. Here the useful region was found empirically; we did not test a quantitatively predicted ceiling or establish that this workflow beats other dose-search policies. It is not a validation of the entire theory.

The study covers one checkpoint/runtime, greedy decoding, three authored behaviors and small related prompt families. In particular, the eight positive-news confirmation cases share one sentence template and vary the keepsake name: they are not eight independent semantic contexts. The four protection cases are two arithmetic problems and two identifiers. Case-level bootstrap intervals describe the authored sample, not uncertainty across unseen template families. Two follow-ups were motivated by main-calibration outcomes and use fresh calibration/confirmation cases; they are transparently exploratory replications, not one untouched preregistration. Random controls received equal candidate-screening budgets in the main study, but only the selected positive-direction random candidate entered the targeted timing rescue. A broad method-superiority or global novelty claim would require a larger frozen inventory and independent prompt families.

The practical answer is **yes for one exact rediscovery and one new, narrowly validated intervention recipe; no for universal recovery or a general signed-PP advantage over RMS**. L30/N271 and L31/N1196 remain causal readout contributors without established general steering utility. No additional model experiments remain running.

## Evidence and reproduction

- [Main protocol](protocol.md), [generation-first audit](generation-audit-protocol.md), [dose/timing rescue protocol](condolence-rescue-protocol.md).
- [Main analysis](../analysis.json), [CAPS audit analysis](../generation-audit/analysis.json), [rescue analysis](../condolence-rescue/analysis.json).
- [Main verification](../verification.json), [CAPS verification](../generation-audit/verification.json), [rescue verification](../condolence-rescue/verification.json).
- [Exact rescue generations](../condolence-rescue/evidence/results/heldout.json), [all rescue calibration recipes](../condolence-rescue/evidence/results/trials.json).
- [Source and offline analysis](../source/), [main launch record](../launch.json), [complete file manifest](../manifest.json).

Stderr contains the audited runtime's reference-kernel notices, not execution failures. Local file-read and plotting-library stalls were resolved without rerunning model inference or changing scientific artifacts; figures were generated using a fresh temporary Matplotlib 3.10.7 environment. Source, selections and numerical results were checked independently offline.
