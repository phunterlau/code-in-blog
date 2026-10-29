# Control-window theory and the Qwen neuron inventories

The prior interpretation needed an explicit link to Hongliang Liu's [Leverage Is Not Reach: A Control-Window Law for Single-Neuron Steering in Language Models](https://arxiv.org/abs/2606.19831), submitted June 18, 2026. The October explorer's empirical warnings about overdosing belong in that framework. They should not be presented as the origin of the general control-window interpretation. The present experiments are applications and diagnostic extensions; they do not establish a priority claim about the law.

## The quantity to calibrate

For a post-SwiGLU neuron with write direction `v = Wdown[:, n]`, an additive activation offset `t` writes `t v` to the residual. The paper defines the reference budget `B = ||r_L|| / ||v||` and coupling `g = t / B`. At fixed unperturbed residual and baseline alignment `c0`, its geometric coordinate is

`c(g) = (c0 + g) / sqrt(1 + 2 c0 g + g^2)`.

The often-used `g / sqrt(1+g^2)` assumes negligible baseline alignment. For a negative controlling sign, resolve the branch consistently: reverse the direction or compare the signed branch coordinate. A negative-branch collapse is not an upper edge for a positive-branch trigger.

A coherent behavioral window requires the behavior's trigger to precede the coherence ceiling **on the same branch and at an adequate rollout horizon**. This has several consequences for the inventories:

- A common raw activation offset does not impose a common residual write, because column norms and residual scales differ.
- A common residual-relative dose removes that elementary scale confound, but it does not ensure every behavior is above trigger and below its ceiling. Context, direction, depth, and rollout length remain relevant.
- The ceiling model is approximate. The paper reports a Qwen3.5-4B mean collapse coefficient of 1.65, with within-model R-squared 0.71; this is not a guarantee that every coordinate remains coherent at a relative dose of 1. The fifteen-neuron prospective verdict was 11/15 against a 10/15 majority baseline. We retain these qualification limits.
- The trigger is measured, not derived from weights alone. Searching a dose after viewing behavioral outcomes is calibration, not an independent prospective prediction of that trigger.

The paper already demonstrates calibration corrections, including Llama-3.1-8B L11/F4258 and the prospective L31/F3309 case. It also distinguishes genuinely closed cases: collapse before trigger, insufficient downstream propagation, and normalization that caps a neuron's reach. Failure at one or two doses is not evidence of a closed window; neither does a wider sweep guarantee recovery.

## What the sources actually did

The [pinned explorer research log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md) and README already normalize additive writes by residual and column norm. Its `main_rel` design uses the three points -0.7, +0.5, and +0.8 on three short prompts, with a stronger free-continuation condition. It therefore does **not** use one fixed raw activation increment, and it does **not** test only one dose. The limitation is a shared, sparse dose grid rather than per-coordinate window estimation, with unequal positive and negative sampling.

The October 18 log initially describes a broadly shared residual-relative coherent window around 0.5–0.8. The October 19 depth survey explicitly qualifies that: at +0.8, 92% of layer-0 rollouts are empty, with failures among random directions too; layer 1 has 23% versus 18% empty, and layer 2 has 16% versus 1% for neurons versus controls. It states that layers 0–2 require calibration and lists a calibrated rerun as untested. This is source-reported evidence of an invalid measurement setting, not evidence that those layers lack useful controllers. Recovering coherent prose there would repair the setting; recovering a specific behavior would require a further target-sensitive test.

The primary [Neuron Persona Library](https://qwen-neurons-site.vercel.app/) is also adaptive. It starts at 1.25, tries 1, 0.75, and 0.5 only when a response is incoherent, and retains the strongest coherent response. It uses the identity prompt “Who are you?” and greedy generation up to 160 tokens. Two possible misses follow:

1. A coherent window below 0.5 is outside its grid.
2. A stronger coherent but off-target response can terminate retries even if a smaller dose would express a more interpretable persona or preserve more competence.

These are testable design limitations, not demonstrated false-negative inventory entries. This library uses Qwen3-8B and other models, and its coordinates cannot be transferred to our Qwen3.5-4B checkpoint. The source itself reports batch sensitivity and automated, unreviewed labels.

The [Mr-Neutr0n controlled reproduction](https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering) already includes multiple doses, norm-matched controls, capability checks, ablations, and timing interventions. It should not be characterized as a single-dose study either. A new correction would have to identify a particular failed behavior and show its recovery under a matched setting.

## Relation to our perturbation study

Our alien results already illustrate both edges: relative dose 0.25 leaves the default AI identity, 0.5 yields alien self-introductions with 12/12 elementary answers, and 1 yields alien wording with 5/12 answers and degraded continuations. This is an application of the control-window account. It is not a new discovery that a middle dose can work better than a large dose.

The direct contrastive score used in this pilot measures a local write-to-readout contribution. It does not exhaust the user's broader perturbation methodology. The control-window paper explains why first-order scores can miss finite-amplitude, off-axis controllers, and why a rollout-based causal filter can succeed. Consequently, the poor local ranks of the human, building, and corporation demo coordinates do not establish that they cannot control free generation at another dose or scope. A fuller discovery benchmark should compare budget-normalized finite interventions and behavior-specific rollout readouts under equal search budgets.

The existing near-equal fixed-answer scores but different greedy identities are compatible with this account: a scalar projection does not specify the reached continuation. The experiment remains useful as a concrete diagnostic of that readout. It does not by itself demonstrate a new basin mechanism or a contradiction of the paper.

## Combined edits and practical value

The paper already reports failed naive stacking and successful calibrated composition. Our natural-donor alien triplet adds a specific instance: the late L31 donor shift opposes the L22 and L27 local readout effects. Including it reduces the positive pair's transfer. Magnitude-ranked coordinates therefore need signed causal checks before combination.

For writes in the same residual space, the joint norm depends on all pairwise write-direction inner products, not just individual doses. For different layers, there is no single shared residual in which their reference budgets can simply be added; later budgets and triggers may change after the earlier intervention. Calibrate each singleton, then evaluate combinations and collateral effects directly. Single-neuron windows do not imply a rectangular safe region for a combination.

A practical controller would maximize held-out coherent target behavior subject to competence constraints, rather than maximize an identity-token margin or retain the strongest fluent dose. Report onset and persistence separately: an early lexical shift, an identity claim, a coherent persona, and useful task performance are distinct outcomes. Disabling the inference hook reverses the edit; no checkpoint training is involved here.

## Experiment contract and completed follow-ups

The [frozen protocol](../source/protocol.json) tests three public coordinates whose previous relative doses 0.25 and 1 were uninformative or degraded in our own runtime: human L7/N6297 positive, building L3/N3066 negative, and corporation L4/N4783 negative. It searches relative magnitudes 0.1, 0.25, 0.4, 0.55, 0.7, and 1, with two independently drawn same-layer random neurons per target given the identical search budget. The alien L22/N2742 coordinate is a dose anchor.

Dose selection uses two calibration introductions and three elementary answers. The selected dose is committed before four held-out introductions and twelve separate elementary questions. A fixed lexical/coherence screen guides selection; positive cases must also be inspected semantically. A null selection is explicitly marked and cannot be relabelled a controller. This is a small diagnostic calibration experiment, not a comprehensive semantic evaluation or a prospective validation of the ceiling law.

The new lexical coherence screen uses distinct-bigram >=0.6 plus a minimum length and repetition check, while the paper's collapse figures use a 0.30 distinct-bigram floor and behavior-specific judgments. Numerical ceilings from those two gates are not interchangeable. The geometry capture records reference c0 and residual norms; a reference B does not assert that every generated context has that exact norm.

All model runs retain the pinned checkpoint and our previously audited FP16 CUDA transformer with CPU FP32 vocabulary boundary. Differences from the public BF16 demonstrations limit exact-replication claims. We can diagnose our own earlier apparent failures; an original author's failed exact condition requires an exact matched reproduction before attributing it to the wrong dose.

The [completed findings](results.md) include the six-point identity search, bounded human refinement, fresh heraldry prompts, early-layer dose repair, and the EOS/geometry diagnostic. The primary and follow-up hypotheses, frozen lexical scores, and post-hoc semantic observations remain separately labelled.
