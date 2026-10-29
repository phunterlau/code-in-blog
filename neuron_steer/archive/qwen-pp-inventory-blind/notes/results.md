# Inventory-excluded signed-mean PP search — completed on lp

**PP recovered the original GitHub alien neuron L22 N2742 at rank 11 and the documented alien-pathway coordinate L27 N8347 at rank 1.** It also nominated alternative coordinates for all 16 requested categories without passing the inventory to the search process. The clearest validation result is for groups: the frozen top 16 shifted the lexical readout in the expected direction on all 128 fresh pairs, in both restoration directions, and exceeded the per-layer norm-matched random control on every pair. This is causal evidence about the measured readouts, not proof of portable persona control.

The limitation is substantial: the top-16 restoration matched the target's first token in **0/117 cases with distinct natural endpoints**. No steered generation, dose optimization, or competence-preservation test was run in this batch. Native target responses are prompt-elicited examples, not outputs of the neuron interventions.

## What was blind, and what was not

The chat had already seen published coordinates and previous results. This is therefore **inventory exclusion from the search inputs, not investigator blinding**. The fresh process received only the checkpoint, generic runtime, fixed behavior definitions, prompts and lexical readouts. It received no inventory, previous candidate list, activation archive, or published coordinate/sign/dose hints. A Python file-access guard and audit trace supported that separation; it was not an OS sandbox and cannot certify all native-library file accesses.

All 16 rankings and random lists were sealed at forward call 258, before any validation. All outputs were then sealed and audited before the precommitted reference snapshots were opened. The GitHub research log was fetched after that final seal; its coordinate cross-check is supplemental, distinct from the precommitted catalog endpoint. The [protocol](protocol.md), [selection seal](../evidence/results/selection-seal.json), [final seal](../evidence/results/final-seal.json), [verification](../verification.json), and [disclosure record](../reference-disclosure.json) preserve this order.

The eight CAPS, alien and sorry/glad discovery pairs are exact copies of earlier authored pairs, passed through a prompt-only extraction. Repeating them is a deterministic reproducibility check, not new independent discovery evidence. The other 13 category pairs were authored from the behavior descriptions. Each category used eight new validation pairs. Those 13 tasks share the same user requests with different system instructions, so the 128 pairs are not 128 independent behavior families.

## Method and measured effect

For coordinate j, `score_j = mean_i(c_j × (a_target,i,j − a_base,i,j))`, with signed scores ranked descending. Here `c_j` projects the down-projection column onto the fixed difference between mean target-token and control-token output embeddings. It is a direct-path approximation, not a downstream gradient. Actual intervention effects are measured separately.

The score uses activations at the final prompt token, immediately before `mlp.down_proj`. Top 1/4/16/50 supporting coordinates were frozen globally, along with layer-matched random prefixes. **There was no causal reranking, RMS ranking, inventory-informed filtering, or validation-based candidate replacement.** Top 50 is a retrieval list; only top 1/4/16 were intervened on.

Restoration sets the selected activations in a base-prompt pass to the corresponding natural target-prompt donor values. Reverse restoration uses the natural base donor in the target prompt. These donor values depend on the particular paired prompt; they are not a fixed portable dose. The table reports changes in the mean target-token minus mean control-token logit. Positive is the expected restoration direction; negative is expected for reverse restoration.

| Category | PP rank 1 | Singleton restoration; positive pairs | Top-16 restoration | Norm-matched random16 | Top-16 reverse |
|---|---|---:|---:|---:|---:|
| CAPS | L31 N5016 | +0.382; 8/8 | +3.005 | +0.057 | -2.895 |
| Oracle | L25 N3530 | +0.074; 8/8 | +1.062 | -0.172 | -1.567 |
| Loss of language | L26 N1 | -0.030; 3/8 | +1.527 | +0.230 | -1.254 |
| Nursery rhyme | L31 N2992 | +0.270; 8/8 | +1.599 | +0.409 | -1.484 |
| Salesperson | L27 N290 | +0.024; 5/8 | +0.532 | +0.056 | -0.692 |
| Building | L31 N1915 | +0.053; 7/8 | +0.792 | +0.259 | -0.740 |
| Corporation | L31 N1690 | +0.262; 8/8 | +0.698 | +0.059 | -0.829 |
| Victorian | L22 N4797 | +0.283; 8/8 | +2.017 | -0.165 | -1.052 |
| Academic | L25 N3530 | +0.015; 2/8 | +1.891 | -0.462 | -1.186 |
| Human role | L24 N2071 | +0.078; 8/8 | +0.694 | +0.047 | -0.750 |
| Worrier | L24 N2658 | +0.391; 8/8 | +2.663 | +0.150 | -2.196 |
| Safety focus | L21 N2227 | +0.413; 8/8 | +1.509 | -0.012 | -0.802 |
| Storyteller | L26 N2167 | +1.406; 8/8 | +3.077 | -0.049 | -3.212 |
| Nostalgia | L22 N8168 | +0.776; 8/8 | +2.423 | -0.103 | -2.812 |
| Alien / AI | L27 N8347 | +0.231; 8/8 | +0.873 | +0.006 | -0.724 |
| Sorry / glad | L22 N4945 | +0.715; 7/8 | +2.705 | +0.192 | -3.077 |

Each mean uses eight held-out pairs. Top-16 restoration was positive on 8/8 and reverse negative on 8/8 for every category. In both directions, top 16 exceeded the matched random control on 8/8 in every category. There is one frozen random coordinate set per category, reused across its prompts; this does not estimate uncertainty across many random masks. The raw natural-random controls and all top-4 results are retained in [analysis.json](../analysis.json).

![Held-out effects](../figures/heldout-effects.png)

## Cross-check, including misses and coverage failures

The precommitted Neuronary and PEFT snapshots contain 167 same-model recipe records and 153 distinct coordinates. The frozen title/alias rule produced **zero category-matched top-50 hits**, and the matched random lists also produced zero. This is not a 0/16 recovery rate: most requested categories lack a matching title in these snapshots. One substring match is plainly spurious—`verse` inside “I created the universe”—and some broad aliases are semantically ambiguous. The frozen result remains unchanged and the defect is disclosed. A simplistic whole-word repair would also remove the legitimate plural `condolences`, so that repair is not treated as ground truth.

Two distinct catalog coordinates appear in the PP50 lists under other categories: L21 N3435 (“Town Crier”) and L17 N8345 (a component of “Primal Storyteller” and “Caveman Qwen”). Neither is a category-matched rediscovery here. There are 627 distinct PP50 coordinates across 800 category slots; 625 are absent from this limited catalog. All 15 distinct rank-1 coordinates are absent from it, yet the GitHub check below already documents one. **Catalog absence does not establish novelty.**

The [pinned explorer research log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md) reports an original alien steering neuron and related pathway coordinates. These were compared only after sealing the search:

| GitHub coordinate | Global signed rank | Signed PP score | Source's role / evidence level |
|---|---:|---:|---|
| L22 N2742 | 11 | +0.045700 | Original alien steering positive control; the GitHub author reports coherent additive rollouts. |
| L27 N8347 | 1 | +0.100082 | Reported downstream write-path coordinate, labeled planet/star; no standalone generation reproduction asserted here. |
| L31 N7843 | 294,911 | -0.481808 | Reported downstream write-path coordinate, labeled Ali/alien. |
| L7 N8842 | 224,373 | -0.000051 | Reported upstream read-path coordinate, labeled scouts/science fiction; adjacent pathway evidence, not the same phenotype. |

Thus the original alien steering coordinate **is exactly recovered in top 16**, and a documented downstream coordinate is rank 1. L27 N8347 also has positive singleton donor effects on 8/8 fresh pairs and negative reverse effects on 8/8. L22 N2742 was tested as part of the top-16 group, not separately in this batch. We have not established causal edges between these neurons.

The 14 coordinates in the user-quoted social post are a separate appendix below; **none appears in its category's top 50**. That is a failure to recover those specific claims under these prompts and lexical readouts, not evidence that the original steering effects cannot occur. The social claims have not been independently reproduced here. We do not pool them with GitHub evidence or revise candidate selection to improve overlap.

## Findings beyond a coordinate inventory

**A low supporting rank need not mean low leverage.** L31 N7843 is second from the bottom of the signed ranking because its measured direct-path contribution is negative for this contrast. It is not a tiny-score coordinate. Calling it unimportant would discard sign information. This suggests a future test of an opposing contribution, but its causal effect was not measured individually here and it was not added to the selected set after disclosure.

![Signed alien pathway comparison](../figures/alien-signed-crosscheck.png)

**Positive contribution does not imply boosting the activation.** The rank-1 CAPS candidate L31 N5016 has mean discovery activation change −10.645 and negative readout coupling; their product is positive. The same sign pattern occurs for alien L27 N8347, worrier L24 N2658, and nostalgia L22 N8168. Using a positive activation push for every positive-ranked neuron would reverse the direction suggested by these contrasts. These activation differences are descriptive observations, not recommended fixed doses.

**Several alternatives merit control tests.** Storyteller L26 N2167 shifts its lexical readout by +1.406 logits as a singleton, positive on 8/8 prompts and reverse-negative on 8/8. Nostalgia L22 N8168 gives +0.776; safety focus L21 N2227 gives +0.413; worrier L24 N2658 gives +0.391. Each is absent from the 153-coordinate primary catalog. These are candidates for useful control, not established new persona neurons. Their labels name the search task, not an intrinsic, exclusive neuron identity.

**Groups are more reliable than the highest-ranked singleton in this assay.** The loss-of-language top candidate L26 N1 is PP-positive on all eight discovery pairs yet has a negative mean held-out donor effect (−0.030; positive on only 3/8). Its top-4 and top-16 groups are positive on 8/8. Academic and oracle share rank-1 coordinate L25 N3530, but the academic singleton is positive on only 2/8 fresh prompts, versus 8/8 for oracle. Shared rank does not establish a shared semantic feature, and improved group effects do not establish synergy; we did not measure every member separately.

**The current assay favors lexical output contributors.** Every rank-1 candidate is in layers 21–31. The explorer also reports depth-dependent differences and lexical effects in late layers. Our fixed token readouts and last-position capture can preferentially find late lexical directions while missing earlier, nonlinear, or sustained-generation controls. The 0/117 target-token conversion result is a useful boundary: partial readout movement can be real without reaching a decoding threshold or delivering a useful behavior.

## Implications for the dose / control-window question

This batch establishes candidate localization and sign-sensitive hypotheses. It does **not** map a control window, show that a published neuron failed only because of dose, or establish an advantage over RMS. It also cannot compare the full causal utility of published neurons against alternatives, because most published coordinates were not individually intervened on here.

The next experiment should keep these selections fixed and separate three choices: **coordinate, direction, and dose/timing**. A useful continuation would compare the fixed PP singleton and group against published controls and random controls on the same calibration prompts; tune a bounded signed dose grid and continuous versus short-pulse intervention; then lock the policy before new generation and competence tests. Target-style success and preservation of factual, numeric, and instruction-following behavior must both pass. Dose should be expressed using recorded activation writes and residual-relative norms, with the actual write verified at the hook. A separately held-out set must establish whether a nonempty window survives without a paired target donor. The low-ranked negative-score alien coordinate should be a separately labeled exploratory arm, not retroactively called a PP-positive discovery.

This would test the practical claim behind the control-window work: whether the same candidate is rejected at one dose but becomes useful within a measured interval, and whether that interval depends on task and timing. The explorer already discusses dose windows and random-direction effects, so novelty must come from a stronger prediction and held-out measurement, not from claiming that dose matters for the first time. No additional GPU batch was appended to the frozen search.

## Execution and reproducibility

- Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; hybrid FP16 transformer on CUDA, tied FP32 vocabulary boundary on CPU, no quantization, greedy native generation.
- Inference PID 1296466 on lp; 52 PID-correlated GPU samples; 3,994 forward calls (3,992 scientific plus two identity checks), zero backwards; about 8.7 minutes; exit 0. No experiment remains running.
- Source, protocol, selections and raw results are hash-sealed. Offline auditing recomputed signed rankings from all raw activations, checked exact random selection and masks, reconciled calls and checked donor/write ledgers. Reference inputs were disclosed only after this audit passed.
- The random norm control matches each PP16 layer's recorded `norm(W_down × actual_activation_delta)`. Across 1,952 layer comparisons, median relative discrepancy after FP16 storage is 0.0186%; the maximum is 9.38% on a tiny 0.00005775 reference norm (absolute error 0.00000542). Maximum absolute discrepancy across comparisons is 0.00748. Matching is finite-precision, not exact in every case.
- Two layer replacements differ from exact donor copies by at most 2.38×10⁻⁷ because the frozen program computes `source + (donor − source)` before the FP16 store. The audit reproduces that arithmetic exactly and reports the deviation. Scores and rankings are unaffected.
- Native baseline/target examples: 64 generations, 52 capped at 32 tokens. These are illustrative prompt checks; they are not a semantic-quality certification or intervention outputs.

The [candidate atlas](candidate-atlas.md) includes fixed readouts, prompts, top candidates, signs and native examples. The complete [raw evidence](../evidence/results/final-seal.json), [machine-readable descriptive analysis](../descriptive-analysis.json), [primary cross-check](../crosscheck.json), and [supplemental GitHub cross-check](../postseal-github-crosscheck.json) are saved locally. The package manifest covers source, data and reports.

## Appendix: user-quoted social coordinates

Ranks below use exactly the same frozen signed-mean scores. Source status: the [user-quoted LinkedIn post](https://www.linkedin.com/feed/update/urn:li:activity:7521233770402897920/), not independently reproduced here.

| Claimed category | Coordinate | Signed rank | Signed score |
|---|---|---:|---:|
| Oracle | L11 N1185 | 37,970 | +0.000239489 |
| Loss of language | L10 N5002 | 32,245 | +0.000360051 |
| Nursery rhyme | L6 N1525 | 288,196 | -0.000772976 |
| Salesperson | L4 N7242 | 160,302 | -1.36865e-06 |
| Building | L3 N3066 | 169,479 | -5.3061e-06 |
| Corporation | L4 N4783 | 268,302 | -0.000233916 |
| Victorian | L15 N5080 | 164,911 | -3.35189e-06 |
| Academic | L7 N950 | 70,860 | +7.31796e-05 |
| Human role | L7 N6297 | 261,477 | -0.000188208 |
| Worrier | L18 N6874 | 199,167 | -3.14267e-05 |
| Safety focus | L9 N1932 | 293,912 | -0.00133086 |
| Storyteller | L19 N264 | 132,331 | +2.33562e-06 |
| Storyteller | L21 N5992 | 273,367 | -0.000394561 |
| Nostalgia | L20 N7 | 31,683 | +0.000187958 |
