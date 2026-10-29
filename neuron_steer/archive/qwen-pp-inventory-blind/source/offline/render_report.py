"""Render the completed sealed search as Markdown and static scientific figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {'caps': 'CAPS', 'oracle': 'Oracle', 'losing_language': 'Loss of language', 'nursery_rhyme': 'Nursery rhyme',
          'salesperson': 'Salesperson', 'building': 'Building', 'corporation': 'Corporation', 'victorian': 'Victorian',
          'academic': 'Academic', 'human': 'Human role', 'worrier': 'Worrier', 'safety': 'Safety focus',
          'storyteller': 'Storyteller', 'nostalgic': 'Nostalgia', 'alien': 'Alien / AI', 'condolence': 'Sorry / glad'}
def read(p): return json.loads(p.read_text())
def coordinate(c): return f'L{c[0]} N{c[1]}'
def fmt(x): return f'{x:+.3f}'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--deliverable', required=True); args = p.parse_args()
    root, d = Path(args.root), Path(args.deliverable)
    a = read(root / 'analysis.json'); s = read(d / 'descriptive-analysis.json'); cross = read(d / 'crosscheck.json')
    protocol = read(root / 'evidence/results/protocol.json')
    categories = list(a['categories']); yy = np.arange(len(categories))
    figs = d / 'figures'; figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 3, figsize=(14, 9), gridspec_kw={'width_ratios': [0.85, 1.4, 1.4]}, sharey=True)
    fig.suptitle('PP-only selection: held-out causal readout shifts', fontsize=18, x=.52, y=.98)
    fig.text(.52, .943, 'Qwen3.5-4B · 16 categories · eight fresh pairs per category · selection sealed before validation', ha='center', color='#46566a')
    gaps = [a['categories'][k]['validation_gap']['mean'] for k in categories]
    axs[0].barh(yy, gaps, height=.55, color='#c2cddd')
    axs[0].set_title('Natural prompt contrast', fontsize=12, pad=14)
    axs[0].set_xlabel('Mean target − base readout')
    axs[0].set_yticks(yy, [LABELS[k] for k in categories]); axs[0].invert_yaxis()
    colors = {'pp1': '#8963b6', 'pp16': '#1367ac', 'norm_random16': '#d8792f', 'random16': '#89939d'}
    legend_names = {'pp1': 'PP top 1', 'pp16': 'PP top 16', 'norm_random16': 'Random 16, norm matched', 'random16': 'Random 16, natural donor'}
    for ax, mode, title in zip(axs[1:], ['restore', 'reverse'], ['Target donor → base prompt', 'Base donor → target prompt']):
        sign = 1 if mode == 'restore' else -1
        for arm, offset in [('pp1', -.23), ('pp16', -.075), ('norm_random16', .075), ('random16', .23)]:
            vals = np.array([a['categories'][k]['arms'][arm][mode]['mean'] * sign for k in categories])
            if arm == 'pp16':
                lo = np.array([min(np.array(a['categories'][k]['arms'][arm][mode]['values']) * sign) for k in categories])
                hi = np.array([max(np.array(a['categories'][k]['arms'][arm][mode]['values']) * sign) for k in categories])
                ax.hlines(yy + offset, lo, hi, color=colors[arm], alpha=.25, linewidth=2)
            ax.scatter(vals, yy + offset, s=24 if arm != 'pp16' else 36, color=colors[arm], label=legend_names[arm], zorder=3)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_xlabel('Mean Δ readout' if mode == 'restore' else 'Mean −Δ readout (expected direction)')
        ax.axvline(0, color='#526172', lw=.8)
        ax.grid(axis='x', alpha=.16)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.54, .049), ncol=2, frameon=False, fontsize=10)
    fig.text(.52, .025, 'Blue spans: range over eight prompts, not confidence intervals. Readouts are lexical proxies; scales differ across categories.', ha='center', fontsize=9, color='#46566a')
    fig.subplots_adjust(left=.14, right=.985, top=.885, bottom=.16, wspace=.22)
    for ext in ['png', 'pdf']: fig.savefig(figs / ('heldout-effects.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    # Every coordinate on this chart was named in the same GitHub pathway passage.
    refs = s['supplemental_github']['alien_references']
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    vals = [r['signed_mean'] for r in refs]
    ax.barh(range(4), vals, color=['#2271a5' if v > 0 else '#bd563d' for v in vals], height=.55)
    ax.set_yticks(range(4), [coordinate(r['coordinate']) for r in refs]); ax.invert_yaxis()
    ax.axvline(0, color='#46566a', lw=.8)
    for i, row in enumerate(refs):
        val = row['signed_mean']
        ax.text(val + (.012 if val >= 0 else -.012), i, f"rank {row['signed_rank']:,}", va='center', ha='left' if val >= 0 else 'right', fontsize=10)
    ax.set_xlim(-.66, .23); ax.set_xlabel('Signed-mean PP score for the alien / AI contrast')
    ax.set_title('Published pathway coordinates need not share a sign', fontsize=14, pad=14)
    fig.text(.51, .02, 'Post-seal GitHub cross-check. Rank is among 294,912 coordinates; no negative-score candidate was added to the selected set.', ha='center', fontsize=8.5, color='#46566a')
    fig.subplots_adjust(left=.16, right=.96, bottom=.22, top=.83)
    for ext in ['png', 'pdf']: fig.savefig(figs / ('alien-signed-crosscheck.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)

    rows = []
    for name in categories:
        cat, desc = a['categories'][name], s['categories'][name]
        one, group, rand = [cat['arms'][x] for x in ['pp1', 'pp16', 'norm_random16']]
        rows.append(f"| {LABELS[name]} | {coordinate(cat['top1'])} | {fmt(one['restore']['mean'])}; {one['restore']['positive']}/8 | {fmt(group['restore']['mean'])} | {fmt(rand['restore']['mean'])} | {fmt(group['reverse']['mean'])} |")
    github_table = '\n'.join(f"| {coordinate(r['coordinate'])} | {r['signed_rank']:,} | {r['signed_mean']:+.6f} | {r['role']} |" for r in refs)
    social_rows = []
    for name, row in cross['categories'].items():
        for c in row['secondary_user_quoted_social']:
            social_rows.append(f"| {LABELS[name]} | {coordinate(c['coordinate'])} | {c['rank']:,} | {c['signed_score']:+.6g} |")
    report = f'''# Inventory-excluded signed-mean PP search — completed on lp

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
{chr(10).join(rows)}

Each mean uses eight held-out pairs. Top-16 restoration was positive on 8/8 and reverse negative on 8/8 for every category. In both directions, top 16 exceeded the matched random control on 8/8 in every category. There is one frozen random coordinate set per category, reused across its prompts; this does not estimate uncertainty across many random masks. The raw natural-random controls and all top-4 results are retained in [analysis.json](../analysis.json).

![Held-out effects](../figures/heldout-effects.png)

## Cross-check, including misses and coverage failures

The precommitted Neuronary and PEFT snapshots contain 167 same-model recipe records and 153 distinct coordinates. The frozen title/alias rule produced **zero category-matched top-50 hits**, and the matched random lists also produced zero. This is not a 0/16 recovery rate: most requested categories lack a matching title in these snapshots. One substring match is plainly spurious—`verse` inside “I created the universe”—and some broad aliases are semantically ambiguous. The frozen result remains unchanged and the defect is disclosed. A simplistic whole-word repair would also remove the legitimate plural `condolences`, so that repair is not treated as ground truth.

Two distinct catalog coordinates appear in the PP50 lists under other categories: L21 N3435 (“Town Crier”) and L17 N8345 (a component of “Primal Storyteller” and “Caveman Qwen”). Neither is a category-matched rediscovery here. There are 627 distinct PP50 coordinates across 800 category slots; 625 are absent from this limited catalog. All 15 distinct rank-1 coordinates are absent from it, yet the GitHub check below already documents one. **Catalog absence does not establish novelty.**

The [pinned explorer research log](https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md) reports an original alien steering neuron and related pathway coordinates. These were compared only after sealing the search:

| GitHub coordinate | Global signed rank | Signed PP score | Source's role / evidence level |
|---|---:|---:|---|
{github_table}

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
{chr(10).join(social_rows)}
'''
    (d / 'notes/results.md').write_text(report)

    atlas = ['# Fixed candidates and native prompt examples', '',
             'All coordinates were selected before validation and reference disclosure. Names below denote the task used to find a candidate, not a verified exclusive neuron identity. Native examples are unsteered outputs under the base or target prompt; they must not be presented as intervention successes.', '',
             'Activation changes are averages over discovery pairs, in raw activation units. They are not portable doses. Supporting candidates were ranked by the signed PP contribution, not by the activation change alone. Full top50 lists and per-pair scores are in [descriptive-analysis.json](../descriptive-analysis.json).', '']
    for name in categories:
        cat, desc, task = a['categories'][name], s['categories'][name], protocol['tasks'][name]
        atlas += [f'## {LABELS[name]}', '',
                  f"Target vocabulary: `{', '.join(task['target_tokens'])}`. Control vocabulary: `{', '.join(task['control_tokens'])}`. Only single-token bare/space-prefixed variants enter the observable; exact IDs are in [{name}/observable.json](../evidence/results/{name}/observable.json).", '',
                  '| Rank | Coordinate | Signed score | Mean activation change | Coupling | Positive discovery pairs |',
                  '|---:|---|---:|---:|---:|---:|']
        for row in desc['top50'][:4]:
            atlas.append(f"| {row['signed_rank']} | {coordinate(row['coordinate'])} | {row['signed_mean']:+.6f} | {row['mean_activation_delta']:+.4f} | {row['coupling']:+.6f} | {row['positive_discovery_pairs']}/8 |")
        atlas += ['', f"Singleton held-out effect: {fmt(cat['arms']['pp1']['restore']['mean'])} logits, positive on {cat['arms']['pp1']['restore']['positive']}/8; reverse {fmt(cat['arms']['pp1']['reverse']['mean'])}, negative on {cat['arms']['pp1']['reverse']['negative']}/8.", '',
                  'First held-out pair, shown according to the frozen order:', '']
        pair = task['validation'][0]
        for side in ['base', 'target']:
            atlas += [f'**{side.capitalize()} prompt**', '']
            for message in pair[side]: atlas.append(f"- {message['role']}: {message['content']}")
            if pair.get('suffix'): atlas.append(f"- Assistant suffix: `{pair['suffix']}`")
            gen = next(g for g in cat['native_generations'] if g['side'] == side and g['pair'] == pair['id'])
            atlas += ['', f"**Native {side} output** ({'capped at 32 tokens' if gen['capped'] else 'terminated naturally'}; no intervention):", '',
                      '```text', gen['text'], '```', '']
    (d / 'notes/candidate-atlas.md').write_text('\n'.join(atlas) + '\n')
    print('Rendered results.md, candidate-atlas.md and two PNG/PDF figures.')


if __name__ == '__main__': main()
