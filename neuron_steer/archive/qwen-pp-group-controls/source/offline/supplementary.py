"""Post-run descriptive summaries only; consumes audited records, no model calls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stats(values):
    v = np.asarray(values, dtype=float)
    return {'mean': float(v.mean()), 'minimum': float(v.min()),
            'median': float(np.median(v)), 'maximum': float(v.max()),
            'positive': int((v > 0).sum()), 'over_005': int((v > .05).sum()),
            'n': len(v), 'values': v.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    out = Path(ap.parse_args().output)
    audit = json.loads((out / 'verification.json').read_text())
    assert audit['passed'] and not audit['synthetic_fixture']
    analysis = json.loads((out / 'analysis.json').read_text())['categories']
    protocol = json.loads((out / 'source/protocol.json').read_text())
    norms = json.loads((out / 'evidence/results/vector-norms.json').read_text())
    rows = [json.loads(line) for line in (out / 'evidence/results/records.jsonl').read_text().splitlines()]
    result = {'scope': 'Post-run descriptive analysis of all frozen arms; zero additional inference or selection for retesting.',
              'categories': {}}
    lines = ['# Additional diagnostics from the frozen group run', '',
             'These are descriptive summaries after the run. They do not select a new neuron, group, or recipe for a confirmatory test. All comparisons use the already audited outputs.', '',
             '## Effect magnitude under two budget definitions', '',
             'Positive means the intended direction. Both budgets use the numerical dose 0.4, but measure different physical costs; they are not interchangeable doses.', '',
             '| Category | Direction | Budget | PP1 mean | PP4 mean | PP16 mean | Random16 mean |',
             '|---|---|---|---:|---:|---:|---:|']
    for cat, c in analysis.items():
        d = {'budget_comparisons': {}, 'components': {}, 'deletions': {}, 'rank1_deletion': {}}
        for side in ['base', 'target']:
            for budget, label, prefix in [('B2', f'{side}:0.4', ''), ('B1', f'{side}:l1:0.4', 'l1_')]:
                arm = c['arms'][label]
                names = [prefix + n for n in ['pp1', 'pp4', 'pp16', 'random16_0']]
                d['budget_comparisons'][label] = {n: arm[n] for n in names}
                lines.append(f'| {cat} | {side} | {budget} | ' + ' | '.join(f"{arm[n]['mean']:+.3f}" for n in names) + ' |')
            deletion = []
            for rank in range(1, 17):
                name = f'drop{rank:02}_matched'
                item = c['deletions'][side][name]
                assert all(x is not None for x in item['retention_where_full_over_005'])
                deletion.append({'rank': rank, 'coordinate': protocol['categories'][cat]['pp16'][rank-1],
                                 'retention': stats(item['retention_where_full_over_005']), 'effect': item['effect']})
            d['deletions'][side] = deletion
            for dose in [.1, .4]:
                label = f'{side}:{dose}'
                arms = c['arms'][label]
                component = np.asarray(c['interactions'][label]['component_effects'])
                means = component.mean(axis=1)
                best = int(np.argmax(means))
                d['components'][label] = {'all_means_by_rank': means.tolist(), 'largest_mean_rank': best+1,
                                        'largest_mean_coordinate': protocol['categories'][cat]['pp16'][best],
                                        'component_sum_mean': float(means.sum()),
                                        'nonadditivity': c['interactions'][label]['group_minus_sum_components']}
                d['rank1_deletion'][label] = {
                    'unscaled_minus_full': stats(np.array(arms['drop01_unscaled']['values']) - np.array(arms['pp16']['values'])),
                    'unscaled_mean': arms['drop01_unscaled']['mean'],
                    'unscaled_over_005': arms['drop01_unscaled']['over_005'],
                    'requested_B2': dose * norms[cat]['drop01_unscaled']['block_l2'],
                    'matched_mean': arms['drop01_matched']['mean'],
                    'full_mean': arms['pp16']['mean']}
        result['categories'][cat] = d
    lines += ['', '## Removing the top-ranked member without rescaling', '',
              '| Category | Direction | Dose | Full PP16 mean | Without rank 1 mean | Change | Remaining B2 | Effects > .05 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for cat, c in result['categories'].items():
        for label, d in c['rank1_deletion'].items():
            side, dose = label.split(':')
            lines.append(f"| {cat} | {side} | {dose} | {d['full_mean']:+.4f} | {d['unscaled_mean']:+.4f} | {d['unscaled_minus_full']['mean']:+.4f} | {d['requested_B2']:.4f} | {d['unscaled_over_005']}/8 |")
    lines += ['', '## Individual components at group-assigned amplitudes', '',
              'The largest member below is identified descriptively after seeing these outcomes; it is not independently validated as an optimal singleton. Members do not receive a full normalized group dose.', '',
              '| Category | Direction / dose | Largest mean component | Mean component sum | Mean group effect | Nonadditivity |',
              '|---|---|---|---:|---:|---:|']
    for cat, c in result['categories'].items():
        for label, d in c['components'].items():
            l, n = d['largest_mean_coordinate']
            gm = analysis[cat]['arms'][label]['pp16']['mean']
            lines.append(f"| {cat} | {label} | rank {d['largest_mean_rank']}: L{l} N{n} | {d['component_sum_mean']:+.4f} | {gm:+.4f} | {d['nonadditivity']['mean']:+.4f} |")
    base = {r['item']: r for r in rows if r['kind']=='protection' and r['arm']=='baseline' and r['category']=='losing_language'}
    tokens = json.loads((out / 'evidence/results/protection-tokenization.json').read_text())
    decode = {v[0]: k for k, v in tokens.items()}
    changes = [r for r in rows if r['kind']=='protection' and r['argmax']!=base[r['item']]['argmax']]
    result['protection'] = {
        'baseline': [{'item': i, 'prompt': protocol['protections'][i]['prompt'],
                      'predicted': decode[r['argmax']], 'expected': decode[r['expected_token']],
                      'correct': r['correct_first_token']} for i, r in base.items()],
        'changes': [{'category': r['category'], 'item': r['item'], 'arm': r['arm'], 'dose': r['sign']*r['dose'],
                     'before': decode[base[r['item']]['argmax']], 'after': decode[r['argmax']],
                     'correct': r['correct_first_token']} for r in changes]}
    lines += ['', '## Exact first-token protection changes', '',
              'Every retained protection argmax is A or B, verified against the saved tokenizer IDs. There are no hidden format-token substitutions in these diagnostics. Baseline is 7/8: it incorrectly chooses A (zebra) instead of B (apple) on the alphabetical-order question.', '',
              '| Category | Arm / signed dose | Question | Before → after | Result |',
              '|---|---|---|---|---|']
    for r in result['protection']['changes']:
        prompt = protocol['protections'][r['item']]['prompt']
        lines.append(f"| {r['category']} | {r['arm']} / {r['dose']:+g} | {prompt} | {r['before']} → {r['after']} | {'corrected' if r['correct'] else 'new error'} |")
    lines += ['', 'PP16 leaves every individual A/B prediction unchanged in all four categories at both signs and doses. Academic PP4 at −0.4 adds one error. Random groups are not inert: storytelling random16 at +0.4 corrects the baseline alphabetical error, while nostalgia random16 at −0.4 adds one error. These are eight-question next-token diagnostics only.', '',
              '![Budget sensitivity](../figures/budget-sensitivity.png)', '',
              '![Deletion retention](../figures/deletion-retention.png)', '',
              'See [all frozen-arm results](results.md), [machine-readable supplementary summaries](../secondary-analysis.json), and [the protocol](protocol.md).', '']
    (out / 'notes/diagnostics.md').write_text('\n'.join(lines))
    dump(out / 'secondary-analysis.json', result)
    colors = ['#8795a5', '#4a8f79', '#256ba1']
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (cat, c) in zip(axs.flat, analysis.items()):
        for j, side in enumerate(['base', 'target']):
            for k, (name, color) in enumerate(zip(['l1_pp1', 'l1_pp4', 'l1_pp16'], colors)):
                y = np.asarray(c['arms'][f'{side}:l1:0.4'][name]['values']); x = j + (k-1)*.23
                ax.scatter(x + np.linspace(-.035, .035, 8), y, s=15, color=color, alpha=.65)
                ax.plot(x, y.mean(), '_', color=color, markersize=15, markeredgewidth=2, label=name.replace('l1_', '') if j==0 else None)
        ax.axhline(0, color='#999999', lw=.8); ax.set_title(cat.replace('_', ' ').capitalize())
        ax.set_xticks([0, 1], ['Base +0.4', 'Target −0.4']); ax.set_ylabel('Oriented lexical effect')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Equal coordinate-L1 cost: larger groups do not always have larger effects')
    fig.supxlabel('Eight prompts per category. Dose 0.4 under B1 is not the same physical budget as dose 0.4 under B2.', fontsize=9)
    for ext in ['png', 'pdf']: fig.savefig(out / f'figures/budget-sensitivity.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (cat, c) in zip(axs.flat, result['categories'].items()):
        for side, color in [('base', '#256ba1'), ('target', '#b16836')]:
            vals = c['deletions'][side]
            ax.plot(range(1,17), [d['retention']['median'] for d in vals], 'o-', ms=3, color=color, label=side)
            ax.fill_between(range(1,17), [d['retention']['minimum'] for d in vals], [d['retention']['maximum'] for d in vals], alpha=.14, color=color)
        ax.axhline(1, color='#999999', lw=.8); ax.set_title(cat.replace('_', ' ').capitalize())
        ax.set_ylim(.5, 1.5); ax.set_xticks([1,4,8,12,16]); ax.set_xlabel('Removed PP rank')
        ax.set_ylabel('Effect / full-group effect')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Leave-one-member-out retention at B2 dose 0.4')
    fig.supxlabel('Lines: median across eight prompts; bands: minimum–maximum. Remaining coordinates are rescaled to the original B2 budget.', fontsize=9)
    for ext in ['png', 'pdf']: fig.savefig(out / f'figures/deletion-retention.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'additional_model_calls': 0, 'categories': list(result['categories']), 'protection_changes': len(changes)}))


if __name__ == '__main__':
    main()
