"""Render completed portable-style experiments, without changing any selection."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read(p): return json.loads(p.read_text())
def co(c): return f'L{c[0]} N{c[1]}'
def recipe(r): return f"g={r['dose']:+g}, {r['schedule']}"


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mirror', required=True); p.add_argument('--output', required=True); a = p.parse_args()
    m, d = Path(a.mirror), Path(a.output)
    protocol = read(m / 'calibration/protocol.json'); choices = read(m / 'selection.json')['choices']
    cal = read(m / 'review-calibration/decoded-calibration-analysis.json'); held = read(m / 'heldout-analysis.json')['summary']
    rows = read(m / 'heldout/rows.json'); figs = d / 'figures'; figs.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(13, 6.5), constrained_layout=True)
    for i, category in enumerate(protocol['categories']):
        for j, arm in enumerate(['candidate', 'random']):
            ax = axs[i, j]; trials = [x for x in cal['trials'] if x['category'] == category and x['arm'] == arm]
            grid = np.array([[next(x['metrics']['joint'] for x in trials if x['recipe']['dose'] == dose and x['recipe']['schedule'] == sched) for dose in protocol['doses']] for sched in protocol['schedules']])
            ax.imshow(grid, vmin=0, vmax=2, cmap='Blues', aspect='auto')
            for y, sched in enumerate(protocol['schedules']):
                for x, dose in enumerate(protocol['doses']):
                    t = next(t for t in trials if t['recipe']['dose'] == dose and t['recipe']['schedule'] == sched)
                    mm = t['metrics']; ax.text(x, y, f"{mm['joint']}/2 | {mm['protection_correct']}/4", ha='center', va='center', fontsize=8.5, color='white' if mm['joint'] == 2 else '#132337')
                    if t['qualified']: ax.add_patch(Rectangle((x-.47,y-.46),.94,.92,fill=False,edgecolor='#2b9d58',linewidth=2.4))
                    if t['recipe'] == choices[category][arm]['recipe']: ax.text(x, y-.28, '★', ha='center', color='#c45c0a', fontsize=14)
            ax.set_xticks(range(8), [f'{x:+g}' for x in protocol['doses']]); ax.set_yticks([0,1], ['Continuous','Pulse 4'])
            ax.set_title(f"{category.capitalize()} · {arm} · {co(protocol['categories'][category][arm])}")
            ax.set_xlabel('Signed residual-relative dose g')
    fig.suptitle('Calibration at each tested dose: joint style/content/quality | exact protection', fontsize=14)
    fig.supxlabel('Green outline: calibration qualified. Star: locked choice (possibly a nonqualified challenger). Each cell has only two creative cases.', fontsize=9)
    for ext in ['png','pdf']: fig.savefig(figs / ('calibration-grid.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, constrained_layout=True)
    conditions = ['baseline','prompt_only','candidate','random']
    for ax, category in zip(axs, protocol['categories']):
        values = [held[category][c]['metrics']['joint'] for c in conditions]
        semantic = [held[category][c]['metrics']['style2'] for c in conditions]
        xs=np.arange(4)
        ax.bar(xs-.18,semantic,width=.34,color='#75a8c5',label='Full semantic style')
        bars = ax.bar(xs+.18, values, width=.34,color='#285c87',label='All joint gates')
        for i, bar in enumerate(bars):
            pp = held[category][conditions[i]]['metrics']['protection_correct']
            ax.text(i,max(values[i],semantic[i])+.15,f'style {semantic[i]}/8; joint {values[i]}/8\nprotected {pp}/6',ha='center',fontsize=8)
        ax.set_ylim(0,9.5); ax.set_xticks(range(4), ['Baseline','Style prompt','PP neuron','Random neuron']); ax.set_title(category.capitalize())
        ax.axhline(6, color='#738194', linestyle='--', lw=.8)
    axs[0].set_ylabel('Held-out creative cases / 8');axs[0].legend(fontsize=8)
    fig.suptitle('Locked recipes on eight new creative prompts and six protection tasks', fontsize=13)
    fig.supxlabel('Overall qualification also requires ≥4 style improvements, 6/6 protections, and calibration qualification. Small authored cohorts.', fontsize=9)
    for ext in ['png','pdf']: fig.savefig(figs / ('heldout-outcomes.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)

    audits = {phase: read(m / f'{phase}-verification.json') for phase in ['calibration','heldout']}
    counts = {phase: audits[phase]['completion']['calls'] for phase in audits}
    passing = [cat for cat in held if held[cat]['candidate']['qualified_control']]
    lead = ('Both fixed PP candidates met the frozen calibration and held-out gates.' if len(passing)==2 else
            f"The {passing[0]} PP candidate met the frozen calibration and held-out gates." if passing else
            'Neither fixed PP candidate met both the frozen calibration and held-out gates in this dose/schedule grid.')
    lines = ['# Portable storytelling and nostalgia tests — results', '', '**'+lead+'**', '',
             'This experiment tests generation under fixed neuron edits. It follows the earlier donor-restoration evidence and uses a separate calibration/held-out design. Labels denote the discovery task, not an established exclusive neuron identity.', '',
             '| Category / arm | Locked recipe | Calibration qualified | Full semantic style | Joint held-out success | Style improvements | Exact protections | Capped creative outputs |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for cat in held:
        for cond in conditions:
            h = held[cat][cond]; mm=h['metrics']
            qual='—' if h['calibration_qualified'] is None else str(h['calibration_qualified'])
            lines.append(f"| {cat} / {cond} | {recipe(h['recipe']) if h['recipe'] else 'no neuron edit'} | {qual} | {mm['style2']}/8 | {mm['joint']}/8 | {mm['style_gain']}/8 | {mm['protection_correct']}/6 | {mm['capped']}/8 |")
    lines += ['', 'Joint success requires semantic style and quality scores of 2, all frozen lexical content terms, exactly two detected sentences, at most 40 words, and no token-limit truncation. Protection counts require exact answers within the generation budget; all protection termination flags are audited separately below. These are small authored cohorts scored by one assistant with condition labels withheld during text review, not an independent human replication.', '',
              'The explicit style prompts also fail the strict joint gate, despite receiving full semantic-style scores on 4/8 storytelling and 3/8 nostalgia cases. Missing exact content terms, content/format violations and some protected-answer failures limit this assay. The selected PP recipes score full semantic style on 0/8 cases each even before the lexical/format filters. This supports a failure of these tested recipes to transfer, not a universal impossibility result for the coordinates or methods.', '',
              'All held-out protection misses under the explicit style prompts were capitalization differences: “Oak” versus “oak” or “Blue” versus “blue.” Their answer content was correct. The frozen exact-string scores are retained, with this distinction made explicit.', '',
              '![Held-out outcomes](../figures/heldout-outcomes.png)', '', '## Calibration and the scope of a dose-window claim', '',
              'Both signs and every fixed dose received equal testing. A green cell satisfies the calibration gate; a star marks the locked recipe, including a diagnostic challenger when nothing qualified. The grid measures specific tested points on two creative prompts. It does not confirm every dose between points or establish a universal interval.', '',
              '![Calibration grid](../figures/calibration-grid.png)', '',
              'The random coordinates were fixed before this experiment and received the same grid and selection budget. At a given dose in the same layer, requested write norms match; independently selected held-out recipes can have different doses or schedules. A pulse stops direct writes after four predictions, but the generated opening can itself condition later text, so persistence is not proof of a latent-state memory mechanism.', '',
              'A specific adverse persistence effect occurred for the storytelling candidate at +0.4 and +0.8: both calibration prompts collapsed into repeated “Once”/“once-time” sequences under both schedules, producing 8/8 truncated outputs at the 72-token cap. The four pulse cases had only four audited write calls, so repetition continued after direct intervention ended. This is persistence of an undesirable output pattern; prefix conditioning and cached-state effects have not been separated.', '',
              'At +0.8, the storytelling candidate also corrupted the exact echo task under both schedules. At -0.8, the nostalgia candidate returned “416” for 6 × 7 under both schedules. These are genuine answer errors at stronger doses. Neither observation establishes a useful control window at a smaller dose.', '',
              '## Fixed-order examples', '', 'The first held-out creative prompt is shown for every condition, rather than choosing the most entertaining output. All outputs and scores are retained in the example appendix.', '']
    for cat in held:
        lines += [f'### {cat.capitalize()}', '']
        for cond in conditions:
            row = next(r for r in rows if r['category']==cat and r['condition']==cond); g=row['target'][0]
            lines += [f"**{cond}** ({'capped' if g['capped'] else 'ended naturally'}):", '', '```text', g['text'], '```', '']
    lines += ['## Protection and runtime checks', '']
    for cat in held:
        for cond in conditions:
            row=next(r for r in rows if r['category']==cat and r['condition']==cond)
            unfinished=[g['item'] for g in row['protection'] if g['capped'] or not g['terminated_eos']]
            wrong=[{'item':g['item'],'text':g['text']} for g in row['protection'] if not g['metrics']['exact_correct']]
            lines.append(f"- {cat}/{cond}: unfinished protections = `{unfinished}`; incorrect protections = `{wrong}`.")
    lines += ['', f"Forward calls: calibration {counts['calibration']:,}; held-out {counts['heldout']:,}; total {sum(counts.values()):,}. Zero backward passes. Both phase audits passed: {all(x['passed'] for x in audits.values())}. CUDA inference PIDs were "+', '.join(str(x['completion']['pid']) for x in audits.values())+'.', '',
              'The audits verify source/result seals, call accounting, each actual post-SwiGLU write, pulse exposure, final-prompt-only scope, frozen scale carryover and repeated neutral-baseline agreement across many intervened runs. The hybrid runtime retains the prior FP16 transformer on CUDA with the tied vocabulary boundary on CPU.', '',
              '## Evidence and limitations', '',
              'See the [frozen protocol](protocol.md), [all held-out examples](examples.md), [launch record](../launch.json), and preserved raw evidence and review seals. Exact lexical/format gates can reject semantically reasonable paraphrases; raw semantic ratings and texts remain available. A failed selected recipe or finite grid does not prove that no intervention can work. This experiment does not compare RMS or establish worldwide novelty.', '']
    notes=d/'notes';notes.mkdir(exist_ok=True);(notes/'results.md').write_text('\n'.join(lines)+'\n')
    grades={}
    mapping=read(m/'review-heldout/mapping.json')
    for g in read(m/'review-heldout/grades.json'): grades[mapping[g['id']]]=g
    examples=['# Every held-out output and sealed semantic score','']
    for row in rows:
        examples += [f"## {row['category']} / {row['condition']}",'']
        for g in row['target']:
            grade=grades[g['record_id']]
            examples += [f"### {g['item']}",'',g['prompt'],'',f"Style {grade['style']}/2; quality {grade['quality']}/2; capped={g['capped']}. {grade['note']}",'','```text',g['text'],'```','']
    (notes/'examples.md').write_text('\n'.join(examples)+'\n')
    print(lead)


if __name__ == '__main__': main()
