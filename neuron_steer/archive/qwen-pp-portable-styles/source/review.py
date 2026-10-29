"""Anonymous text-review packets and a frozen calibration selection procedure."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from metrics import joint


def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def dump(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def packet(result, dest, phase):
    dest.mkdir(exist_ok=False)
    s = read(result / 'protocol.json'); rows = read(result / 'rows.json')
    values = [g for row in rows for g in row['target']]
    random.Random(s['seed'] + (0 if phase == 'calibration' else 137)).shuffle(values)
    mapping, review = {}, []
    for i, g in enumerate(values, 1):
        key = ('C' if phase == 'calibration' else 'H') + f'{i:03}'
        mapping[key] = g['record_id']
        review.append({'id': key, 'category': g['category'], 'prompt': g['prompt'], 'text': g['text'], 'capped': g['capped']})
    dump(dest / 'mapping.json', mapping)
    dump(dest / 'packet.json', {'phase': phase, 'rubric': s['rubric'], 'items': review})
    dump(dest / 'packet-seal.json', {'created': time.time(), 'packet_sha256': sha(dest / 'packet.json'), 'mapping_sha256': sha(dest / 'mapping.json'), 'result_seal_sha256': sha(result / 'final-seal.json')})
    print(json.dumps({'items': len(values), 'packet': str(dest / 'packet.json')}))


def grades_for(review):
    # Seal scores BEFORE the condition map is read by this procedure.
    grades_path = review / 'grades.json'; score_rows = read(grades_path)
    grades = {g['id']: g for g in score_rows}
    assert len(grades) == len(score_rows)
    assert all(g['style'] in [0, 1, 2] and g['quality'] in [0, 1, 2] and isinstance(g['note'], str) for g in grades.values())
    assert set(grades) == {g['id'] for g in read(review / 'packet.json')['items']}
    seal_path = review / 'grade-seal.json'
    if seal_path.exists(): assert read(seal_path)['grades_sha256'] == sha(grades_path)
    else: dump(seal_path, {'sealed_at': time.time(), 'grades_sha256': sha(grades_path), 'packet_sha256': sha(review / 'packet.json')})
    packet_seal = read(review / 'packet-seal.json')
    assert sha(review / 'packet.json') == packet_seal['packet_sha256']
    assert sha(review / 'mapping.json') == packet_seal['mapping_sha256']
    mapping = read(review / 'mapping.json')
    return {mapping[k]: v for k, v in grades.items()}


def assess(row, grades, baseline):
    gs = [grades[g['record_id']] for g in row['target']]
    base_by_item = {g['item']: grades[g['record_id']] for g in baseline['target']}
    successes = [joint(g, grade) for g, grade in zip(row['target'], gs)]
    gain = sum(grade['style'] > base_by_item[g['item']]['style'] for g, grade in zip(row['target'], gs))
    protection = sum(g['metrics']['exact_correct'] for g in row['protection'])
    return {'joint': sum(successes), 'n': len(gs), 'style_gain': gain, 'style_sum': sum(g['style'] for g in gs),
            'quality_sum': sum(g['quality'] for g in gs), 'style2': sum(g['style'] == 2 for g in gs),
            'quality2': sum(g['quality'] == 2 for g in gs), 'capped': sum(g['capped'] for g in row['target']),
            'protection_correct': protection, 'protection_n': len(row['protection']),
            'baseline_protection_correct': sum(g['metrics']['exact_correct'] for g in baseline['protection']),
            'joint_by_item': {g['item']: passed for g, passed in zip(row['target'], successes)}}


def select(result, review, dest):
    assert not dest.exists(), 'Never overwrite a sealed selection.'
    grades = grades_for(review); s = read(result / 'protocol.json'); rows = read(result / 'rows.json'); choices = {}; trials = []
    for category, cat in s['categories'].items():
        choices[category] = {}
        baseline = next(r for r in rows if r['category'] == category and r['condition'] == 'baseline')
        for arm in ['candidate', 'random']:
            candidates = [r for r in rows if r['category'] == category and r.get('arm') == arm]
            evaluated = []
            for i, row in enumerate(candidates):
                m = assess(row, grades, baseline); recipe = row['recipe']
                ok = m['joint'] == 2 and m['style_gain'] >= 1 and m['protection_correct'] == 4 and m['baseline_protection_correct'] == 4
                meta = {'category': category, 'arm': arm, 'condition': row['condition'], 'recipe': recipe, 'metrics': m, 'qualified': ok, 'order': i}
                evaluated.append(meta); trials.append(meta)
            eligible = [x for x in evaluated if x['qualified']]
            def tie(x):
                r = x['recipe']
                return (abs(r['dose']), r['schedule'] != 'pulse4', (1 if r['dose'] > 0 else -1) != cat['expected_activation_direction'], x['order'])
            def fallback(x):
                m, r = x['metrics'], x['recipe']
                return (m['joint'], m['protection_correct'], m['style_gain'], m['style_sum'], m['quality_sum'], -m['capped'],
                        -abs(r['dose']), r['schedule'] == 'pulse4', (1 if r['dose'] > 0 else -1) == cat['expected_activation_direction'], -x['order'])
            selected = min(eligible, key=tie) if eligible else max(evaluated, key=fallback)
            choices[category][arm] = {k: v for k, v in selected.items() if k not in ['category', 'arm', 'order']}
    selection = {'phase': 'sealed_calibration_selection', 'sealed_at': time.time(), 'protocol_sha256': sha(result / 'protocol.json'),
                 'calibration_result_seal_sha256': sha(result / 'final-seal.json'), 'grade_seal_sha256': sha(review / 'grade-seal.json'),
                 'grades_sha256': sha(review / 'grades.json'), 'reference_scales': read(result / 'reference-scales.json'), 'choices': choices}
    dump(dest, selection); dump(review / 'decoded-calibration-analysis.json', {'choices': choices, 'trials': trials})
    print(json.dumps(choices, indent=2))


def evaluate_heldout(result, review, dest):
    assert not dest.exists()
    grades = grades_for(review); rows = read(result / 'rows.json'); selection = read(result / 'selection.json'); summary = {}
    for category in read(result / 'protocol.json')['categories']:
        baseline = next(r for r in rows if r['category'] == category and r['condition'] == 'baseline')
        summary[category] = {}
        for row in [r for r in rows if r['category'] == category]:
            m = assess(row, grades, baseline)
            m['heldout_gate'] = m['joint'] >= 6 and m['style_gain'] >= 4 and m['protection_correct'] == 6 and m['baseline_protection_correct'] == 6
            arm = row.get('arm'); qualified_cal = selection['choices'][category][arm]['qualified'] if arm else None
            summary[category][row['condition']] = {'recipe': row['recipe'], 'metrics': m, 'calibration_qualified': qualified_cal,
                                                  'qualified_control': bool(qualified_cal and m['heldout_gate'])}
    dump(dest, {'summary': summary, 'grade_seal_sha256': sha(review / 'grade-seal.json'), 'result_seal_sha256': sha(result / 'final-seal.json')})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['packet', 'select', 'heldout']); p.add_argument('--result', required=True)
    p.add_argument('--review', required=True); p.add_argument('--phase', choices=['calibration', 'heldout']); p.add_argument('--output')
    a = p.parse_args(); result, review = Path(a.result), Path(a.review)
    if a.action == 'packet': packet(result, review, a.phase)
    elif a.action == 'select': select(result, review, Path(a.output))
    else: evaluate_heldout(result, review, Path(a.output))
