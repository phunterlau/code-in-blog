"""Offline integrity and activation-write audit; never prints generated text."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np


def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--workspace', required=True); p.add_argument('--phase', required=True, choices=['calibration', 'heldout']); p.add_argument('--source', required=True)
    a = p.parse_args(); workspace, source = Path(a.workspace), Path(a.source); result = workspace / a.phase
    spec = importlib.util.spec_from_file_location('frozen_metrics', source / 'metrics.py'); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    s = read(result / 'protocol.json'); rows = read(result / 'rows.json'); checks = {}
    final = read(result / 'final-seal.json')
    for name, digest in final['files'].items(): checks['seal:' + name] = sha(result / name) == digest
    checks['protocol'] = sha(result / 'protocol.json') == sha(source / 'protocol.json')
    for name, digest in read(result / 'source-hashes.json').items():
        checks['source:' + name] = sha(source / name) == digest == sha(workspace / name)
    items = {x['id']: x for x in s[a.phase] + s['protection_' + a.phase]}
    scales = {int(k): v for k, v in read(result / 'reference-scales.json').items()}
    records = [json.loads(line) for line in (result / 'records.jsonl').read_text().splitlines()]
    from_rows = [g for row in rows for k in ['target', 'protection'] for g in row[k]]
    checks['journal_equals_rows'] = records == from_rows
    checks['unique_ids'] = len({g['record_id'] for g in records}) == len(records)
    # Same neutral baseline prompts are repeated after many intervened runs. Exact
    # agreement checks for persistent hook/cache contamination between generations.
    baselines = [r for r in rows if r['condition'] == 'baseline']
    base_tokens = [{g['item']: g['tokens'] for kind in ['target', 'protection'] for g in r[kind]} for r in baselines]
    checks['neutral_baseline_repeatability'] = all(x == base_tokens[0] for x in base_tokens[1:])
    total_calls = 0; deviations = []; zero_writes = 0; active_counts = []
    for row in rows:
        recipe = row['recipe']; cat = s['categories'][row['category']]
        if recipe:
            assert recipe['coordinate'] == cat[row['arm']]
            assert recipe['dose'] in s['doses'] and recipe['schedule'] in s['schedules']
        for kind in ['target', 'protection']:
            for g in row[kind]:
                assert g['recipe'] == recipe and g['kind'] == kind and g['condition'] == row['condition']
                item = items[g['item']]; assert g['prompt'] == item['prompt']
                expected_system = cat['prompt_only_system'] if row['condition'] == 'prompt_only' else s['neutral_system']
                assert g['system'] == expected_system
                assert g['metrics'] == module.measure(g['text'], item, g['capped'])
                assert g['calls'] == len(g['tokens']) <= g['cap']
                assert g['cap'] == s['target_cap' if kind == 'target' else 'protection_cap']
                assert g['capped'] == (len(g['tokens']) == g['cap'] and not g['terminated_eos'])
                total_calls += g['calls']
                steps = [i for i in range(g['calls']) if recipe is not None and module.active(recipe['schedule'], i)]
                assert [w['step'] for w in g['writes']] == steps
                active_counts.append({'record_id': g['record_id'], 'calls': g['calls'], 'active_calls': len(steps)})
                for w in g['writes']:
                    assert [w['layer'], w['neuron']] == recipe['coordinate']
                    assert w['reference_residual_norm'] == scales[w['layer']]
                    assert w['sequence_positions'] == (len(g['input_ids']) if w['step'] == 0 else 1)
                    assert w['edited_positions'] == 1
                    delta = recipe['dose'] * scales[w['layer']] / w['column_norm']
                    assert delta == w['requested_delta']
                    expected_after = float(np.float16(np.float32(w['before']) + np.float32(np.float16(delta))))
                    assert expected_after == w['after'], (g['record_id'], w, expected_after)
                    actual = w['after'] - w['before']
                    assert actual == w['actual_delta']
                    assert abs(actual) * w['column_norm'] == w['actual_write_norm']
                    expected_norm = abs(recipe['dose']) * scales[w['layer']]
                    assert expected_norm == w['requested_write_norm']
                    deviations.append(abs(w['actual_write_norm'] - expected_norm) / expected_norm)
                    zero_writes += actual == 0
    for category in s['categories']:
        catrows = [r for r in rows if r['category'] == category]
        assert len(catrows) == (2 + 2 * len(s['doses']) * len(s['schedules']) if a.phase == 'calibration' else 4)
        for row in catrows:
            assert [g['item'] for g in row['target']] == [x['id'] for x in s[a.phase]]
            expected = [] if a.phase == 'calibration' and row['condition'] == 'prompt_only' else [x['id'] for x in s['protection_' + a.phase]]
            assert [g['item'] for g in row['protection']] == expected
        if a.phase == 'calibration':
            for arm in ['candidate', 'random']:
                expected = [(schedule, dose) for schedule in s['schedules'] for dose in s['doses']]
                assert [(r['recipe']['schedule'], r['recipe']['dose']) for r in catrows if r.get('arm') == arm] == expected
        else:
            selection = read(result / 'selection.json')
            for row in catrows:
                if row.get('arm'): assert row['recipe'] == selection['choices'][category][row['arm']]['recipe']
            checks['selection_before_heldout'] = selection['sealed_at'] < read(result / 'start.json')['time']
            checks['selection_hash'] = sha(result / 'selection.json') == read(result / 'selection-input.json')['sha256']
            checks['scale_carryover'] = read(result / 'reference-scales.json') == selection['reference_scales']
    completion = read(result / 'completion.json'); runtime = read(result / 'runtime.json'); pid = completion['pid']
    checks['call_accounting'] = completion['calls'] == total_calls + (3 if a.phase == 'calibration' else 0)
    checks['completion'] = completion['status'] == 'completed' and completion['backwards'] == 0 and completion['calls'] <= s['phase_limits'][a.phase]['max_calls']
    checks['exit_zero'] = (workspace / 'logs' / (a.phase + '.exitcode')).read_text().strip() == '0'
    samples = [l for l in (workspace / 'logs' / (a.phase + '.telemetry')).read_text().splitlines() if re.match(rf'^{pid},', l)]
    checks['gpu_evidence'] = len(samples) >= 2 and runtime['pid'] == pid and runtime['transformer_parameters_all_cuda']
    checks['write_and_output_checks'] = True
    out = {'passed': all(checks.values()), 'checks': checks, 'completion': completion, 'gpu_samples': len(samples),
           'write_norm_relative_error': {'median': float(np.median(deviations)), 'max': float(max(deviations)), 'writes': len(deviations), 'zero_writes': zero_writes},
           'active_counts': active_counts, 'final_seal_sha256': sha(result / 'final-seal.json')}
    (workspace / (a.phase + '-verification.json')).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k,v in out.items() if k not in ['checks', 'active_counts']}, indent=2))
    if not out['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
