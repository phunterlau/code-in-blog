"""Offline audit of sealed PP-only outputs. Does not open reference inventories."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import numpy as np
from torch_numpy import load


def read(path): return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def stats(values):
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values), values=values) if values else None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True); a = parser.parse_args()
    root = Path(a.root); ev = root / 'evidence'; result = ev / 'results'
    s = read(result / 'protocol.json'); final = read(result / 'final-seal.json'); selection_seal = read(result / 'selection-seal.json')
    selections = read(result / 'selections.json'); checks = {}
    for name, digest in final['files'].items(): checks['final_seal_' + name] = sha(result / name) == digest
    checks['final_selection_hash'] = sha(result / 'selections.json') == final['selection_sha256'] == selection_seal['selection_sha256']
    checks['final_source_hash'] = sha(result / 'source-hashes.json') == final['source_hashes_sha256']
    for name, digest in selection_seal['discovery_files'].items(): checks['selection_seal_' + name] = sha(result / name) == digest
    for name, digest in read(result / 'source-hashes.json').items(): checks['source_' + name] = sha(ev / name) == digest == sha(root / 'source' / name)
    checks['selections_before_validation'] = selection_seal['calls'] == 2 + len(s['tasks']) * 16
    checks['prompt_only_task_schema'] = all(set(t) == {'discovery', 'validation', 'target_tokens', 'control_tokens', 'prompt_origin'} for t in s['tasks'].values())
    report = dict(categories={}); generation_calls = 0; norm_errors = []; norm_absolute = []; norm_fallbacks = 0
    donor_rounding = []
    for category_index, (name, task) in enumerate(s['tasks'].items()):
        folder = result / name
        tensors = load(folder / 'discovery.pt')
        b, t, c = tensors['base'].astype(np.float32), tensors['target'].astype(np.float32), tensors['coupling']
        importance = (t - b) * c
        mu = importance.mean(axis=0)
        saved = tensors['signed_mean']
        checks[name + '_signed_mean'] = bool(np.allclose(mu, saved, atol=1e-7, rtol=1e-5))
        width = saved.shape[1]; flat = saved.reshape(-1); order = np.argsort(-flat, kind='stable'); positive = order[flat[order] > 0]
        expected = [[int(j // width), int(j % width)] for j in positive[:50]]
        checks[name + '_pp50_exact'] = expected == selections[name]['pp50']
        rng = random.Random(s['seed'] + category_index); forbidden = {tuple(x) for x in expected}; random50 = []
        for layer, _ in expected:
            while True:
                neuron = rng.randrange(width)
                if (layer, neuron) not in forbidden: break
            forbidden.add((layer, neuron)); random50.append([layer, neuron])
        checks[name + '_random50_exact'] = random50 == selections[name]['random50']
        for k in s['selection_k']:
            checks[f'{name}_prefix{k}'] = selections[name][f'pp{k}'] == expected[:k] and selections[name][f'random{k}'] == random50[:k]
        top = read(folder / 'ranking-top100.json')
        checks[name + '_top100'] = [x['coordinate'] for x in top] == [[int(j // width), int(j % width)] for j in order[:100]]
        dp = read(folder / 'discovery-parents.json'); vp = read(folder / 'validation-parents.json')
        rows = read(folder / 'interventions.json'); donors = {tuple(x['coordinate']): x for x in read(folder / 'validation-donors.json')}
        assert len(rows) == 8 * 14
        parent_by_id = {x['pair']: x for x in vp}
        for i, parent in enumerate(vp):
            for mode in ['restore', 'reverse']:
                reference = next(x for x in rows if x['pair'] == parent['pair'] and x['arm'] == 'pp16' and x['mode'] == mode)
                random_norm = next(x for x in rows if x['pair'] == parent['pair'] and x['arm'] == 'norm_random16' and x['mode'] == mode)
                norms = {w['layer']: w['write_norm'] for w in reference['writes']}
                for w in random_norm['planned_writes']:
                    assert w['target_write_norm'] == norms[w['layer']]
                    norm_fallbacks += int(w['zero_direction_fallback'])
                for w in random_norm['writes']:
                    absolute = abs(w['write_norm'] - norms[w['layer']]); norm_absolute.append(absolute)
                    if norms[w['layer']] > 1e-6: norm_errors.append(absolute / norms[w['layer']])
            for row in [x for x in rows if x['pair'] == parent['pair']]:
                arm = 'random16' if row['arm'] == 'norm_random16' else row['arm']
                assert row['coordinates'] == selections[name][arm]
                for w in row['writes']:
                    assert np.allclose(np.array(w['after']) - np.array(w['before']), w['delta'], atol=1e-8, rtol=1e-6)
                    if row['arm'] != 'norm_random16':
                        side = 'target' if row['mode'] == 'restore' else 'base'
                        expected_values = [donors[w['layer'], neuron][side][i] for neuron in w['neurons']]
                        source_side = 'base' if side == 'target' else 'target'
                        source_values = np.array([donors[w['layer'], neuron][source_side][i] for neuron in w['neurons']], dtype=np.float32)
                        expected_values = np.array(expected_values, dtype=np.float32)
                        # Reproduce the frozen program's float32 subtraction/addition,
                        # followed by the actual FP16 input store. Do not hide deviations
                        # from an exact donor copy; quantify them separately.
                        reconstructed = source_values + (expected_values - source_values)
                        planned = next(p for p in row['planned_writes'] if p['layer'] == w['layer'])
                        by_neuron = dict(zip(planned['neurons'], planned['values']))
                        actual_planned = np.array([by_neuron[n] for n in w['neurons']], dtype=np.float32)
                        assert np.array_equal(reconstructed, actual_planned)
                        assert np.array_equal(np.array(w['after']), actual_planned.astype(np.float16).astype(np.float32))
                        difference = float(np.max(np.abs(np.array(w['after']) - expected_values)))
                        if difference:
                            donor_rounding.append(dict(category=name, pair=row['pair'], arm=row['arm'], mode=row['mode'], layer=w['layer'], max_absolute_error=difference))
        arms = {}
        for arm in ['pp1', 'pp4', 'pp16', 'random1', 'random4', 'random16', 'norm_random16']:
            forward = [x for x in rows if x['arm'] == arm and x['mode'] == 'restore']
            reverse = [x for x in rows if x['arm'] == arm and x['mode'] == 'reverse']
            different = [x for x in forward if parent_by_id[x['pair']]['base']['argmax'] != parent_by_id[x['pair']]['target']['argmax']]
            arms[arm] = dict(restore=stats([x['delta'] for x in forward]), reverse=stats([x['delta'] for x in reverse]),
                             fraction=stats([x['fraction'] for x in forward if x['fraction'] is not None]),
                             new_target_argmax_matches=sum(x['target_argmax_match'] for x in different), distinct_endpoint_denominator=len(different))
        native = read(folder / 'native-generations.json')
        assert len(native) == 4
        for row in native:
            assert row['calls'] == len(row['tokens']) and row['capped'] == (row['calls'] == s['generation_cap'])
            generation_calls += row['calls']
        report['categories'][name] = dict(top1=selections[name]['pp1'][0], top16=selections[name]['pp16'],
                                         discovery_gap=stats([x['target']['gap'] - x['base']['gap'] for x in dp]),
                                         validation_gap=stats([x['target']['gap'] - x['base']['gap'] for x in vp]),
                                         arms=arms, native_generations=native)
        checks[name + '_validation_donor_and_ledger_checks'] = True
        del tensors, b, t, c, importance, mu, saved
    audit = read(result / 'input-audit.json'); allowed = [Path(p) for p in audit['allowed_roots']]
    checks['observed_open_paths_allowlisted'] = all(any(Path(x['path']) == p or p in Path(x['path']).parents for p in allowed) for x in audit['accesses'])
    checks['no_blocked_input_attempts'] = not audit['blocked_attempts']
    completion = read(result / 'completion.json'); pid = completion['pid']; runtime = read(result / 'runtime.json')
    expected_calls = 2 + len(s['tasks']) * (16 + 16 + 8 * 14) + generation_calls
    checks['forward_ledger'] = completion['calls'] == expected_calls and completion['scientific_forward_calls'] == expected_calls - 2
    checks['completion'] = completion['status'] == 'completed' and completion['backwards'] == 0 and completion['calls'] <= s['max_forward_calls']
    checks['seal_after_completion'] = final['sealed_at'] >= completion['finished'] > selection_seal['time']
    checks['exit_zero'] = (ev / 'logs/main.exitcode').read_text().strip() == '0'
    telemetry = [line for line in (ev / 'logs/main.telemetry').read_text().splitlines() if re.match(rf'^{pid},', line)]
    checks['gpu_evidence'] = len(telemetry) > 1 and runtime['pid'] == pid and runtime['transformer_parameters_all_cuda']
    report.update(completion=completion, pid_correlated_gpu_samples=len(telemetry),
                  donor_rounding=dict(deviating_layer_replacements=len(donor_rounding), max_absolute_error=max([x['max_absolute_error'] for x in donor_rounding], default=0.), details=donor_rounding),
                  norm_matching=dict(max_relative_error=max(norm_errors), median_relative_error=float(np.median(norm_errors)), max_absolute_error=max(norm_absolute), layer_comparisons=len(norm_absolute), zero_direction_fallbacks=norm_fallbacks),
                  selection_seal_sha256=sha(result / 'selection-seal.json'), final_seal_sha256=sha(result / 'final-seal.json'))
    (root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    verification = dict(passed=all(checks.values()), checks=checks, completion=completion, pid_correlated_gpu_samples=len(telemetry), norm_matching=report['norm_matching'], donor_rounding=report['donor_rounding'])
    (root / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(dict(passed=verification['passed'], failed=[k for k, v in checks.items() if not v], completion=completion, gpu_samples=len(telemetry)), indent=2))
    if not verification['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
