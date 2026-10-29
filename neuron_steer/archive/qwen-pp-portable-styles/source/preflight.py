"""Local checks of the intervention schedule, output gates and frozen inputs."""
import ast
import hashlib
import json
from pathlib import Path
from metrics import active, joint, measure


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parent
    for p in root.glob('*.py'): ast.parse(p.read_text(), filename=str(p))
    s = json.loads((root / 'protocol.json').read_text())
    assert s['categories']['storyteller']['candidate'] == [26, 2167]
    assert s['categories']['nostalgic']['candidate'] == [22, 8168]
    assert [active('pulse4', i) for i in range(6)] == [True, True, True, True, False, False]
    assert all(active('continuous', i) for i in range(6))
    item = s['calibration'][1]
    text = 'The kettle asked the cup for an orchard song. The cup sang from the shelf until the kettle laughed.'
    m = measure(text, item, False)
    assert m['all_terms'] and m['format_pass']
    g = {'metrics': m}
    assert joint(g, {'style': 2, 'quality': 2})
    assert not joint(g, {'style': 1, 'quality': 2})
    assert not joint({'metrics': measure(text, item, True)}, {'style': 2, 'quality': 2})
    assert not measure('The kettle sits on a shelf. It is hot.', item, False)['all_terms']
    assert measure(' 25\n', s['protection_calibration'][0], False)['exact_correct']
    assert not measure('The answer is 25.', s['protection_calibration'][0], False)['exact_correct']
    seen = set()
    for key in ['calibration', 'heldout', 'protection_calibration', 'protection_heldout']:
        for x in s[key]:
            assert x['id'] not in seen; seen.add(x['id'])
    assert all(len(x) == 2 and x[0] in range(32) and x[1] in range(9216) for c in s['categories'].values() for x in [c['candidate'], c['random']])
    assert all(c['candidate'][0] == c['random'][0] and c['candidate'] != c['random'] for c in s['categories'].values())
    frozen = root.parents[1] / 'outputs/qwen-pp-inventory-blind/evidence/results/selections.json'
    previous = json.loads(frozen.read_text())
    assert sha(frozen) == s['selection_provenance']['sha256']
    assert all(c['candidate'] == previous[k]['pp1'][0] and c['random'] == previous[k]['random1'][0] for k, c in s['categories'].items())
    assert s['worst_case_forward_bound']['total'] == 18147 <= s['total_max_calls']
    result = {'passed': True, 'checks': ['syntax', 'pulse_scope', 'joint_gate', 'capped_failure', 'exact_protection', 'unique_prompts', 'fixed_coordinates', 'matched_random_layer', 'prior_selection_hash', 'worst_case_budget'], 'worst_case_forward_bound': s['worst_case_forward_bound']}
    (root / 'preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
