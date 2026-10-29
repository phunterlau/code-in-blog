"""Derive an immutable group study from sealed PP discoveries, without reranking."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from torch_numpy import load
from geometry import expected_calls


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prior', required=True); p.add_argument('--portable', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(); prior = Path(a.prior); out = Path(a.out)
    assert not out.exists(), 'Never overwrite the frozen protocol.'
    evidence = prior / 'evidence/results'
    sealed = json.loads((evidence / 'final-seal.json').read_text())
    selection = json.loads((evidence / 'selections.json').read_text())
    old = json.loads((evidence / 'protocol.json').read_text())
    assert sha(evidence / 'selections.json') == sealed['selection_sha256']
    requests = [
        'Describe a bicycle leaning against a closed bookstore.',
        'Give a brief account of someone repairing a torn map.',
        'Explain how to organize a community seed exchange.',
        'Describe a street after the first snowfall of the season.',
        'Give a short description of an empty theater before a performance.',
        'Describe a family sorting a box of unfamiliar keys.',
        'Explain what happens when a small bridge is closed for repairs.',
        'Describe a lantern being carried up a stone staircase.',
    ]
    old_requests = {m['content'] for t in old['tasks'].values() for split in ['discovery', 'validation']
                    for pair in t[split] for side in ['base', 'target'] for m in pair[side] if m['role'] == 'user'}
    portable = json.loads(Path(a.portable).read_text())
    old_requests.update(x['prompt'] for split in ['calibration', 'heldout'] for x in portable[split])
    assert not set(requests) & old_requests
    s = dict(version='pp-group-controls-v1', seed=2026102907, model_path=old['model_path'], model_revision=old['model_revision'],
             categories={}, requests=requests, reference_text=portable['reference_text'],
             max_calls=5200, max_seconds=1500, expected_calls=0, doses=[.1, .4],
             selection='Existing signed-mean PP top 1/4/16; no reranking or new coordinate search.',
             provenance={'prior_final_seal_sha256': sha(evidence / 'final-seal.json'), 'prior_selection_sha256': sha(evidence / 'selections.json'), 'files': {}},
             interpretation='Fixed discovery-mean directions; last-prompt-token lexical readout assay, not full generation or a semantic-quality test.',
             primary_effect_threshold=.05, no_retry='No outcome-based changes or scientific retries.',
             protections=[
                 {'id':'p1','prompt':'Compute 8 + 9. If the result is 17 answer A; if it is 16 answer B. Output only the letter.','expected':'A'},
                 {'id':'p2','prompt':'Compute 7 * 6. If the result is 41 answer A; if it is 42 answer B. Output only the letter.','expected':'B'},
                 {'id':'p3','prompt':'The vase is blue. A means the vase is blue; B means it is red. Output only the correct letter.','expected':'A'},
                 {'id':'p4','prompt':'Choose the word that comes first alphabetically: A means zebra; B means apple. Output only the letter.','expected':'B'},
                 {'id':'p5','prompt':'Which integer is smaller? A means 14; B means 19. Output only the letter.','expected':'A'},
                 {'id':'p6','prompt':'A box contains exactly three coins. A means two coins; B means three coins. Output only the true letter.','expected':'B'},
                 {'id':'p7','prompt':'Return the letter A and nothing else.','expected':'A'},
                 {'id':'p8','prompt':'Return the letter B and nothing else.','expected':'B'},
             ])
    for ci, cat in enumerate(['losing_language', 'academic', 'storyteller', 'nostalgic']):
        for filename in ['discovery.pt', 'observable.json']:
            rel = f'{cat}/{filename}'; digest = sha(evidence / rel)
            assert digest == sealed['files'][rel]
            s['provenance']['files'][rel] = digest
        data = load(evidence / cat / 'discovery.pt')
        coordinates = selection[cat]['pp16']
        assert coordinates[:1] == selection[cat]['pp1'] and coordinates[:4] == selection[cat]['pp4']
        mean = (data['target'].astype(np.float64) - data['base'].astype(np.float64)).mean(axis=0)
        means = [float(mean[l,n]) for l,n in coordinates]
        randoms = [selection[cat]['random16']]
        used = {tuple(x) for x in coordinates + randoms[0]}
        for mask in [1, 2]:
            rng = random.Random(s['seed'] + ci * 100 + mask)
            picked = []
            for l, _ in coordinates:
                available = [n for n in range(mean.shape[1]) if (l,n) not in used]
                n = rng.choice(available); used.add((l,n)); picked.append([l,n])
            randoms.append(picked)
        task = old['tasks'][cat]
        s['categories'][cat] = {'pp16': coordinates, 'mean_activation_delta': means, 'random16': randoms,
                                'base_system': task['discovery'][0]['base'][0]['content'],
                                'target_system': task['discovery'][0]['target'][0]['content'],
                                'observable': json.loads((evidence / cat / 'observable.json').read_text())}
    s['expected_calls'] = expected_calls(s)
    assert s['expected_calls'] == 4899 < s['max_calls']
    out.write_text(json.dumps(s, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'protocol':str(out),'sha256':sha(out),'expected_forward_calls':s['expected_calls'],'model_calls_during_preparation':0}))


if __name__ == '__main__': main()
