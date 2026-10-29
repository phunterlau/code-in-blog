"""Post-seal descriptive analysis. No model calls or candidate reselection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from torch_numpy import load


def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--deliverable', required=True); a = p.parse_args()
    root, dest = Path(a.root), Path(a.deliverable)
    result = root / 'evidence/results'
    analysis, selections = read(root / 'analysis.json'), read(result / 'selections.json')
    refs = read(dest / 'reference-mapping.json'); cross = read(dest / 'crosscheck.json')
    known = {tuple(c) for r in refs['references'] for c in r['coordinates']}
    social_known = {tuple(c) for cs in refs['secondary_user_quoted_social']['coordinates'].values() for c in cs}
    # This supplemental source was fetched only after the search and final seal.
    # Include all four coordinates in its alien-pathway paragraph, plus the original
    # steering positive control; no outcome-based choice between these references.
    alien_reference = [
        {'coordinate': [22, 2742], 'role': 'Original alien steering positive control; the GitHub author reports coherent additive rollouts.'},
        {'coordinate': [27, 8347], 'role': 'Reported downstream write-path coordinate, labeled planet/star; no standalone generation reproduction asserted here.'},
        {'coordinate': [31, 7843], 'role': 'Reported downstream write-path coordinate, labeled Ali/alien.'},
        {'coordinate': [7, 8842], 'role': 'Reported upstream read-path coordinate, labeled scouts/science fiction; adjacent pathway evidence, not the same phenotype.'},
    ]
    supplemental = {'status': 'Post-seal GitHub source cross-check; not part of the precommitted primary catalog.',
                    'source': 'https://github.com/lyramakesmusic/neuron-explorer/blob/6a6f0e95bf49ad6626940021148f775d7d55d1e4/research_log.md',
                    'source_blob_sha': '1ee3d62e770ad909df4c699c16f93afd12cac9cb',
                    'snapshot_sha256': sha(dest / 'reference-snapshots/explorer-research-log.md'),
                    'selection_sha256': sha(result / 'selections.json'), 'final_seal_sha256': sha(result / 'final-seal.json'),
                    'compiled_at': time.time(), 'alien_references': alien_reference}
    summary = {'categories': {}, 'supplemental_github': supplemental}
    for name, cat in analysis['categories'].items():
        cap = load(result / name / 'discovery.pt'); mu = cap['signed_mean']; width = mu.shape[1]
        order = np.argsort(-mu.reshape(-1), kind='stable'); ranks = np.empty_like(order); ranks[order] = np.arange(len(order)) + 1
        delta = (cap['target'].astype(np.float32) - cap['base'].astype(np.float32)).mean(axis=0)
        if name == 'alien':
            for row in alien_reference:
                l, n = row['coordinate']; row.update(signed_rank=int(ranks[l * width + n]), signed_mean=float(mu[l, n]),
                                                     mean_activation_delta=float(delta[l, n]), pp50_hit=row['coordinate'] in selections[name]['pp50'])
        raw = read(result / name / 'interventions.json')
        norms = []
        for row in raw:
            if row['arm'] == 'norm_random16':
                ref = next(x for x in raw if x['pair'] == row['pair'] and x['mode'] == row['mode'] and x['arm'] == 'pp16')
                for w in row['writes']:
                    rn = next(x['write_norm'] for x in ref['writes'] if x['layer'] == w['layer'])
                    norms.append({'pair': row['pair'], 'mode': row['mode'], 'layer': w['layer'], 'reference_norm': rn,
                                  'actual_random_norm': w['write_norm'], 'absolute_error': abs(w['write_norm'] - rn),
                                  'relative_error': abs(w['write_norm'] - rn) / rn if rn else None})
        contrasts = {}
        for mode in ['restore', 'reverse']:
            pp = np.array(cat['arms']['pp16'][mode]['values'])
            norm = np.array(cat['arms']['norm_random16'][mode]['values'])
            natural = np.array(cat['arms']['random16'][mode]['values'])
            oriented = 1 if mode == 'restore' else -1
            d = (pp - norm) * oriented
            contrasts[mode] = {'pp16_beats_norm_random16': int((d > 0).sum()), 'n': len(d),
                               'mean_oriented_difference': float(d.mean()), 'values': d.tolist(),
                               'pp16_beats_natural_random16': int(((pp - natural) * oriented > 0).sum())}
        candidates = []
        ranking = read(result / name / 'ranking-top100.json')
        for row in ranking[:50]:
            l, n = row['coordinate']
            candidates.append({**row, 'signed_rank': int(ranks[l * width + n]), 'mean_activation_delta': float(delta[l, n]),
                               'coupling': float(cap['coupling'][l, n]), 'positive_discovery_pairs': sum(v > 0 for v in row['pair_importances']),
                               'in_primary_catalog': (l, n) in known, 'in_user_quoted_social_list': (l, n) in social_known})
        a1 = cat['arms']['pp1']; a16 = cat['arms']['pp16']
        summary['categories'][name] = {'top50': candidates, 'matched_control_contrasts': contrasts,
            'top1_validation': {'restore_mean': a1['restore']['mean'], 'restore_positive': a1['restore']['positive'],
                                'reverse_mean': a1['reverse']['mean'], 'reverse_negative': a1['reverse']['negative']},
            'pp16_ratio_of_means': a16['restore']['mean'] / cat['validation_gap']['mean'],
            'norm_matching_worst_relative': max(norms, key=lambda x: x['relative_error'] or 0),
            'pp50_layers': dict(Counter(c[0] for c in selections[name]['pp50']))}
        del cap, mu, delta
    pp50_union = {tuple(c) for cat in selections.values() for c in cat['pp50']}
    top1_union = {tuple(cat['top1']) for cat in analysis['categories'].values()}
    summary['totals'] = {'categories': len(analysis['categories']), 'heldout_pairs': 8 * len(analysis['categories']),
        'top50_coordinate_slots': sum(len(cat['pp50']) for cat in selections.values()), 'unique_top50_coordinates': len(pp50_union),
        'unique_top1_coordinates': len(top1_union), 'unique_top50_in_primary_catalog': len(pp50_union & known),
        'unique_top1_absent_primary_catalog': len(top1_union - known),
        'pp16_restore_positive': sum(c['arms']['pp16']['restore']['positive'] for c in analysis['categories'].values()),
        'pp16_reverse_negative': sum(c['arms']['pp16']['reverse']['negative'] for c in analysis['categories'].values()),
        'pp16_restore_beats_norm': sum(c['matched_control_contrasts']['restore']['pp16_beats_norm_random16'] for c in summary['categories'].values()),
        'pp16_reverse_beats_norm': sum(c['matched_control_contrasts']['reverse']['pp16_beats_norm_random16'] for c in summary['categories'].values()),
        'pp16_target_argmax_matches_distinct_endpoints': sum(c['arms']['pp16']['new_target_argmax_matches'] for c in analysis['categories'].values()),
        'distinct_endpoint_denominator': sum(c['arms']['pp16']['distinct_endpoint_denominator'] for c in analysis['categories'].values()),
        'native_generations': sum(len(c['native_generations']) for c in analysis['categories'].values()),
        'native_capped': sum(g['capped'] for c in analysis['categories'].values() for g in c['native_generations'])}
    # Preserve the frozen substring result and expose obvious semantic mismatches.
    # This sensitivity analysis changes neither candidates nor the primary endpoint.
    boundary_misses = []
    for ref in refs['references']:
        text = ' '.join(ref['title_fields'].values()).lower()
        text = ' '.join(re.sub(r'[_\-·]+', ' ', text).split())
        for category, aliases in ref['primary_categories'].items():
            if not any(re.search(r'(?<!\w)' + re.escape(alias.lower().replace('-', ' ')) + r'(?!\w)', text) for alias in aliases):
                boundary_misses.append({'reference_id': ref['id'], 'title': ref['title'], 'category': category, 'substring_aliases': aliases})
    summary['primary_matching_limitations'] = {'substring_without_word_boundary_matches': boundary_misses,
        'warning': 'The frozen substring matcher can confuse universe with verse. The primary zero-hit result is retained, not repaired after seeing outcomes. Catalog coverage is incomplete; absent title matches are not recovery failures.'}
    (dest / 'descriptive-analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    (dest / 'postseal-github-crosscheck.json').write_text(json.dumps(supplemental, indent=2) + '\n')
    print(json.dumps({'totals': summary['totals'], 'github': alien_reference, 'mapping_limitations': boundary_misses}, indent=2))


if __name__ == '__main__': main()
