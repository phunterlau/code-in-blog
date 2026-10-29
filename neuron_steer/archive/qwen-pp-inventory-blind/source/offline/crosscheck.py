"""Post-seal reference compilation and exact-coordinate matching; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from torch_numpy import load


def read(path): return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def normalized(text):
    return ' '.join(re.sub(r'[_\-·]+', ' ', text.lower()).split())


def model_matches(record):
    return 'qwen354b' in re.sub(r'[^a-z0-9]', '', str(record.get('model', record.get('model_id', ''))).lower())


def coordinates(record):
    result = []
    for c in record.get('neurons', record.get('coordinates', [])):
        if isinstance(c, dict) and 'layer' in c and 'neuron' in c:
            result.append([int(c['layer']), int(c['neuron'])])
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--deliverable', required=True); a = p.parse_args()
    root = Path(a.root); deliverable = Path(a.deliverable)
    assert read(root / 'verification.json')['passed']
    result = root / 'evidence/results'; protocol = read(result / 'protocol.json')
    disclosure = read(deliverable / 'reference-disclosure.json')
    assert disclosure['final_seal_sha256'] == sha(result / 'final-seal.json')
    assert disclosure['reference_disclosure_started'] > read(result / 'final-seal.json')['sealed_at']
    for ref in disclosure['references']:
        assert sha(deliverable / ref['snapshot']) == ref['sha256']
    snapshots = deliverable / 'reference-snapshots'
    recipes = read(snapshots / 'neuronary-inventory.json')
    presets = read(snapshots / 'peft-preset-index.json')
    refs = []
    for kind, records in [('community_catalog', recipes['recipes']), ('git_preset', presets['presets'])]:
        for i, record in enumerate(records):
            if not model_matches(record): continue
            cs = coordinates(record)
            if not cs: continue
            fields = {k: record[k] for k in protocol['comparison_rule']['title_fields'] if isinstance(record.get(k), str)}
            fields_text = normalized(' '.join(fields.values()))
            description = normalized(str(record.get('description', '')))
            matches, secondary = {}, {}
            for category, aliases in protocol['comparison_rule']['aliases'].items():
                hit = [alias for alias in aliases if normalized(alias) in fields_text]
                if hit: matches[category] = hit
                hit = [alias for alias in aliases if normalized(alias) in description]
                if hit and category not in matches: secondary[category] = hit
            refs.append(dict(id=f'{kind}:{record.get("id", record.get("name", i))}', kind=kind,
                             title=record.get('name', record.get('title', record.get('label', ''))), title_fields=fields,
                             coordinates=cs, primary_categories=matches, description_only_categories=secondary,
                             url=record.get('url', presets['source'] if kind == 'git_preset' else recipes['source']),
                             raw_record=record))
    # Supplemental coordinates are transcribed from the user's quoted social post,
    # only after the search seal. They do not enter the primary catalogue matcher.
    social = {
        'oracle': [[11, 1185]], 'losing_language': [[10, 5002]], 'nursery_rhyme': [[6, 1525]],
        'salesperson': [[4, 7242]], 'building': [[3, 3066]], 'corporation': [[4, 4783]],
        'victorian': [[15, 5080]], 'academic': [[7, 950]], 'human': [[7, 6297]],
        'worrier': [[18, 6874]], 'safety': [[9, 1932]], 'storyteller': [[19, 264], [21, 5992]], 'nostalgic': [[20, 7]],
    }
    mapping = dict(compiled_at=time.time(), rule=protocol['comparison_rule'], references=refs,
                   secondary_user_quoted_social=dict(source='https://www.linkedin.com/feed/update/urn:li:activity:7521233770402897920/',
                                                    status='User-quoted claims; separate from frozen primary snapshots and not independently reproduced here.', coordinates=social))
    mapping_path = deliverable / 'reference-mapping.json'
    assert not mapping_path.exists(), 'Preserve the first compiled reference map.'
    mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')
    mapping_hash = sha(mapping_path)
    # Only now load the PP selections/scores for comparison.
    selections = read(result / 'selections.json')
    all_known = {tuple(c) for r in refs for c in r['coordinates']}
    outcome = dict(reference_mapping_sha256=mapping_hash, categories={}, reference_records=len(refs), distinct_reference_coordinates=len(all_known))
    for category in protocol['categories']:
        cap = load(result / category / 'discovery.pt'); mu = cap['signed_mean']; width = mu.shape[1]
        flat = mu.reshape(-1); order = np.argsort(-flat, kind='stable'); inverse = np.empty_like(order); inverse[order] = np.arange(len(order)) + 1
        mean_delta = (cap['target'].astype(np.float32) - cap['base'].astype(np.float32)).mean(axis=0)
        relevant = [r for r in refs if category in r['primary_categories']]
        valid = sorted({tuple(c) for r in relevant for c in r['coordinates']})
        candidate_sets = {k: {tuple(c) for c in selections[category][f'pp{k}']} for k in protocol['selection_k']}
        random_set = {tuple(c) for c in selections[category]['random50']}

        def info(c):
            l, n = c
            return dict(coordinate=list(c), rank=int(inverse[l * width + n]), signed_score=float(mu[l, n]),
                        discovery_mean_activation_delta=float(mean_delta[l, n]))
        coordinate_rows = []
        for c in valid:
            evidence = [r for r in relevant if list(c) in r['coordinates']]
            coordinate_rows.append(dict(**info(c), references=[dict(id=r['id'], title=r['title'], kind=r['kind'], url=r['url']) for r in evidence]))
        coordinate_rows.sort(key=lambda x: x['rank'])
        recipe_hits = []
        for r in relevant:
            cs = {tuple(c) for c in r['coordinates']}
            recipe_hits.append(dict(reference_id=r['id'], title=r['title'], kind=r['kind'], coordinate_count=len(cs),
                                    pp50_components=[list(c) for c in sorted(cs & candidate_sets[50])],
                                    pp50_all_coordinates_recovered=cs <= candidate_sets[50]))
        any_known = sorted(candidate_sets[50] & all_known)
        outcome['categories'][category] = dict(reference_title_records=len(relevant), reference_coordinate_count=len(valid),
                                               coordinate_ranks=coordinate_rows,
                                               primary_hits={str(k): [list(c) for c in valid if c in candidate_sets[k]] for k in protocol['selection_k']},
                                               random50_primary_hits=[list(c) for c in valid if c in random_set],
                                               recipe_hits=recipe_hits,
                                               all_catalog_pp50_overlaps=[dict(**info(c), titles=[r['title'] for r in refs if list(c) in r['coordinates']]) for c in any_known],
                                               all_catalog_random50_overlaps=[list(c) for c in sorted(random_set & all_known)],
                                               secondary_user_quoted_social=[info(tuple(c)) for c in social.get(category, [])])
        del cap, mu, flat, order, inverse, mean_delta
    outcome['summary'] = dict(categories=len(outcome['categories']),
                              categories_with_title_matched_references=sum(x['reference_coordinate_count'] > 0 for x in outcome['categories'].values()),
                              categories_with_primary_top50_hit=sum(bool(x['primary_hits']['50']) for x in outcome['categories'].values()),
                              categories_with_random50_primary_hit=sum(bool(x['random50_primary_hits']) for x in outcome['categories'].values()),
                              unique_primary_top50_coordinates=len({tuple(c) for x in outcome['categories'].values() for c in x['primary_hits']['50']}),
                              unique_all_catalog_top50_overlaps=len({tuple(c['coordinate']) for x in outcome['categories'].values() for c in x['all_catalog_pp50_overlaps']}),
                              social_coordinates=sum(len(x['secondary_user_quoted_social']) for x in outcome['categories'].values()),
                              social_top50_hits=sum(c['rank'] <= 50 and c['signed_score'] > 0 for x in outcome['categories'].values() for c in x['secondary_user_quoted_social']))
    (deliverable / 'crosscheck.json').write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps(outcome['summary'], indent=2))
    for category, row in outcome['categories'].items():
        print(category, 'reference coordinates', row['reference_coordinate_count'], 'top50', row['primary_hits']['50'],
              'best', row['coordinate_ranks'][0]['rank'] if row['coordinate_ranks'] else None,
              'social', [(c['coordinate'], c['rank']) for c in row['secondary_user_quoted_social']])


if __name__ == '__main__': main()
