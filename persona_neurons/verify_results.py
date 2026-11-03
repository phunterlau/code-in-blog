"""CPU-only integrity, signed-mean and blog-example checks (requires NumPy)."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify():
    hashes = read(ROOT / 'provenance/artifact_sha256.json')
    for relative, expected in hashes.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    totals = dict(generations=0, ratings=0, calibration_ratings=0)
    sources = {'single_identity': 'single_identity.py', 'confirm_single': 'confirm_single.py'}
    for directory in sorted((ROOT / 'results').iterdir()):
        metadata = read(directory / 'manifest.json')
        assert metadata['status'] == 'complete', directory.name
        output_file = 'scores.jsonl' if directory.name.startswith('self_') else 'outputs.jsonl'
        output = rows(directory / output_file)
        jobs = read(directory / 'jobs.json')
        assert len(output) == metadata['planned'] == len(jobs)
        assert len(output) == len({r['id'] for r in output})
        assert {r['id'] for r in output} == {r['id'] for r in jobs}
        if output_file == 'scores.jsonl':
            totals['ratings'] += len(output)
            calibration = read(directory / 'calibration.json')
            assert len(calibration) == 8
            totals['calibration_ratings'] += len(calibration)
            script = 'grade_outputs.py'
            for rubric, valid in metadata['rubric_anchor_pass'].items():
                anchors = {r['pole']: r['score'] for r in calibration if r['rubric'] == rubric}
                assert (anchors['high'] - anchors['low'] >= 3) == valid
            lookup = {}
            for row in output:
                if row['dataset'] not in lookup:
                    lookup[row['dataset']] = {r['id']: r for r in rows(
                        ROOT / 'results' / row['dataset'] / 'outputs.jsonl')}
                source = lookup[row['dataset']][row['source_id']]
                assert row['text'] == source['text'] and row['question'] == source['question']
        else:
            totals['generations'] += len(output)
            script = next(v for k, v in sources.items() if directory.name.startswith(k))
            assert metadata['smoke']['hook_cleanup']
        assert metadata['script_sha256'] == hashlib.sha256((ROOT / 'historical' / script).read_bytes()).hexdigest()
        assert metadata['runtime_sha256'] == hashlib.sha256((ROOT / 'historical/lab_runtime.py').read_bytes()).hexdigest()
        for target in ['OpenAI', 'Google']:
            file = directory / f'{target}_scores.npz'
            if not file.exists():
                continue
            with np.load(file, allow_pickle=False) as data:
                score = data['coupling'] * data['scenario_delta'].mean(0)
                assert np.allclose(score, data['score'], atol=2e-6)
                rank = np.argsort(-np.abs(score.flatten()))
                width = score.shape[1]
                top = read(directory / f'{target}_top100.json')
                assert all(r['layer'] * width + r['index'] == int(i) for r, i in zip(top, rank[:100]))
                if directory.name == 'single_identity_qwen3':
                    actual = int(np.flatnonzero(rank == 24 * width + 386)[0]) + 1
                    recorded = next(r for r in rows(directory / 'benchmark_rank.jsonl') if r['target'] == target)
                    assert actual == recorded['rank']
                    print(f'Qwen3 L24:386 {target} rank: {actual}')
    assert totals == {'generations': 1174, 'ratings': 1328, 'calibration_ratings': 48}, totals
    for example in read(ROOT / 'analysis/blog_examples.json'):
        lookup = {r['id']: r for r in rows(ROOT / 'results' / example['dataset'] / 'outputs.jsonl')}
        for label in ['baseline', 'edited']:
            assert example[label] == lookup[example[label]['id']]
    print(json.dumps({'verified_artifacts': len(hashes), **totals}, indent=2))
    return totals


if __name__ == '__main__':
    verify()
