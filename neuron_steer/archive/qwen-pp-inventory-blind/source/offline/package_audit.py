"""Verify collected evidence, links, source seals, and a portable package manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--original-evidence'); a = p.parse_args()
    root = Path(a.root).resolve(); checks = {}; result = root / 'evidence/results'
    for name, digest in read(result / 'final-seal.json')['files'].items():
        checks['final_seal:' + name] = sha(result / name) == digest
    checks['mapping_hash'] = sha(root / 'reference-mapping.json') == read(root / 'crosscheck.json')['reference_mapping_sha256']
    for name, digest in read(root / 'source/source-hashes.json').items():
        checks['source_hash:' + name] = sha(root / 'source' / name) == digest
    if a.original_evidence:
        original = Path(a.original_evidence)
        old = {str(f.relative_to(original)): sha(f) for f in original.rglob('*') if f.is_file()}
        new = {str(f.relative_to(root / 'evidence')): sha(f) for f in (root / 'evidence').rglob('*') if f.is_file()}
        checks['collected_evidence_exactly_matches_audited_mirror'] = old == new
    missing = []
    for f in root.rglob('*.md'):
        # External reference snapshots have their own relative links and are not
        # locally authored deliverable navigation.
        if 'reference-snapshots' in f.parts: continue
        for target in re.findall(r'\]\(([^)]+)\)', f.read_text()):
            if re.match(r'[a-zA-Z]+://', target) or target.startswith('#'): continue
            target = target.split('#')[0]
            if target == 'MANIFEST.sha256': continue  # Written below.
            path = (f.parent / target).resolve()
            if not path.exists(): missing.append({'file': str(f.relative_to(root)), 'target': target})
    checks['local_markdown_links'] = not missing
    checks['offline_scientific_verification'] = read(root / 'verification.json')['passed']
    summary = read(root / 'descriptive-analysis.json')['totals']
    checks['numerical_summary'] = summary['heldout_pairs'] == 128 and summary['pp16_restore_beats_norm'] == 128 and summary['pp16_reverse_beats_norm'] == 128 and summary['distinct_endpoint_denominator'] == 117
    final = {'passed': all(checks.values()), 'checks': checks, 'missing_links': missing}
    (root / 'package-verification.json').write_text(json.dumps(final, indent=2) + '\n')
    if not final['passed']:
        print(json.dumps({'passed': False, 'failures': [k for k,v in checks.items() if not v], 'missing': missing}, indent=2))
        raise SystemExit(1)
    files = sorted(f for f in root.rglob('*') if f.is_file() and f.name not in ['MANIFEST.sha256', '.DS_Store'] and '__pycache__' not in f.parts)
    manifest = ''.join(f'{sha(f)}  {f.relative_to(root)}\n' for f in files)
    (root / 'MANIFEST.sha256').write_text(manifest)
    for line in manifest.splitlines():
        digest, path = line.split('  ', 1); assert sha(root / path) == digest
    print(json.dumps({'passed': True, 'checks': len(checks), 'manifest_files': len(files), 'total_bytes': sum(f.stat().st_size for f in files)}, indent=2))


if __name__ == '__main__': main()
