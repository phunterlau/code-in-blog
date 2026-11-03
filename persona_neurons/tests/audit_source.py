"""Optional source-overlap audit; never used by experiments or model loading.

Example: python tests/audit_source.py --reference /path/to/ai-lobotomy
The report lists counts and paths, not reference source text.
"""
import argparse
import ast
import collections
import hashlib
import io
import json
import subprocess
import tokenize
from pathlib import Path


def code_tokens(path):
    ignored = {tokenize.STRING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
               tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER}
    return [t.string for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
            if t.type not in ignored]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    width = 40
    index = collections.defaultdict(set)
    for file in args.reference.rglob('*.py'):
        if any(part.startswith('.') for part in file.relative_to(args.reference).parts):
            continue
        try:
            tokens = code_tokens(file)
        except (SyntaxError, tokenize.TokenError, UnicodeDecodeError):
            continue
        for i in range(len(tokens) - width + 1):
            index[tuple(tokens[i:i + width])].add(str(file.relative_to(args.reference)))
    commit = subprocess.check_output(['git', '-C', str(args.reference), 'rev-parse', 'HEAD'], text=True).strip()
    report = {'reference_commit': commit,
              'token_window': width, 'excluded': 'strings, comments, whitespace/layout',
              'qualification': 'Overlap screening supports, but does not prove, independent authorship.',
              'files': {}}
    for file in sorted(root.rglob('*.py')):
        if any(part in ['scratch', '.venv-core', '.venv-qwen3'] for part in file.relative_to(root).parts):
            continue
        tokens = code_tokens(file)
        matches = collections.Counter()
        for i in range(len(tokens) - width + 1):
            for source in index.get(tuple(tokens[i:i + width]), []):
                matches[source] += 1
        imports = []
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == 'persona_lab' or name.startswith('persona_lab.') for name in imports)
        report['files'][str(file.relative_to(root))] = {
            'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
            'matching_windows': dict(matches), 'imports': sorted(set(imports)),
        }
    text = json.dumps(report, indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
