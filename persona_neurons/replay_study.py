"""Replay the six archived generation datasets and their same-model grading.

The frozen jobs retain exact dose/control choices. This does not select new
candidates; use experiment.py discover and sweep for a new search.
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-python', required=True)
    parser.add_argument('--qwen3-python', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stages = [('llama', 'single_identity_llama', 'self_single_llama', 96),
              ('qwen3', 'single_identity_qwen3', 'self_single_qwen3', 96),
              ('qwen', 'single_identity_qwen', 'self_single_qwen', 96),
              ('llama', 'confirm_single_llama', 'self_confirm_llama', 128),
              ('qwen3', 'confirm_single_qwen3', 'self_confirm_qwen3', 128),
              ('llama', 'confirm_single_llama_refine', 'self_confirm_llama_refine', 128)]
    status = {}
    for model, dataset, grades, cap in stages:
        python = args.qwen3_python if model == 'qwen3' else args.core_python
        steps = [(dataset, ['--max-tokens', str(cap), '--batch-size', '2' if model == 'llama' else '4',
                            'replay', '--jobs', str(root / 'results' / dataset / 'jobs.json')]),
                 (grades, ['grade', '--dataset', str(out / dataset)])]
        for label, command in steps:
            busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            if busy.strip():
                raise RuntimeError('GPU has another compute process; replay not started')
            cmd = [python, '-u', str(root / 'experiment.py'), '--model', model, '--out', str(out / label)]
            if args.offline:
                cmd.append('--offline')
            print('START', label, flush=True)
            with (out / f'{label}.log').open('w') as log:
                result = subprocess.run(cmd + command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            status[label] = result.returncode
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            if result.returncode:
                raise RuntimeError(f'{label} failed; inspect its log')
            print('DONE', label, flush=True)


if __name__ == '__main__':
    main()
