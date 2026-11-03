"""Optional sequential CUDA validation on already cached checkpoints.

Pass paths to the two Python environments described in the README.
This runs six demo generations, a small self-grade, and Qwen3 discovery.
"""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core-python', required=True)
    parser.add_argument('--qwen3-python', required=True)
    parser.add_argument('--out', default='scratch/package-validation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if busy.strip():
        raise RuntimeError('GPU already has a compute process; validation not started')
    steps = [
        ('llama_openai', args.core_python, 'llama', ['demo', '--example', 'openai']),
        ('llama_bard', args.core_python, 'llama', ['demo', '--example', 'bard']),
        ('llama_grades', args.core_python, 'llama', ['grade', '--dataset', str(out / 'llama_openai')]),
        ('qwen3_openai', args.qwen3_python, 'qwen3', ['demo', '--example', 'openai']),
        ('qwen3_discovery', args.qwen3_python, 'qwen3', ['discover']),
    ]
    status = {}
    for label, python, model, command in steps:
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if busy.strip():
            raise RuntimeError('GPU acquired another compute process; remaining validation not started')
        print('START', label, flush=True)
        with (out / f'{label}.log').open('w') as log:
            result = subprocess.run([python, '-u', str(root / 'experiment.py'), '--model', model,
                                     '--offline', '--out', str(out / label), *command],
                                    cwd=root, stdout=log, stderr=subprocess.STDOUT)
        status[label] = result.returncode
        (out / 'status.json').write_text(json.dumps(status, indent=2))
        if result.returncode:
            raise RuntimeError(f'{label} failed; see its log')
        print('DONE', label, flush=True)


if __name__ == '__main__':
    main()
