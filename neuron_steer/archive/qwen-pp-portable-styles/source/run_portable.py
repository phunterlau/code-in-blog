"""Portable single-neuron generation with verified post-SwiGLU activation writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from model_runtime import Runtime, load_text_model, metadata
from metrics import active, measure


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


class PortableRuntime(Runtime):
    def __init__(self, model, tok):
        self.control = None; self.generation_start = 0; self.ledger = []; self.scales = {}; self.column_norms = {}
        super().__init__(model, tok)

    def prepare(self, coordinate):
        l, n = coordinate
        key = (l, n)
        if key not in self.column_norms:
            self.column_norms[key] = float(self.layers[l].mlp.down_proj.weight[:, n].detach().float().norm())
        assert self.column_norms[key] > 0

    def pre_hook(self, layer):
        parent = super().pre_hook(layer)
        def hook(module, args):
            ordinary = parent(module, args)
            c = self.control
            if c is None or c['coordinate'][0] != layer: return ordinary
            step = self.calls - self.generation_start - 1
            assert step >= 0
            if not active(c['schedule'], step): return ordinary
            h = args[0]; changed = h.clone(); neuron = c['coordinate'][1]
            col = self.column_norms[layer, neuron]; reference = self.scales[layer]
            delta = c['dose'] * reference / col
            before = float(h[0, -1, neuron].detach())
            changed[:, -1, neuron] += torch.as_tensor(delta, dtype=h.dtype, device=h.device)
            after = float(changed[0, -1, neuron].detach())
            actual = after - before
            assert torch.isfinite(changed[:, -1, neuron]).all()
            self.ledger.append({'step': step, 'layer': layer, 'neuron': neuron, 'sequence_positions': int(h.shape[1]),
                                'edited_positions': 1, 'before': before, 'after': after, 'requested_delta': delta,
                                'actual_delta': actual, 'column_norm': col, 'reference_residual_norm': reference,
                                'requested_write_norm': abs(c['dose']) * reference, 'actual_write_norm': abs(actual) * col})
            return (changed,)
        return hook

    @torch.inference_mode()
    def generate_item(self, item, recipe, system, cap):
        ids = self.ids([{'role': 'system', 'content': system}, {'role': 'user', 'content': item['prompt']}])
        self.control = recipe; self.generation_start = self.calls; self.ledger = []
        eos = sorted(set(self.tok.convert_tokens_to_ids(['<|im_end|>', '<|endoftext|>'])))
        if recipe is not None: self.prepare(recipe['coordinate'])
        try:
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                      max_new_tokens=cap, use_cache=True, pad_token_id=self.tok.eos_token_id, eos_token_id=eos)
            assert out.is_cuda
            tokens = out[0, ids.shape[1]:].tolist(); calls = self.calls - self.generation_start
            assert calls == len(tokens)
            terminated = bool(tokens and tokens[-1] in eos)
            capped = len(tokens) == cap and not terminated
            text = self.tok.decode(tokens, skip_special_tokens=True)
            expected_steps = [i for i in range(calls) if recipe is not None and active(recipe['schedule'], i)]
            assert [w['step'] for w in self.ledger] == expected_steps
            return {'item': item['id'], 'prompt': item['prompt'], 'input_ids': ids[0].tolist(), 'system': system,
                    'text': text, 'tokens': tokens, 'calls': calls, 'cap': cap, 'capped': capped, 'terminated_eos': terminated,
                    'writes': list(self.ledger), 'metrics': measure(text, item, capped)}
        finally:
            self.control = None; self.ledger = []


def main():
    p = argparse.ArgumentParser(); p.add_argument('--protocol', required=True); p.add_argument('--out', required=True)
    p.add_argument('--phase', choices=['calibration', 'heldout'], required=True); p.add_argument('--selection')
    args = p.parse_args(); here = Path(__file__).resolve().parent
    s = json.loads(Path(args.protocol).read_text()); out = Path(args.out); out.mkdir(exist_ok=False)
    started = time.monotonic(); limits = s['phase_limits'][args.phase]; signal.alarm(limits['max_seconds'])
    torch.set_num_threads(4); torch.manual_seed(s['seed'])
    hashes = json.loads((here / 'source-hashes.json').read_text())
    for name, digest in hashes.items(): assert sha(here / name) == digest, name
    dump(out / 'source-hashes.json', hashes); dump(out / 'protocol.json', s)
    dump(out / 'start.json', {'pid': os.getpid(), 'phase': args.phase, 'time': time.time(), 'protocol_sha256': sha(Path(args.protocol))})
    model, tok = load_text_model(s['model_path']); model.requires_grad_(False)
    runtime = PortableRuntime(model, tok); runtime.max_calls = limits['max_calls']; dump(out / 'runtime.json', metadata(model))
    for cat in s['categories'].values():
        for arm in ['candidate', 'random']: runtime.prepare(cat[arm])
    if args.phase == 'calibration':
        ids = tok(s['reference_text'], add_special_tokens=False, return_tensors='pt').input_ids.to('cuda:0')
        _, _, residuals, _ = runtime.forward(ids, capture=True)
        runtime.scales = {l: float(values.median()) for l, values in residuals.items()}
        # An identity edit through the exact generation-time hook must preserve logits.
        ids = runtime.ids([{'role': 'user', 'content': 'Reply with the word ready.'}])
        baseline, *_ = runtime.forward(ids)
        runtime.control = {'coordinate': s['categories']['storyteller']['candidate'], 'dose': 0., 'schedule': 'pulse4'}
        runtime.generation_start = runtime.calls; runtime.ledger = []
        same, *_ = runtime.forward(ids)
        assert torch.equal(baseline, same) and len(runtime.ledger) == 1 and runtime.ledger[0]['actual_write_norm'] == 0
        runtime.control = None; runtime.ledger = []
        dump(out / 'smoke.json', {'passed': True, 'reference_calls': 1, 'identity_calls': 2})
        selection = None
    else:
        assert args.selection
        selection_path = Path(args.selection); selection = json.loads(selection_path.read_text())
        assert selection['protocol_sha256'] == sha(Path(args.protocol))
        assert selection['phase'] == 'sealed_calibration_selection'
        dump(out / 'selection.json', selection)
        dump(out / 'selection-input.json', {'sha256': sha(selection_path), 'sealed_at': selection['sealed_at']})
        runtime.scales = {int(k): v for k, v in selection['reference_scales'].items()}
    dump(out / 'reference-scales.json', runtime.scales)
    rows = []; records = out / 'records.jsonl'
    def ev(category, condition, recipe, items, kind, system):
        vals = []
        for item in items:
            value = runtime.generate_item(item, recipe, system, s['target_cap'] if kind == 'target' else s['protection_cap'])
            value.update(category=category, condition=condition, kind=kind, recipe=recipe)
            value['record_id'] = f'{category}:{condition}:{kind}:{item["id"]}'
            with records.open('a') as f:
                f.write(json.dumps(value, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
            vals.append(value)
        return vals
    for category, cat in s['categories'].items():
        items = s['calibration' if args.phase == 'calibration' else 'heldout']
        protections = s['protection_' + args.phase]
        for label, system in [('baseline', s['neutral_system']), ('prompt_only', cat['prompt_only_system'])]:
            row = {'category': category, 'condition': label, 'recipe': None,
                   'target': ev(category, label, None, items, 'target', system),
                   'protection': ev(category, label, None, protections, 'protection', system) if args.phase == 'heldout' or label == 'baseline' else []}
            rows.append(row); dump(out / 'rows.json', rows)
        for arm in ['candidate', 'random']:
            if args.phase == 'calibration':
                recipes = [{'coordinate': cat[arm], 'dose': dose, 'schedule': schedule} for schedule in s['schedules'] for dose in s['doses']]
            else:
                chosen = selection['choices'][category][arm]
                assert chosen['recipe']['coordinate'] == cat[arm]
                recipes = [chosen['recipe']]
            for j, recipe in enumerate(recipes):
                label = f'{arm}-{j:02}' if args.phase == 'calibration' else arm
                row = {'category': category, 'arm': arm, 'condition': label, 'recipe': recipe,
                       'target': ev(category, label, recipe, items, 'target', s['neutral_system']),
                       'protection': ev(category, label, recipe, protections, 'protection', s['neutral_system'])}
                rows.append(row); dump(out / 'rows.json', rows)
                print(args.phase.upper(), category, label, recipe['dose'], recipe['schedule'], 'calls', runtime.calls, flush=True)
    complete = {'status': 'completed', 'phase': args.phase, 'pid': os.getpid(), 'calls': runtime.calls, 'backwards': 0,
                'seconds': time.monotonic() - started, 'finished': time.time(), 'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
                'generations': sum(len(r['target']) + len(r['protection']) for r in rows)}
    dump(out / 'completion.json', complete)
    seal = {'sealed_at': time.time(), 'files': {str(f.relative_to(out)): sha(f) for f in sorted(out.iterdir()) if f.is_file()}}
    dump(out / 'final-seal.json', seal); print('COMPLETE', args.phase, runtime.calls, flush=True)


if __name__ == '__main__': main()
