"""PP-only discovery, sealed selection, validation, and a final pre-reference seal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import torch
from model_runtime import Runtime, load_text_model, metadata


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def install_guard(workspace, model_path):
    """A Python file-read guard and trace, not an operating-system sandbox."""
    roots = [workspace.resolve(), Path(model_path).resolve(), Path(sys.prefix).resolve(),
             Path('/usr'), Path('/lib'), Path('/lib64'), Path('/etc'), Path('/proc'), Path('/sys'), Path('/dev')]
    accesses, rejected = set(), []

    def audit(event, args):
        if event in ['socket.connect', 'socket.getaddrinfo']:
            rejected.append(dict(event=event))
            raise PermissionError('Network access is excluded from this search.')
        if event != 'open' or not args or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        allowed = any(path == root or root in path.parents for root in roots)
        if not allowed:
            rejected.append(dict(event=event, path=str(path)))
            raise PermissionError('Non-allowlisted input path: ' + str(path))
        accesses.add((str(path), str(args[1])))
    sys.addaudithook(audit)
    return roots, accesses, rejected


class SearchRuntime(Runtime):
    def __init__(self, model, tok):
        self.write_ledger = []
        super().__init__(model, tok)

    def pre_hook(self, layer):
        parent = super().pre_hook(layer)

        def hook(module, args):
            changed = parent(module, args)
            if changed is not None and layer in self.edits:
                neurons = sorted({n for e in self.edits[layer] for n in e['neurons']})
                before = args[0][0, -1, neurons].detach().float()
                after = changed[0][0, -1, neurons].detach().float()
                delta = after - before
                weight = self.layers[layer].mlp.down_proj.weight[:, neurons].detach().float()
                self.write_ledger.append(dict(layer=layer, neurons=neurons,
                                              before=before.cpu().tolist(), after=after.cpu().tolist(), delta=delta.cpu().tolist(),
                                              write_norm=float((weight @ delta).norm())))
            return changed
        return hook

    @torch.inference_mode()
    def native_generation(self, item, side, cap):
        ids = self.ids(item[side], item.get('suffix', ''))
        before = self.calls
        eos = list(set(self.tok.convert_tokens_to_ids(['<|im_end|>', '<|endoftext|>'])))
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                  max_new_tokens=cap, use_cache=True, pad_token_id=self.tok.eos_token_id, eos_token_id=eos)
        assert out.is_cuda
        tokens = out[0, ids.shape[1]:].tolist()
        assert self.calls - before == len(tokens)
        return dict(pair=item['id'], side=side, text=self.tok.decode(tokens, skip_special_tokens=True),
                    tokens=tokens, capped=len(tokens) == cap, calls=len(tokens))


def observable(model, tok, target, control):
    def token_ids(words):
        values = set()
        for word in words:
            for variant in [word, ' ' + word]:
                ids = tok.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    values.add(ids[0])
        return sorted(values)
    ti, ci = token_ids(target), token_ids(control)
    assert ti and ci and not set(ti) & set(ci)
    w = model.lm_head.weight.detach()
    direction = w[ti].float().mean(0) - w[ci].float().mean(0)
    coupling = torch.stack([(layer.mlp.down_proj.weight.detach().float().T @ direction.to('cuda:0')).cpu() for layer in model.model.layers])
    return ti, ci, coupling


def gap(logits, target_ids, control_ids):
    return float(logits[target_ids].mean() - logits[control_ids].mean())


def capture_pairs(runtime, pairs, target_ids, control_ids):
    base, target, parents = [], [], []
    for pair in pairs:
        row = dict(pair=pair['id'])
        for side, arrays in [('base', base), ('target', target)]:
            ids = runtime.ids(pair[side], pair.get('suffix', ''))
            z, acts, *_ = runtime.forward(ids, capture=True)
            tensor = torch.stack([acts[l] for l in range(len(runtime.layers))])
            arrays.append(tensor)
            row[side] = dict(ids=ids[0].tolist(), gap=gap(z, target_ids, control_ids),
                             argmax=int(z.argmax()), argmax_text=runtime.tok.decode([int(z.argmax())]))
        parents.append(row)
    return torch.stack(base), torch.stack(target), parents


def evaluate(runtime, coordinate_set, source, donor, parent, side, target_ids, control_ids, norm_reference=None):
    grouped = {}
    for layer, neuron in coordinate_set:
        grouped.setdefault(layer, []).append(neuron)
    edits, planned = {}, []
    for layer, neurons in sorted(grouped.items()):
        delta = donor[layer, neurons] - source[layer, neurons]
        weight = runtime.layers[layer].mlp.down_proj.weight[:, neurons].detach().float()
        norm = float((weight @ delta.to('cuda:0')).norm())
        factor, fallback = 1., False
        if norm_reference is not None:
            if norm == 0:
                delta = torch.ones_like(delta)
                norm = float((weight @ delta.to('cuda:0')).norm())
                fallback = True
            assert norm > 0
            factor = norm_reference[layer] / norm
            delta = delta * factor
        operation = 'add' if norm_reference is not None else 'set'
        values = delta if operation == 'add' else source[layer, neurons] + delta
        edits[layer] = [dict(operation=operation, neurons=neurons, values=values.tolist())]
        planned.append(dict(layer=layer, neurons=neurons, operation=operation, values=values.tolist(),
                            norm_match_factor=factor, zero_direction_fallback=fallback,
                            target_write_norm=norm_reference[layer] if norm_reference is not None else None))
    runtime.write_ledger = []
    z, *_ = runtime.forward(torch.tensor([parent[side]['ids']], device='cuda:0'), edits=edits)
    delta = gap(z, target_ids, control_ids) - parent[side]['gap']
    natural_gap = parent['target']['gap'] - parent['base']['gap']
    return dict(delta=delta, natural_gap=natural_gap, fraction=delta / natural_gap if natural_gap > 1e-6 else None,
                argmax=int(z.argmax()), argmax_text=runtime.tok.decode([int(z.argmax())]),
                target_argmax_match=int(z.argmax()) == parent['target']['argmax'],
                base_argmax_match=int(z.argmax()) == parent['base']['argmax'],
                writes=list(runtime.write_ledger), planned_writes=planned)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parent
    protocol_path = Path(args.protocol).resolve()
    assert protocol_path.parent == workspace
    s = json.loads(protocol_path.read_text())
    out = Path(args.out).resolve()
    assert out.parent == workspace
    out.mkdir(exist_ok=False)
    roots, accesses, rejected = install_guard(workspace, s['model_path'])
    started = time.monotonic()
    signal.alarm(s['max_seconds'])
    torch.set_num_threads(4)
    torch.manual_seed(s['seed'])
    hashes = json.loads((workspace / 'source-hashes.json').read_text())
    for name, expected in hashes.items():
        assert sha(workspace / name) == expected, name
    dump(out / 'source-hashes.json', hashes)
    dump(out / 'protocol.json', s)
    dump(out / 'start.json', dict(pid=os.getpid(), time=time.time(), protocol_sha256=sha(protocol_path)))
    model, tok = load_text_model(s['model_path'])
    model.requires_grad_(False)
    runtime = SearchRuntime(model, tok)
    runtime.max_calls = s['max_forward_calls']
    dump(out / 'runtime.json', metadata(model))
    # Structural identity check at the first valid coordinate; no semantic hint.
    first = next(iter(s['tasks'].values()))['discovery'][0]
    ids = runtime.ids(first['base'], first.get('suffix', ''))
    z, acts, *_ = runtime.forward(ids, capture=True)
    runtime.write_ledger = []
    same, *_ = runtime.forward(ids, edits={0: [dict(operation='set', neurons=[0], values=[float(acts[0][0])])]})
    assert torch.equal(z, same)
    assert len(runtime.write_ledger) == 1 and runtime.write_ledger[0]['write_norm'] == 0
    dump(out / 'smoke.json', dict(passed=True, forward_calls=2, coordinate_rule='first valid coordinate; identity write only'))

    cached = {}
    selections = {}
    for category_index, (name, task) in enumerate(s['tasks'].items()):
        dest = out / name
        dest.mkdir()
        ti, ci, coupling = observable(model, tok, task['target_tokens'], task['control_tokens'])
        base, target, parents = capture_pairs(runtime, task['discovery'], ti, ci)
        importance = (target - base) * coupling
        signed = importance.mean(0)
        width = signed.shape[1]
        order = signed.flatten().argsort(descending=True, stable=True)
        positive = order[signed.flatten()[order] > 0]
        assert len(positive) >= max(s['selection_k'])
        pp50 = [list(divmod(int(j), width)) for j in positive[:50]]
        rng = random.Random(s['seed'] + category_index)
        forbidden = {tuple(c) for c in pp50}
        random50 = []
        for layer, _ in pp50:
            while True:
                neuron = rng.randrange(width)
                if (layer, neuron) not in forbidden:
                    break
            forbidden.add((layer, neuron))
            random50.append([layer, neuron])
        groups = dict(pp50=pp50, random50=random50)
        selected = {f'pp{k}': pp50[:k] for k in s['selection_k']}
        selected.update({f'random{k}': random50[:k] for k in s['selection_k']})
        selections[name] = selected
        assert torch.equal(base, base.half().float()) and torch.equal(target, target.half().float())
        torch.save(dict(base=base.half(), target=target.half(), coupling=coupling, signed_mean=signed), dest / 'discovery.pt')
        dump(dest / 'discovery-parents.json', parents)
        dump(dest / 'observable.json', dict(target_ids=ti, control_ids=ci, target_text=[tok.decode([i]) for i in ti], control_text=[tok.decode([i]) for i in ci]))
        dump(dest / 'groups.json', groups)
        dump(dest / 'ranking-top100.json', [dict(coordinate=list(divmod(int(j), width)), signed_mean=float(signed.flatten()[j]),
                                                  pair_importances=importance.reshape(len(parents), -1)[:, j].tolist()) for j in order[:100]])
        cached[name] = dict(target_ids=ti, control_ids=ci, coupling=coupling, selections=selected)
        print('DISCOVERY', name, 'complete; calls', runtime.calls, flush=True)

    dump(out / 'selections.json', selections)
    commitment = dict(time=time.time(), calls=runtime.calls, selection_sha256=sha(out / 'selections.json'),
                      discovery_files={str(f.relative_to(out)): sha(f) for name in s['tasks'] for f in sorted((out / name).iterdir()) if f.is_file()})
    dump(out / 'selection-seal.json', commitment)
    print('ALL_SELECTIONS_SEALED', runtime.calls, flush=True)

    for name, task in s['tasks'].items():
        dest = out / name
        data = cached[name]
        ti, ci, selected = data['target_ids'], data['control_ids'], data['selections']
        base, target, parents = capture_pairs(runtime, task['validation'], ti, ci)
        dump(dest / 'validation-parents.json', parents)
        coordinates = sorted({tuple(c) for arm in ['pp16', 'random16'] for c in selected[arm]})
        dump(dest / 'validation-donors.json', [dict(coordinate=list(c), base=base[:, c[0], c[1]].tolist(), target=target[:, c[0], c[1]].tolist()) for c in coordinates])
        rows = []
        for i, parent in enumerate(parents):
            for arm in ['pp1', 'pp4', 'pp16', 'random1', 'random4', 'random16']:
                for mode in ['restore', 'reverse']:
                    side = 'base' if mode == 'restore' else 'target'
                    source, donor = (base[i], target[i]) if mode == 'restore' else (target[i], base[i])
                    result = evaluate(runtime, selected[arm], source, donor, parent, side, ti, ci)
                    rows.append(dict(pair=parent['pair'], arm=arm, mode=mode, coordinates=selected[arm], **result))
            for mode in ['restore', 'reverse']:
                reference = next(row for row in rows if row['pair'] == parent['pair'] and row['arm'] == 'pp16' and row['mode'] == mode)
                norms = {w['layer']: w['write_norm'] for w in reference['writes']}
                side = 'base' if mode == 'restore' else 'target'
                source, donor = (base[i], target[i]) if mode == 'restore' else (target[i], base[i])
                result = evaluate(runtime, selected['random16'], source, donor, parent, side, ti, ci, norms)
                rows.append(dict(pair=parent['pair'], arm='norm_random16', mode=mode, coordinates=selected['random16'], **result))
        dump(dest / 'interventions.json', rows)
        generations = []
        for i in s['native_generation_pairs']:
            for side in ['base', 'target']:
                generations.append(runtime.native_generation(task['validation'][i], side, s['generation_cap']))
        dump(dest / 'native-generations.json', generations)
        print('VALIDATION', name, 'complete; calls', runtime.calls, flush=True)

    completion = dict(status='completed', pid=os.getpid(), calls=runtime.calls,
                      scientific_forward_calls=runtime.calls - 2, smoke_forward_calls=2, backwards=0,
                      seconds=time.monotonic() - started, finished=time.time(), cuda_peak_allocated=torch.cuda.max_memory_allocated())
    dump(out / 'completion.json', completion)
    dump(out / 'input-audit.json', dict(scope='Python open events from guard installation through all model work; not an OS sandbox.',
                                      allowed_roots=[str(p) for p in roots], declared_read_before_guard=str(protocol_path),
                                      accesses=[dict(path=p, mode=m) for p, m in sorted(accesses)], blocked_attempts=rejected,
                                      trace_cutoff_time=time.time()))
    final_files = {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'final-seal.json'}
    dump(out / 'final-seal.json', dict(sealed_at=time.time(), phase='before reference disclosure', files=final_files,
                                     selection_sha256=sha(out / 'selections.json'), source_hashes_sha256=sha(out / 'source-hashes.json')))
    print('FINAL_SEAL_COMPLETE', json.dumps(completion), flush=True)


if __name__ == '__main__':
    main()
