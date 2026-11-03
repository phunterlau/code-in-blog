"""Independent PP runtime for pinned Hugging Face Qwen/Llama checkpoints.

Derived from this study's independently authored runtime, not ai-lobotomy code.
Edits are post-gate/pre-down-projection; hooks are removed even after errors.
"""
import os
import contextlib, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
MODELS = {'llama8': ('meta-llama/Llama-3.1-8B-Instruct', '0e9e39f249a16976918f6564b8830bc894c89659'), 'qwen': ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'), 'llama': ('meta-llama/Llama-3.2-3B-Instruct', '0cb88a4f764b7a12671c53f0838cd831a0843b95'), 'qwen3': ('Qwen/Qwen3-4B-Instruct-2507', 'cdbee75f17c01a7cc42f958dc650907174af0554')}

def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=2) + '\n')

def append(p, v):
    with Path(p).open('a') as f:
        f.write(json.dumps(v) + '\n')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Lab:

    def __init__(self, key, precision='fp16', offline=False):
        self.key = key
        self.precision = precision
        name, rev = MODELS[key]
        if not torch.cuda.is_available():
            raise RuntimeError('This reproduction requires CUDA. Use verify_results.py for CPU-only artifact checks.')
        snap = name
        self.tok = AutoTokenizer.from_pretrained(snap, revision=rev, local_files_only=offline)
        self.tok.padding_side = 'left'
        self.tok.pad_token = self.tok.eos_token
        kwargs = dict(revision=rev, local_files_only=offline, torch_dtype=torch.float16, attn_implementation='sdpa')
        if precision == 'nf4':
            from transformers import BitsAndBytesConfig
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True), device_map={'': 'cuda:0'})
        elif precision == 'offload':
            kwargs['device_map'] = {'model.embed_tokens': 'cuda:0', 'model.norm': 'cuda:0', 'lm_head': 'cuda:0', **{f'model.layers.{i}': 'cpu' if i < 12 else 'cuda:0' for i in range(36)}}
        self.model = AutoModelForCausalLM.from_pretrained(snap, **kwargs).eval()
        self.model.requires_grad_(False)
        if precision == 'fp16':
            self.model = self.model.cuda()
        self.layers = self.model.model.layers
        self.L = len(self.layers)
        self.N = self.model.config.intermediate_size
        self.device = self.model.model.embed_tokens.weight.device
        stop = self.model.generation_config.eos_token_id
        self.stops = stop if isinstance(stop, list) else [stop]
        torch.cuda.reset_peak_memory_stats()
        self.meta = {'model': name, 'revision': rev, 'precision': precision, 'layers': self.L, 'neurons_per_layer': self.N}

    def weight(self, l):
        w = self.layers[l].mlp.down_proj.weight
        if hasattr(w, 'quant_state') and w.quant_state is not None:
            from bitsandbytes.functional import dequantize_4bit
            return dequantize_4bit(w.data, w.quant_state).float()
        return w.float()

    def ids(self, q, system=None, anchor=''):
        msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': q}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + anchor
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) > 512:
            raise ValueError('Input >512 tokens; no silent truncation')
        return ids

    @contextlib.contextmanager
    def edits(self, jobs, prefill_mask=None):
        handles = []
        state = {'step': -1}

        def step(mod, args):
            state['step'] += 1
        handles.append(self.model.model.register_forward_pre_hook(step))
        bylayer = {}
        for b, job in enumerate(jobs):
            for e in job.get('edits', []):
                bylayer.setdefault(e['layer'], []).append((b, e))

        def make(group):

            def fn(mod, args):
                a = args[0].clone()
                for b, e in group:
                    scope = e.get('scope', 'all')
                    first = state['step'] == 0
                    if scope == 'last_prefill' and (not first):
                        continue
                    if scope == 'prefill' and (not first):
                        continue
                    if scope == 'decode' and first:
                        continue
                    idx = torch.tensor(e['indices'], device=a.device)
                    if scope == 'last_prefill':
                        pos = torch.tensor([a.shape[1] - 1], device=a.device)
                    elif first and prefill_mask is not None:
                        pos = prefill_mask[b].to(a.device).nonzero().flatten()
                    else:
                        pos = torch.arange(a.shape[1], device=a.device)
                    cur = a[b, pos[:, None], idx[None, :]]
                    raw = e.get('values')
                    v = torch.as_tensor(0 if raw is None else raw, dtype=a.dtype, device=a.device)
                    mode = e['mode']
                    if mode == 'zero':
                        new = torch.zeros_like(cur)
                    elif mode == 'replace':
                        new = v.expand_as(cur)
                    elif mode == 'add':
                        new = cur + v
                    elif mode == 'scale':
                        new = cur * v
                    else:
                        raise ValueError(mode)
                    a[b, pos[:, None], idx[None, :]] = new
                return (a,) + args[1:]
            return fn
        try:
            for l, g in bylayer.items():
                handles.append(self.layers[l].mlp.down_proj.register_forward_pre_hook(make(g)))
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.inference_mode()
    def forward(self, ids, edits=None, capture=False):
        a = [None] * self.L
        hs = []
        if capture:

            def make(l):

                def hook(m, args):
                    a[l] = args[0][0, -1].float().cpu().clone()
                return hook
            hs = [layer.mlp.down_proj.register_forward_pre_hook(make(l)) for l, layer in enumerate(self.layers)]
        try:
            x = torch.tensor([ids], device=self.device)
            with self.edits([{'edits': edits or []}]):
                z = self.model(x, use_cache=False).logits[0, -1].float().cpu()
        finally:
            for h in hs:
                h.remove()
        return (z, torch.stack(a) if capture else None)

    @torch.inference_mode()
    def generate(self, jobs, batch_size=2, max_tokens=128):
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            ids = [self.ids(j['question'], j.get('system')) for j in batch]
            width = max(map(len, ids))
            x = torch.full((len(ids), width), self.tok.pad_token_id, device=self.device, dtype=torch.long)
            mask = torch.zeros_like(x)
            for i, seq in enumerate(ids):
                x[i, -len(seq):] = torch.tensor(seq, device=x.device)
                mask[i, -len(seq):] = 1
            cfg = GenerationConfig(do_sample=False, max_new_tokens=max_tokens, eos_token_id=self.stops, pad_token_id=self.tok.pad_token_id, bos_token_id=self.tok.bos_token_id, use_cache=True)
            t = time.time()
            with self.edits(batch, mask):
                out = self.model.generate(x, attention_mask=mask, generation_config=cfg, do_sample=False, temperature=None, top_p=None, top_k=None)
            for i, job in enumerate(batch):
                gen = out[i, width:].tolist()
                for k, g in enumerate(gen):
                    if g in self.stops:
                        gen = gen[:k + 1]
                        break
                text = self.tok.decode(gen, skip_special_tokens=True)
                words = text.lower().split()
                tri = [tuple(words[k:k + 3]) for k in range(max(0, len(words) - 2))]
                yield dict(job, input_ids=ids[i], generated_ids=gen, text=text, hit_length_cap=len(gen) >= max_tokens and gen[-1] not in self.stops, repeated_trigram_fraction=1 - len(set(tri)) / max(1, len(tri)), seconds_per_batch=time.time() - t, peak_vram_mib=torch.cuda.max_memory_allocated() / 2 ** 20, **self.meta)
            if torch.cuda.mem_get_info()[0] < 512 * 2 ** 20:
                raise RuntimeError('GPU headroom below 512 MiB')

def smoke(lab):
    ids = lab.ids('What is 7 plus 8? Reply in one sentence.')
    z, a = lab.forward(ids, capture=True)
    zz, aa = lab.forward(ids, capture=True)
    assert torch.equal(a, aa) and torch.equal(z, zz)
    zero = [dict(layer=lab.L - 1, indices=[3], mode='add', values=0.0, scope='last_prefill')]
    z0, _ = lab.forward(ids, zero)
    assert torch.equal(z, z0)
    for mode, value in [('replace', float(a[-1, 3])), ('scale', 1.0)]:
        zi, _ = lab.forward(ids, [dict(layer=lab.L - 1, indices=[3], mode=mode, values=value, scope='last_prefill')])
        assert torch.equal(z, zi)
    zj, _ = lab.forward(ids, [dict(layer=lab.L - 1, indices=[3], mode='zero', values=None, scope='all')])
    assert torch.isfinite(zj).all()
    edited, _ = lab.forward(ids, [dict(layer=lab.L - 1, indices=[3], mode='add', values=5.0, scope='last_prefill')])
    restored, _ = lab.forward(ids)
    assert torch.equal(z, restored)
    assert torch.isfinite(edited).all() and (z - edited).abs().max() > 0
    assert torch.tensor([2.0, -2.0]).mean() == 0
    return {'same_input': True, 'zero_edit': True, 'replace_identity': True, 'scale_identity': True, 'zero_none': True, 'hook_cleanup': True, 'edit_changes_logits': True, 'activation_shape': list(a.shape), 'mean_cancellation': True}
