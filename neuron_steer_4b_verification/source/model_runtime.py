"""Read-only checkpoint loading and exact post-SwiGLU hooks for this pilot."""
import gc
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding


class BoundaryEmbedding(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, ids):
        return torch.nn.functional.embedding(ids.cpu(), self.weight).to("cuda:0", dtype=torch.float16)


class BoundaryHead(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden.float().cpu(), self.weight).to("cuda:0")


class HybridQwen(Qwen3_5ForCausalLM):
    @property
    def device(self):
        return torch.device("cuda:0")


def load_text_model(path, dtype=torch.float16):
    """Load only language weights; no quantization, vision, or random parameters."""
    path = Path(path)
    config = AutoConfig.from_pretrained(path, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = HybridQwen(config)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    states = {}
    for shard in sorted(set(index.values())):
        with safe_open(path / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.language_model."):
                    new_key = "model." + key.removeprefix("model.language_model.")
                    states[new_key] = f.get_tensor(key).to(dtype)
                elif key == "lm_head.weight":
                    states[key] = f.get_tensor(key).to(dtype)
    if "lm_head.weight" not in states:
        states["lm_head.weight"] = states["model.embed_tokens.weight"]
    model.load_state_dict(states, strict=True, assign=True)
    model.tie_weights()
    # Rotary frequencies are nonpersistent buffers, absent from the checkpoint.
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config).to("cuda:0")
    del states
    gc.collect()
    # The GPU has 7.6 GiB usable VRAM. Keep the tied vocabulary matrix on CPU.
    # Its FP16 values are losslessly promoted for the CPU output dot product.
    vocab_weight = torch.nn.Parameter(model.model.embed_tokens.weight.float(), requires_grad=False)
    model.model.embed_tokens = BoundaryEmbedding(vocab_weight)
    model.lm_head = BoundaryHead(vocab_weight)
    for layer in model.model.layers:
        layer.to("cuda:0")
    model.model.norm.to("cuda:0")
    model.eval()
    assert all(p.device == torch.device("cuda:0") for layer in model.model.layers for p in layer.parameters())
    assert not any(p.is_meta for p in model.parameters())
    assert not any(b.is_meta for b in model.buffers())
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tok


class Runtime:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.layers = model.model.layers
        self.capture = False
        self.edits = {}
        self.acts, self.resids = {}, {}
        self.full_acts = {}
        self.capture_full = False
        self.calls = 0
        self.max_calls = 3000
        self.handles = []
        self.handles.append(model.register_forward_pre_hook(self.count_call))
        for l, layer in enumerate(self.layers):
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self.pre_hook(l)))
            self.handles.append(layer.register_forward_hook(self.post_hook(l)))

    def count_call(self, mod, args):
        self.calls += 1
        if self.calls > self.max_calls:
            raise RuntimeError("frozen forward budget exceeded")

    def pre_hook(self, l):
        def hook(mod, args):
            h = args[0]
            if self.capture:
                self.acts[l] = h[0, -1].detach().float().cpu()
                if self.capture_full:
                    self.full_acts[l] = h[0].detach().float().cpu()
            if l not in self.edits:
                return None
            h = h.clone()
            for edit in self.edits[l]:
                indices = edit["neurons"]
                positions = slice(None) if edit.get("all_tokens") else -1
                if edit["operation"] == "add":
                    h[:, positions, indices] += torch.as_tensor(edit["values"], device=h.device, dtype=h.dtype)
                elif edit["operation"] == "set":
                    h[:, positions, indices] = torch.as_tensor(edit["values"], device=h.device, dtype=h.dtype)
                elif edit["operation"] == "scale":
                    h[:, positions, indices] *= edit["values"]
                else:
                    raise ValueError(edit)
            return (h,)
        return hook

    def post_hook(self, l):
        def hook(mod, args, out):
            if self.capture:
                h = out[0] if isinstance(out, tuple) else out
                self.resids[l] = h[0, 1:].detach().float().norm(dim=-1).cpu()
        return hook

    def ids(self, messages, suffix=""):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(text + suffix, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")

    @torch.inference_mode()
    def forward(self, ids, edits=None, capture=False, full=False):
        self.capture, self.capture_full = capture, full
        self.edits = edits or {}
        self.acts, self.resids, self.full_acts = {}, {}, {}
        try:
            out = self.model(ids, use_cache=False, logits_to_keep=1)
            assert out.logits.device == torch.device("cuda:0")
            logits = out.logits[0, -1].float().cpu()
            assert torch.isfinite(logits).all()
            return logits, dict(self.acts), dict(self.resids), dict(self.full_acts)
        finally:
            self.capture, self.capture_full, self.edits = False, False, {}

    @torch.inference_mode()
    def generate(self, ids, edits=None, max_new_tokens=48):
        self.edits = edits or {}
        try:
            out = self.model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens,
                                      use_cache=True, pad_token_id=self.tok.eos_token_id,
                                      eos_token_id=self.tok.eos_token_id)
            assert out.device == torch.device("cuda:0")
            return {"text":self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True),
                    "token_ids":out[0, ids.shape[1]:].tolist(),
                    "length_capped":out.shape[1]-ids.shape[1] == max_new_tokens}
        finally:
            self.edits = {}


def metadata(model):
    import importlib.metadata
    return {"pid":os.getpid(), "time":time.time(), "device":"cuda:0",
            "dtype":str(next(model.parameters()).dtype),
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "versions":{x:importlib.metadata.version(x) for x in ["torch","transformers","safetensors","accelerate"]},
            "gpu":torch.cuda.get_device_name(), "capability":torch.cuda.get_device_capability(),
            "parameters_all_cuda":all(p.is_cuda for p in model.parameters()),
            "transformer_parameters_all_cuda":all(p.is_cuda for layer in model.model.layers for p in layer.parameters()),
            "cpu_boundary":"tied embedding and output head, float32; transformer float16 CUDA",
            "parameter_devices":{str(d):sum(p.numel() for p in model.parameters() if p.device.type==d) for d in ["cpu","cuda"]}}
