"""Fixed-dose group interventions; no validation donors, reranking or generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from model_runtime import Runtime, load_text_model, metadata
from geometry import key, layer_of, scale, norms, make_vectors, causal_conditions, protection_conditions, expected_calls


def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, obj):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


class GroupRuntime(Runtime):
    def __init__(self, model, tok):
        self.vector = {}; self.ledger = []
        super().__init__(model, tok)

    def pre_hook(self, layer):
        parent = super().pre_hook(layer)
        def hook(module, args):
            ordinary = parent(module, args)
            values = {k:v for k,v in self.vector.items() if layer_of(k) == layer}
            if not values: return ordinary
            h = args[0]; edited = h.clone()
            names = list(values); indices = [int(k.split('N')[1]) for k in names]
            before = h[0,-1,indices].detach().float().cpu().tolist()
            add = torch.tensor(list(values.values()), dtype=h.dtype, device=h.device)
            edited[:,-1,indices] += add
            assert torch.isfinite(edited[:,-1,indices]).all()
            after = edited[0,-1,indices].detach().float().cpu().tolist()
            self.ledger.extend({'coordinate':k,'before':b,'after':a,'requested_delta':values[k],
                                'actual_delta':a-b,'sequence_positions':int(h.shape[1]),'edited_positions':1}
                               for k,b,a in zip(names,before,after))
            return (edited,)
        return hook

    def trial(self, ids, vector):
        self.vector = vector; self.ledger = []
        before = self.calls
        try:
            z, *_ = self.forward(ids)
            assert self.calls == before + 1
            assert {w['coordinate'] for w in self.ledger} == set(vector)
            return z, list(self.ledger)
        finally:
            self.vector = {}; self.ledger = []


def statistics(z, baseline, target_ids, control_ids):
    q = torch.log_softmax(z, dim=-1)
    p = torch.log_softmax(baseline, dim=-1)
    return {'readout':float(z[target_ids].mean() - z[control_ids].mean()),
            'target_ids':target_ids, 'control_ids':control_ids,
            'target_logits':z[target_ids].tolist(), 'control_logits':z[control_ids].tolist(),
            'argmax':int(z.argmax()), 'argmax_logit':float(z.max()),
            'kl_baseline_to_edited':max(0., float((p.exp() * (p - q)).sum()))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--protocol', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); here = Path(__file__).resolve().parent
    s = json.loads(Path(args.protocol).read_text()); out = Path(args.out); out.mkdir(exist_ok=False)
    started = time.monotonic(); signal.alarm(s['max_seconds'])
    hashes = json.loads((here/'source-hashes.json').read_text())
    for name, digest in hashes.items(): assert sha(here/name) == digest, name
    dependency=json.loads((here/'portable-completion-receipt.json').read_text())
    assert dependency['status']=='completed_and_reviewed'
    assert dependency['protocol_sha256']=='31c4ed50e358fd26861b5374035f14d5e4b5d08968b1ad6415deb05d9503eb85'
    assert dependency['created_at'] < time.time()
    dump(out/'portable-completion-receipt.json',dependency)
    dump(out/'source-hashes.json',hashes); dump(out/'protocol.json',s)
    dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':sha(Path(args.protocol))})
    torch.set_num_threads(4); torch.manual_seed(s['seed'])
    model,tok = load_text_model(s['model_path']); model.requires_grad_(False)
    runtime = GroupRuntime(model,tok); runtime.max_calls = s['max_calls']; dump(out/'runtime.json',metadata(model))
    ids = tok(s['reference_text'], add_special_tokens=False, return_tensors='pt').input_ids.to('cuda:0')
    _,_,residuals,_ = runtime.forward(ids,capture=True)
    references = {l:float(v.median()) for l,v in residuals.items()}; dump(out/'reference-scales.json',references)
    weights = {}
    for category in s['categories'].values():
        for coords in [category['pp16']] + category['random16']:
            for co in coords:
                l,n=co; weights[key(co)] = runtime.layers[l].mlp.down_proj.weight[:,n].detach().float().cpu().numpy().astype(np.float64)
    np.savez_compressed(out/'projection-columns.npz',**weights)
    vectors = {c:make_vectors(v,weights,references) for c,v in s['categories'].items()}
    dump(out/'vectors.json',vectors)
    dump(out/'vector-norms.json',{c:{a:norms(v,weights,references) for a,v in arms.items()} for c,arms in vectors.items()})
    smoke_ids = runtime.ids([{'role':'user','content':'Reply with the word ready.'}])
    z,_ = runtime.trial(smoke_ids,{})
    zeros = {k:0. for k in vectors['academic']['pp16']}
    same,writes=runtime.trial(smoke_ids,zeros)
    assert torch.equal(z,same) and all(w['actual_delta']==0 for w in writes)
    dump(out/'smoke.json',{'passed':True,'reference_calls':1,'identity_calls':2,'multi_layer_identity_coordinates':list(zeros)})
    letter_ids={letter:tok.encode(letter,add_special_tokens=False) for letter in ['A','B']}
    assert all(len(ids)==1 for ids in letter_ids.values()),letter_ids
    dump(out/'protection-tokenization.json',letter_ids)
    journal = out/'records.jsonl'; count=0

    def record(z,baseline,ids,vector,ledger,info,ti,ci):
        nonlocal count
        count+=1
        actual={w['coordinate']:w['actual_delta'] for w in ledger}
        row={**info,'record_number':count,'input_ids':ids[0].tolist(),
             'vector':vector,'writes':ledger,'requested_norms':norms(vector,weights,references),
             'actual_norms':norms(actual,weights,references),**statistics(z,baseline,ti,ci)}
        if 'expected_token' in info: row['correct_first_token'] = row['argmax'] == info['expected_token']
        with journal.open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        return row

    for category,cat in s['categories'].items():
        obs=cat['observable']; ti,ci=obs['target_ids'],obs['control_ids']
        for i,request in enumerate(s['requests']):
            for side,sign in [('base',1),('target',-1)]:
                messages=[{'role':'system','content':cat[side+'_system']},{'role':'user','content':request}]
                ids=runtime.ids(messages); base,_=runtime.trial(ids,{})
                info={'kind':'causal','category':category,'item':i,'side':side,'sign':sign}
                record(base,base,ids,{},[],{**info,'arm':'baseline','dose':0.},ti,ci)
                for arm,dose in causal_conditions():
                    vector=scale(vectors[category][arm],sign*dose)
                    z,ledger=runtime.trial(ids,vector)
                    record(z,base,ids,vector,ledger,{**info,'arm':arm,'dose':dose},ti,ci)
            print('PAIR',category,i,'calls',runtime.calls,flush=True)
        for i,item in enumerate(s['protections']):
            ids=runtime.ids([{'role':'system','content':"Follow the user's exact-answer requirements."},{'role':'user','content':item['prompt']}])
            ti=letter_ids[item['expected']]; ci=letter_ids['B' if item['expected']=='A' else 'A']
            base,_=runtime.trial(ids,{})
            info={'kind':'protection','category':category,'item':i,'side':'neutral','expected_token':ti[0]}
            record(base,base,ids,{},[],{**info,'arm':'baseline','dose':0.,'sign':0},ti,ci)
            for arm,signed_dose in protection_conditions():
                vector=scale(vectors[category][arm],signed_dose)
                z,ledger=runtime.trial(ids,vector)
                record(z,base,ids,vector,ledger,{**info,'arm':arm,'dose':abs(signed_dose),'sign':1 if signed_dose>0 else -1},ti,ci)
        print('CATEGORY',category,'calls',runtime.calls,flush=True)
    assert runtime.calls == expected_calls(s) == s['expected_calls'] == count + 3
    complete={'status':'completed','pid':os.getpid(),'calls':runtime.calls,'records':count,'backwards':0,
              'seconds':time.monotonic()-started,'finished':time.time(),'cuda_peak_allocated':torch.cuda.max_memory_allocated()}
    dump(out/'completion.json',complete)
    dump(out/'final-seal.json',{'sealed_at':time.time(),'files':{str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file()}})
    print('COMPLETE',runtime.calls,flush=True)


if __name__ == '__main__': main()
