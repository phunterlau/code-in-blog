"""Frozen, two-stage benign dose search with equal-budget random controls."""
import argparse,hashlib,json,os,re,signal,statistics,time
from pathlib import Path
import torch
from model_runtime import load_text_model,metadata
from scoped_runtime import ScopedRuntime


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def screen(text,target):
    text=text.lower().replace('\u2019',"'")
    words=re.findall(r"\b[\w']+\b",text)
    bigrams=list(zip(words,words[1:]));d2=len(set(bigrams))/len(bigrams) if bigrams else 0.
    coherent=len(words)>=6 and d2>=.6 and not re.search(r'\b(\w+)(?:\s+\1){5,}\b',text)
    vocab={'human':r'human|person|student|scientist|man|woman|engineer|teacher|doctor',
           'building':r'building|house|skyscraper|tower|castle|cottage|cathedral|apartment building',
           'corporation':r'corporation|company|business|enterprise',
           'alien':r'alien|extraterrestrial'}
    pattern=r"\b(?:i am|i'm|we are|we're)\s+(?:a|an|the)\s+(?:[a-z]+[ -]){0,4}(?:"+vocab[target]+r')\b'
    hits=[m.group(0) for m in re.finditer(pattern,text)]
    hits=[m for m in hits if not re.search(r'\b(?:ai|artificial|model|digital|virtual|simulated|not|never)\b',m)]
    return {'word_count':len(words),'distinct2':d2,'coherent_screen':bool(coherent),'literal_target':bool(hits),'coherent_target':bool(coherent and hits),'matched_clauses':hits}


def normalize(s):return re.sub(r'[^a-z0-9]','',s.lower())


def edits(c):
    if c['layer'] is None:return {}
    return {c['layer']:[{'neurons':[c['neuron']],'operation':'add','values':[c['offset']],'scope':'all'}]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    raw=Path(a.protocol).read_bytes();s=json.loads(raw);out=Path(a.out);out.mkdir(exist_ok=False)
    started=time.monotonic();signal.alarm(s['budgets']['max_seconds']);torch.set_num_threads(4);torch.manual_seed(s['seed'])
    dump(out/'protocol.json',s);dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':hashlib.sha256(raw).hexdigest()})
    print('load',flush=True);model,tok=load_text_model(s['model_path']);r=ScopedRuntime(model,tok);r.max_calls=s['budgets']['max_forward_calls'];dump(out/'runtime.json',metadata(model))
    cohort=[]
    for t in s['targets']:
        for kind,n in [('target',t['neuron'])]+[(f'random{j}',n) for j,n in enumerate(t['random_neurons'])]:
            cohort.append({'id':t['id']+'_'+kind,'target':t['id'],'kind':kind,'layer':t['layer'],'neuron':n,'sign':t['sign']})
    cohort.append({'id':'alien_anchor','target':'alien','kind':'anchor','layer':22,'neuron':2742,'sign':1})
    # Exact baseline geometry, at the residual leaving the edited layer.
    geometry={};handles=[]
    for c in cohort:
        l,n=c['layer'],c['neuron'];v=r.layers[l].mlp.down_proj.weight[:,n].detach().float();vn=float(v.norm())
        def capture(mod,args,out,c=c,v=v,vn=vn):
            h=(out[0] if isinstance(out,tuple) else out)[0,1:].detach().float();rn=h.norm(dim=-1);cos=h@v/(rn*vn)
            geometry[c['id']]={'layer':c['layer'],'neuron':c['neuron'],'column_norm':vn,'residual_norms':rn.cpu().tolist(),'c0_by_position':cos.cpu().tolist(),'B':float(rn.median())/vn,'median_c0':float(cos.median())}
        handles.append(r.layers[l].register_forward_hook(capture))
    cal=tok(s['calibration_text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0');r.forward(cal)
    for h in handles:h.remove()
    dump(out/'geometry.json',geometry)
    conditions=[{'id':'baseline','layer':None,'neuron':None,'offset':0.,'target':'all','kind':'baseline','g':0.}]
    for c in cohort:
        doses=[.25,.5,1.] if c['kind']=='anchor' else s['relative_doses']
        for dose in doses:
            b=geometry[c['id']]['B'];g=c['sign']*dose;c0=geometry[c['id']]['median_c0']
            conditions.append({**c,'id':c['id']+f'_g{g:g}','cohort_id':c['id'],'g':g,'B':b,'offset':g*b,'c0':c0,'predicted_c':(c0+g)/(1+2*c0*g+g*g)**.5})
    dump(out/'conditions.json',conditions);gens=[];facts=[]
    def collect(c,phase):
        print('condition',phase,c['id'],'calls',r.calls,flush=True)
        for p in s['calibration_prompts' if phase=='calibration' else 'heldout_prompts']:
            ids=r.ids([{'role':'user','content':p['text']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=s['max_tokens'][phase])
            rec={'phase':phase,'condition':c['id'],'prompt_id':p['id'],'prompt':p['text'],**gen}
            rec['screens']={t:screen(gen['text'],t) for t in ['human','building','corporation','alien']};gens.append(rec)
        for f in s['calibration_facts' if phase=='calibration' else 'heldout_facts']:
            ids=r.ids([{'role':'system','content':s['fact_instruction']},{'role':'user','content':f['question']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=s['max_tokens']['facts'])
            facts.append({'phase':phase,'condition':c['id'],'fact_id':f['id'],'question':f['question'],'expected':f['answers'],'correct':normalize(gen['text']) in [normalize(v) for v in f['answers']],**gen})
        dump(out/'generations.json',gens);dump(out/'facts.json',facts)
    for c in conditions:collect(c,'calibration')
    selected=[];selection=[]
    # Read calibration only. The target and both controls have the same six-point budget.
    for c in cohort:
        if c['kind']=='anchor':continue
        options=[]
        for d in conditions:
            if d.get('cohort_id')!=c['id']:continue
            gs=[x for x in gens if x['phase']=='calibration' and x['condition']==d['id']];fs=[x for x in facts if x['phase']=='calibration' and x['condition']==d['id']]
            options.append({'condition':d['id'],'g':d['g'],'hits':sum(x['screens'][c['target']]['coherent_target'] for x in gs),'facts_correct':sum(x['correct'] for x in fs),'facts_n':len(fs)})
        eligible=[x for x in options if x['facts_correct']==x['facts_n'] and x['hits']>0]
        best=sorted(eligible,key=lambda x:(-x['hits'],abs(x['g'])))[0] if eligible else min(options,key=lambda x:abs(x['g']))
        selection.append({'cohort':c,'options':options,'selected_condition':best['condition'],'qualified':bool(eligible),'selection_basis':'calibration only'})
        selected.append(next(x for x in conditions if x['id']==best['condition']))
    dump(out/'selection.json',selection)
    # The selection is persisted and hashed before the first held-out generation.
    dump(out/'selection_commitment.json',{'sha256':hashlib.sha256((out/'selection.json').read_bytes()).hexdigest(),'calls_before_heldout':r.calls,'time':time.time()})
    for c in conditions[:1]+[x for x in conditions if x['kind']=='anchor']+selected:collect(c,'heldout')
    summary=[]
    for phase in ['calibration','heldout']:
        for c in conditions:
            gs=[x for x in gens if x['phase']==phase and x['condition']==c['id']];fs=[x for x in facts if x['phase']==phase and x['condition']==c['id']]
            if not gs:continue
            summary.append({'phase':phase,'condition':c['id'],'target':c['target'],'kind':c['kind'],'g':c['g'],'offset':c['offset'],'n':len(gs),'coherent':sum(x['screens']['alien']['coherent_screen'] for x in gs),'empty':sum(not x['text'].strip() for x in gs),'capped':sum(x['length_capped'] for x in gs),'target_hits':{t:sum(x['screens'][t]['coherent_target'] for x in gs) for t in ['human','building','corporation','alien']},'facts_correct':sum(x['correct'] for x in fs),'facts_n':len(fs)})
    dump(out/'summary.json',summary);dump(out/'completion.json',{'pid':os.getpid(),'calls':r.calls,'seconds':time.monotonic()-started,'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved(),'finished':time.time()})
    print('complete',r.calls,flush=True)

if __name__=='__main__':main()
