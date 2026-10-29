import argparse,hashlib,json,os,re,signal,time
from pathlib import Path
import torch
from model_runtime import load_text_model,metadata
from scoped_runtime import ScopedRuntime
from run_window import dump,screen,normalize,edits


def heraldry(text):
    explicit=bool(re.search(r'\b(?:coats? of arms|blazon\w*|escutcheon\w*)\b',text,re.I))
    terms=sorted(set(re.findall(r'\b(?:azure|argent|gules|sable|purpure|chevron|rampant|passant|guardant|heraldic|heraldry)\b',text.lower())))
    return {'heraldry_lexical':bool(explicit or len(terms)>=2),'heraldry_terms':terms,'explicit_heraldry':explicit}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();raw=Path(a.protocol).read_bytes();s=json.loads(raw);out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic();signal.alarm(s['budgets']['max_seconds']);torch.set_num_threads(4);torch.manual_seed(s['seed']);dump(out/'protocol.json',s);dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':hashlib.sha256(raw).hexdigest()})
    model,tok=load_text_model(s['model_path']);r=ScopedRuntime(model,tok);r.max_calls=s['budgets']['max_forward_calls'];dump(out/'runtime.json',metadata(model));geometry={};hs=[]
    for c in s['early_coordinates']+s['heraldry_coordinates']+s['refine_coordinates']:
        l,n=c['layer'],c['neuron'];v=r.layers[l].mlp.down_proj.weight[:,n].detach().float();vn=float(v.norm())
        def capture(mod,args,out,l=l,n=n,v=v,vn=vn):
            h=(out[0] if isinstance(out,tuple) else out)[0,1:].detach().float();rn=h.norm(dim=-1);co=h@v/(rn*vn);pr=rn.pow(4)/h.pow(4).sum(-1)
            geometry[f'{l}:{n}']={'layer':l,'neuron':n,'column_norm':vn,'residual_norms':rn.cpu().tolist(),'c0_by_position':co.cpu().tolist(),'participation_ratio':pr.cpu().tolist(),'B':float(rn.median())/vn,'median_c0':float(co.median()),'median_PR':float(pr.median())}
        hs.append(r.layers[l].register_forward_hook(capture))
    ids=tok(s['calibration_text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0');r.forward(ids)
    for h in hs:h.remove()
    dump(out/'geometry.json',geometry);cs=[]
    for study in ['early','heraldry']:
        cs.append({'id':study+'_baseline','study':study,'kind':'baseline','layer':None,'neuron':None,'g':0.,'offset':0.})
        for c in s[study+'_coordinates']:
            b=geometry[f"{c['layer']}:{c['neuron']}"]['B'];ds=s['early_doses'] if study=='early' else s['heraldry_doses']+(s['heraldry_target_extras'] if c['kind']=='target' else [])
            for g in ds:cs.append({**c,'id':f"{study}_L{c['layer']}N{c['neuron']}_g{g:g}",'study':study,'g':g,'B':b,'offset':g*b})
    dump(out/'conditions.json',cs);gens=[];facts=[];summary=[]
    for c in cs:
        print('condition',c['id'],'calls',r.calls,flush=True);gs=[];fs=[]
        for p in s[c['study']+'_prompts']:
            ids=r.ids([{'role':'user','content':p['text']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=s['max_new_tokens']);sc=screen(gen['text'],'alien');hr=heraldry(gen['text']);gs.append({'condition':c['id'],'study':c['study'],'prompt_id':p['id'],'prompt':p['text'],**gen,**sc,**hr,'coherent_heraldry':bool(sc['coherent_screen'] and hr['heraldry_lexical'])})
        for f in s['facts']:
            ids=r.ids([{'role':'system','content':s['fact_instruction']},{'role':'user','content':f['question']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=s['fact_max_new_tokens']);fs.append({'condition':c['id'],'study':c['study'],'fact_id':f['id'],'question':f['question'],'expected':f['answers'],'correct':normalize(gen['text']) in [normalize(x) for x in f['answers']],**gen})
        gens.extend(gs);facts.extend(fs);summary.append({**c,'generation_n':len(gs),'coherent':sum(x['coherent_screen'] for x in gs),'coherent_heraldry':sum(x['coherent_heraldry'] for x in gs),'empty':sum(not x['text'].strip() for x in gs),'capped':sum(x['length_capped'] for x in gs),'facts_correct':sum(x['correct'] for x in fs),'facts_n':len(fs)});dump(out/'generations.json',gens);dump(out/'facts.json',facts);dump(out/'summary.json',summary)
    # A separately bounded refinement inside the coarse human dose gap.
    refine=[];refine_rows=[];selection=[]
    for coord in s['refine_coordinates']:
        b=geometry[f"{coord['layer']}:{coord['neuron']}"]['B']
        opts=[]
        for g in s['refine_doses']:
            c={**coord,'id':f"refine_L{coord['layer']}N{coord['neuron']}_g{g:g}",'study':'refine','g':g,'B':b,'offset':g*b};refine.append(c);print('condition',c['id'],'calls',r.calls,flush=True);gs=[];fs=[]
            for p in s['refine_prompts']:
                ids=r.ids([{'role':'user','content':p['text']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=s['max_new_tokens']);gs.append({'condition':c['id'],'study':'refine','phase':'calibration','prompt_id':p['id'],'prompt':p['text'],**gen,**screen(gen['text'],'human')})
            for f in s['facts']:
                ids=r.ids([{'role':'system','content':s['fact_instruction']},{'role':'user','content':f['question']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=s['fact_max_new_tokens']);fs.append({'condition':c['id'],'study':'refine','phase':'calibration','fact_id':f['id'],'question':f['question'],'expected':f['answers'],'correct':normalize(gen['text']) in [normalize(x) for x in f['answers']],**gen})
            gens.extend(gs);facts.extend(fs);opts.append({'condition':c,'hits':sum(x['coherent_target'] for x in gs),'facts':sum(x['correct'] for x in fs)})
        eligible=[x for x in opts if x['hits'] and x['facts']==len(s['facts'])];best=sorted(eligible,key=lambda x:(-x['hits'],x['condition']['g']))[0] if eligible else None;selection.append({'coordinate':coord,'options':opts,'selected':best,'qualified':bool(best)})
    dump(out/'refine-selection.json',selection);dump(out/'refine-commitment.json',{'selection_sha256':hashlib.sha256((out/'refine-selection.json').read_bytes()).hexdigest(),'calls_before_heldout':r.calls})
    for choice in selection:
        if not choice['qualified']:continue
        c=choice['selected']['condition'];print('heldout',c['id'],'calls',r.calls,flush=True)
        for p in s['refine_heldout_prompts']:
            ids=r.ids([{'role':'user','content':p['text']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=96);gens.append({'condition':c['id'],'study':'refine','phase':'heldout','prompt_id':p['id'],'prompt':p['text'],**gen,**screen(gen['text'],'human')})
        for f in s['refine_heldout_facts']:
            ids=r.ids([{'role':'system','content':s['fact_instruction']},{'role':'user','content':f['question']}]);gen=r.generate(ids,edits=edits(c),max_new_tokens=s['fact_max_new_tokens']);facts.append({'condition':c['id'],'study':'refine','phase':'heldout','fact_id':f['id'],'question':f['question'],'expected':f['answers'],'correct':normalize(gen['text']) in [normalize(x) for x in f['answers']],**gen})
    dump(out/'conditions.json',cs+refine);dump(out/'generations.json',gens);dump(out/'facts.json',facts)
    dump(out/'completion.json',{'pid':os.getpid(),'calls':r.calls,'seconds':time.monotonic()-start,'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'finished':time.time()});print('complete',r.calls,flush=True)

if __name__=='__main__':main()
