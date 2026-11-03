"""Signed-mean PP, coordinate-excluded selection, causal sweeps and held-out control."""
import argparse,hashlib,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from model_runtime import Runtime,load_text_model,metadata
from metrics import active,measure

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def co_name(co):return f'L{co[0]}N{co[1]}'
def ids(rt,item):return rt.ids([{'role':'user','content':item['prompt']}])
def gap(z,t,c):return float(z[t].mean()-z[c].mean())
def vocab_ids(tok,words):
    result=set()
    for w in words:
        for text in [w,' '+w]:
            a=tok.encode(text,add_special_tokens=False)
            if len(a)==1:result.add(a[0])
    return sorted(result)

class StudyRuntime(Runtime):
    def __init__(self,*args):self.step=None;self.ledger=[];super().__init__(*args)
    def pre_hook(self,l):
        parent=super().pre_hook(l)
        def hook(m,args):
            changed=parent(m,args)
            if changed is not None and l in self.edits and self.step is not None:
                n=self.edits[l][0]['neurons'][0]
                before=args[0][0,-1,n].float();after=changed[0][0,-1,n].float()
                self.ledger.append({'step':self.step,'layer':l,'neuron':n,'positions':args[0].shape[1],
                    'requested_offset':self.edits[l][0]['values'][0],'before':float(before),'after':float(after),'actual_last_offset':float(after-before)})
            return changed
        return hook
    def edits_for(self,co,g):
        l,n=co;B=self.scales[l]/float(self.layers[l].mlp.down_proj.weight[:,n].detach().float().norm())
        return {l:[{'neurons':[n],'operation':'add','values':[g*B],'all_tokens':True}]}
    @torch.inference_mode()
    def gen(self,item,recipe,cap):
        prefix=ids(self,item);current=prefix;cache=None;tokens=[];self.ledger=[];before=self.calls
        stops={self.tok.eos_token_id,self.tok.convert_tokens_to_ids('<|im_end|>'),self.tok.convert_tokens_to_ids('<|endoftext|>')}
        stops={x for x in stops if isinstance(x,int) and x>=0}
        try:
            for step in range(cap):
                self.step=step
                self.edits=self.edits_for(recipe['coordinate'],recipe['dose']) if recipe and active(recipe['schedule'],step) else {}
                out=self.model(input_ids=current,use_cache=True,past_key_values=cache,logits_to_keep=1)
                assert out.logits.is_cuda and torch.isfinite(out.logits).all()
                token=int(out.logits[0,-1].argmax());tokens.append(token)
                if token in stops:break
                cache=out.past_key_values;current=torch.tensor([[token]],device='cuda:0')
            capped=len(tokens)==cap and tokens[-1] not in stops
            text=self.tok.decode(tokens,skip_special_tokens=True)
            return {'id':item['id'],'text':text,'token_ids':tokens,'capped':capped,'metrics':measure(item,text,capped),
                    'calls':self.calls-before,'writes':list(self.ledger)}
        finally:self.edits={};self.step=None;self.ledger=[]

def summary(targets,protections,base_targets,base_protections):
    retained_t=sum((not b['metrics']['correct']) or x['metrics']['correct'] for x,b in zip(targets,base_targets))==len(targets)
    retained_p=sum((not b['metrics']['correct']) or x['metrics']['correct'] for x,b in zip(protections,base_protections))==len(protections)
    joint=sum(x['metrics']['joint'] for x in targets)
    return {'joint':joint,'target_n':len(targets),'target_correct':sum(x['metrics']['correct'] for x in targets),
            'protection_correct':sum(x['metrics']['correct'] for x in protections),'protection_n':len(protections),
            'baseline_correct_targets_retained':retained_t,'baseline_correct_protections_retained':retained_p,
            'capped_targets':sum(x['capped'] for x in targets),'eligible':retained_t and retained_p and joint>=1 and not any(x['capped'] for x in targets)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();here=Path(__file__).resolve().parent
    s=json.loads((here/'protocol.json').read_text());out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    started=time.monotonic();signal.alarm(s['max_seconds']);torch.set_num_threads(4);torch.manual_seed(s['seed'])
    hashes=json.loads((here/'source-hashes.json').read_text())
    for name,digest in hashes.items():assert sha(here/name)==digest,name
    dump(out/'source-hashes.json',hashes);dump(out/'protocol.json',s)
    dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':sha(here/'protocol.json')})
    m,t=load_text_model(s['model_path']);r=StudyRuntime(m,t);r.max_calls=s['max_forward_calls']
    dump(out/'runtime.json',metadata(m))
    _,_,resids,_=r.forward(t(s['calibration_text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0'),capture=True)
    r.scales={l:float(x.median()) for l,x in resids.items()};assert all(v>0 for v in r.scales.values());dump(out/'reference-scales.json',r.scales)
    discovery={};numeric={};readouts={}
    for name,task in s['tasks'].items():
        target,control=vocab_ids(t,task['target_tokens']),vocab_ids(t,task['control_tokens'])
        assert target and control and not set(target)&set(control)
        d=m.lm_head.weight[target].detach().float().mean(0)-m.lm_head.weight[control].detach().float().mean(0)
        coupling=torch.stack([(layer.mlp.down_proj.weight.detach().float().T@d.to('cuda:0')).cpu() for layer in r.layers])
        bases=[];targets=[];parents=[]
        for pair in task['discovery']:
            parent={'id':pair['id']}
            for side,store in [('base',bases),('target',targets)]:
                zz,acts,*_=r.forward(r.ids(pair[side]),capture=True)
                store.append(torch.stack([acts[l] for l in range(len(r.layers))]))
                parent[side]={'gap':gap(zz,target,control),'argmax':int(zz.argmax()),'argmax_text':t.decode([int(zz.argmax())])}
            parents.append(parent)
        b=torch.stack(bases);a_=torch.stack(targets);delta=a_-b;terms=delta*coupling;score=terms.mean(0);rms=terms.square().mean(0).sqrt()
        order=torch.argsort(score.flatten(),descending=True,stable=True);top=[]
        for idx in order[:50]:
            l,n=divmod(int(idx),score.shape[1]);top.append({'coordinate':[l,n],'score':float(score[l,n]),'mean_activation_delta':float(delta[:,l,n].mean()),'coupling':float(coupling[l,n])})
        assert all(x['score']>0 for x in top[:s['pp_candidates']])
        discovery[name]={'top50':top,'parents':parents,'target_ids':target,'control_ids':control}
        numeric[name]=(score,rms,coupling,delta);readouts[name]=(target,control)
        np.savez_compressed(out/(name+'-discovery.npz'),base=b.numpy(),target=a_.numpy(),coupling=coupling.numpy(),signed=score.numpy(),rms=rms.numpy())
        dump(out/'discovery.json',discovery);print('DISCOVERY',name,top[:4],'calls',r.calls,flush=True)
    dump(out/'discovery-seal.json',{'sha256':sha(out/'discovery.json'),'calls':r.calls,'time':time.time(),'method':'signed-mean descending; references not yet decoded'})
    # Reference coordinates enter only after all task rankings are fixed.
    references=json.loads((here/'references.json').read_text());pub_ranks={}
    for name,(score,rms,coupling,delta) in numeric.items():
        l,n=references[name]
        pub_ranks[name]={'coordinate':[l,n],'signed_rank':int((score>score[l,n]).sum())+1,'rms_rank':int((rms>rms[l,n]).sum())+1,
            'signed_score':float(score[l,n]),'coupling':float(coupling[l,n]),'mean_activation_delta':float(delta[:,l,n].mean()),
            'pair_contributions':(delta[:,l,n]*coupling[l,n]).tolist()}
    dump(out/'published-ranks.json',pub_ranks)
    rng=random.Random(s['seed']);random_coords=[list(divmod(x,9216)) for x in rng.sample(range(32*9216),s['random_sweep_n'])];dump(out/'random-coordinates.json',random_coords)
    protection_z=[r.forward(ids(r,x))[0] for x in s['protections_cal']]
    def protected(zs):
        keep=[]
        for item,base,z in zip(s['protections_cal'],protection_z,zs):
            expected=str(item['expected']);first={t.encode(v,add_special_tokens=False)[0] for v in [expected,' '+expected]}
            b=int(base.argmax()) in first;keep.append(not b or int(z.argmax())==int(base.argmax()))
        return all(keep)
    def screen(name,co,doses,origin,basez,items):
        ti,ci=readouts[name];rows=[]
        for dose in doses:
            edits=r.edits_for(co,dose)
            zz=[r.forward(ids(r,x),edits=edits)[0] for x in items]
            pz=[r.forward(ids(r,x),edits=edits)[0] for x in s['protections_cal']]
            gain=sum(gap(z,ti,ci)-gap(b,ti,ci) for z,b in zip(zz,basez))/len(zz)
            rows.append({'coordinate':co,'origin':origin,'dose':dose,'gap_gain':gain,'protected_argmaxes_retained':protected(pz),
                         'eligible':protected(pz) and gain>0,'target_argmax':[t.decode([int(z.argmax())]) for z in zz]})
        return rows
    def screen_priority(row):return row['eligible'],row['gap_gain'],-abs(row['dose'])
    screens={};candidates={}
    for name,task in s['tasks'].items():
        items=task['calibration'][:2];basez=[r.forward(ids(r,x))[0] for x in items]
        pp=[x['coordinate'] for x in discovery[name]['top50'][:s['pp_candidates']]]
        rows=[]
        for co in pp:rows+=screen(name,co,s['screen_doses'],'pp',basez,items)
        ppwin=max(rows,key=screen_priority)
        public_rows=screen(name,references[name],s['screen_doses'],'public',basez,items);rows+=public_rows
        rw=[]
        for co in random_coords:rw+=screen(name,co,s['random_sweep_doses'],'random_sweep',basez,items)
        rows+=rw
        candidates[name]={'pp':max([x for x in rows if x['origin']=='pp'],key=screen_priority),
                          'public':max(public_rows,key=screen_priority),'sweep':max(rw,key=screen_priority)}
        screens[name]=rows;dump(out/'screens.json',screens);dump(out/'candidates.json',candidates)
        print('SCREEN',name,candidates[name],'calls',r.calls,flush=True)
    dump(out/'candidate-seal.json',{'sha256':sha(out/'candidates.json'),'calls':r.calls,'time':time.time()})
    def ev(items,recipe,cap):return [r.gen(x,recipe,cap) for x in items]
    calbase={};trials={};choices={}
    schedule_rank={'prefill_only':2,'pulse4':1,'continuous':0}
    def gen_priority(row):
        rec=row['recipe'];met=row['metrics']
        return met['joint'],met['target_correct'],met['protection_correct'],-abs(rec['dose']) if rec else 0,schedule_rank[rec['schedule']] if rec else 3
    for name,task in s['tasks'].items():
        cap=s['zero_token_cap'] if name=='zero' else s['target_token_cap']
        bg=ev(task['calibration'],None,cap);bp=ev(s['protections_cal'],None,s['protection_token_cap']);calbase[name]={'target':bg,'protection':bp}
        base_row={'recipe':None,'metrics':summary(bg,bp,bg,bp),'target':bg,'protection':bp}
        trials[name]={};choices[name]={};dump(out/'calibration-baseline.json',calbase)
        for origin,win in candidates[name].items():
            rows=[]
            for dose in [win['dose']/2,win['dose']]:
              for schedule in s['schedules']:
                recipe={'coordinate':win['coordinate'],'dose':dose,'schedule':schedule}
                g=ev(task['calibration'],recipe,cap);p=ev(s['protections_cal'],recipe,s['protection_token_cap'])
                rows.append({'recipe':recipe,'metrics':summary(g,p,bg,bp),'target':g,'protection':p})
            eligible=[x for x in [base_row]+rows if x['metrics']['eligible']]
            chosen=max(eligible,key=gen_priority) if eligible else base_row
            challenger=max(rows,key=lambda x:(x['metrics']['eligible'],*gen_priority(x)))
            choices[name][origin]={'selected':chosen['recipe'],'selected_metrics':chosen['metrics'],
                                    'challenger':challenger['recipe'],'challenger_metrics':challenger['metrics']}
            trials[name][origin]=rows;dump(out/'calibration.json',trials);dump(out/'choices.json',choices)
            print('CALIBRATED',name,origin,choices[name][origin],'calls',r.calls,flush=True)
    dump(out/'selection-seal.json',{'sha256':sha(out/'choices.json'),'calls':r.calls,'time':time.time()})
    held={}
    for name,task in s['tasks'].items():
        cap=s['zero_token_cap'] if name=='zero' else s['target_token_cap']
        bg=ev(task['heldout'],None,cap);bp=ev(s['protections_held'],None,s['protection_token_cap'])
        held[name]={'baseline':{'target':bg,'protection':bp},'arms':{}}
        for origin,choice in choices[name].items():
            arm={}
            for label in ['selected','challenger']:
                recipe=choice[label]
                if recipe is None:g,p=bg,bp
                elif label=='challenger' and recipe==choice['selected']:
                    arm[label]={'alias':'selected'};continue
                else:g,p=ev(task['heldout'],recipe,cap),ev(s['protections_held'],recipe,s['protection_token_cap'])
                arm[label]={'recipe':recipe,'target':g,'protection':p,'metrics':summary(g,p,bg,bp)}
            held[name]['arms'][origin]=arm;dump(out/'heldout.json',held)
            print('HELDOUT',name,origin,arm['selected']['metrics'],'calls',r.calls,flush=True)
    dump(out/'completion.json',{'status':'completed','pid':os.getpid(),'calls':r.calls,'backwards':0,'seconds':time.monotonic()-started,
          'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'finished':time.time(),'heldout_sha256':sha(out/'heldout.json')})
    print('COMPLETE',r.calls,flush=True)
if __name__=='__main__':main()
