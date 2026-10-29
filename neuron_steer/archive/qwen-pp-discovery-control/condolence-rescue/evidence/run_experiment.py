import argparse,hashlib,json,os,random,re,signal,time
from pathlib import Path
import torch
from model_runtime import load_text_model,metadata
from science_runtime import ScienceRuntime,observable,gap,ffn_diagnostic

def dump(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def coord(j):return list(divmod(int(j),9216))
def key(c):return f'L{c[0]}N{c[1]}'
def clean(t):return t.strip().strip('"`').strip()
def assess(task,item,text):
    t=clean(text);words=re.findall(r'[A-Za-z]+',t.lower());rep=bool(words) and max(words.count(w) for w in set(words))>=6
    if task=='protection':return {'content':t==item['expected'],'trait':True,'joint':t==item['expected']}
    if task=='caps':
        letters=[c for c in t if c.isalpha()];fraction=sum(c.isupper() for c in letters)/max(1,len(letters))
        content=t.lower()==item['expected'].lower();trait=fraction>=.85 and not rep
        return {'content':content,'trait':trait,'joint':content and trait,'uppercase_fraction':fraction,'repetitive':rep}
    content=item['required'].lower() in words and not rep
    if task=='alien':trait=bool(re.search(r'\b(alien|extraterrestrial|martian|other planet|another planet|distant planet)\b',t.lower()))
    else:trait=bool(re.search(r'\b(sorry|condolences|sympath\w*|sorrow|heart goes out)\b',t.lower()))
    return {'content':content,'trait':trait,'joint':content and trait,'repetitive':rep}

class Runtime(ScienceRuntime):
    def __init__(self,m,t):self.control=None;self.units={};self.scales={};self.ledger=[];super().__init__(m,t)
    def pre_hook(self,l):
        parent=super().pre_hook(l)
        def hook(mod,args):
            old=parent(mod,args);a=old[0] if old is not None else args[0];c=self.control
            if c is None or c['layer']!=l or c['method']=='residual':return old
            n=c['neuron'];y=a.clone();v=a[...,n]
            if c['method']=='scale':y[...,n]=v*c['value']
            elif c['method']=='clamp':y[...,n]=c['value']
            else:raise ValueError(c)
            raw=float((y[0,-1,n]-v[0,-1]).float());wn=float(self.layers[l].mlp.down_proj.weight[:,n].float().norm())
            self.ledger.append({'layer':l,'positions':a.shape[1],'activation_delta_last':raw,'requested_write_norm':abs(raw)*wn})
            return (y,)
        return hook
    def post_hook(self,l):
        parent=super().post_hook(l)
        def hook(mod,args,out):
            parent(mod,args,out);c=self.control
            if c is None or c['layer']!=l or c['method']!='residual':return None
            h=out[0] if isinstance(out,tuple) else out;n=c['neuron'];u=self.units[l,n].to(h.device);d=u*(c['value']*self.scales[l]);y=h+d.to(h.dtype)
            self.ledger.append({'layer':l,'positions':h.shape[1],'requested_write_norm':float(d.norm()),'realized_write_norm_last':float((y[0,-1].float()-h[0,-1].float()).norm())})
            return (y,*out[1:]) if isinstance(out,tuple) else y
        return hook
    def prepare(self,c):
        l,n=c
        if (l,n) not in self.units:
            w=self.layers[l].mlp.down_proj.weight[:,n].detach().float();self.units[l,n]=(w/w.norm()).cpu()
    def controlled(self,ids,control):
        self.control=control;self.ledger=[]
        try:
            z,*_=self.forward(ids);return z,list(self.ledger)
        finally:self.control=None
    @torch.inference_mode()
    def generate_control(self,item,control,cap,task):
        ids=self.ids([{'role':'user','content':item['prompt']}]);self.control=control;self.ledger=[];before=self.calls
        try:
            eos=list(set(self.tok.convert_tokens_to_ids(['<|im_end|>','<|endoftext|>'])))
            out=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=cap,use_cache=True,pad_token_id=self.tok.eos_token_id,eos_token_id=eos)
            assert out.is_cuda;tokens=out[0,ids.shape[1]:].tolist();assert self.calls-before==len(tokens)
            text=self.tok.decode(tokens,skip_special_tokens=True)
            return {'id':item['id'],'text':text,'tokens':tokens,'capped':len(tokens)==cap,'calls':self.calls-before,'assessment':assess(task,item,text),'writes':list(self.ledger)}
        finally:self.control=None

def first_eval(r,items,control,ti,ci,baseline=None):
    out=[];zs=[]
    for i,item in enumerate(items):
        ids=r.ids([{'role':'user','content':item['prompt']}]);z,ledger=r.controlled(ids,control);zs.append(z)
        row={'id':item['id'],'gap':gap(z,ti,ci),'argmax':int(z.argmax()),'argmax_text':r.tok.decode([int(z.argmax())]),'writes':ledger}
        if baseline is not None:
            b=baseline[i];blp=b.log_softmax(-1);lp=z.log_softmax(-1);row['kl']=float((blp.exp()*(blp-lp)).sum());row['same_argmax']=int(z.argmax())==int(b.argmax())
        out.append(row)
    return out,zs
def summarize_gens(rows,protect,pbase):
    retained=all(not b['assessment']['content'] or x['assessment']['content'] for x,b in zip(protect,pbase))
    joint=sum(x['assessment']['joint'] for x in rows);trait=sum(x['assessment']['trait'] for x in rows)
    return {'joint':joint,'trait':trait,'n':len(rows),'protection_retained':retained,'eligible':retained and joint==len(rows)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();s=json.loads(Path(a.protocol).read_text());out=Path(a.out);out.mkdir(exist_ok=False)
    began=time.monotonic();signal.alarm(s['max_seconds']);torch.set_num_threads(4);torch.manual_seed(s['seed'])
    hashes=json.loads((Path(__file__).parent/'source-hashes.json').read_text())
    for n,h in hashes.items():assert sha(Path(__file__).parent/n)==h,n
    dump(out/'source-hashes.json',hashes);dump(out/'protocol.json',s);dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':sha(Path(a.protocol))})
    m,t=load_text_model(s['model_path']);r=Runtime(m,t);r.max_calls=s['max_forward_calls'];dump(out/'runtime.json',metadata(m))
    _,_,rr,_=r.forward(t(s['calibration_text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0'),capture=True);r.scales={l:float(v.median()) for l,v in rr.items()};dump(out/'reference-scales.json',r.scales)
    all_summary={}
    for task_index,(name,task) in enumerate(s['tasks'].items()):
        dest=out/name;dest.mkdir();ti,ci,d,c=observable(m,t,task['target_tokens'],task['control_tokens']);dump(dest/'observable.json',{'target_ids':ti,'control_ids':ci,'target_text':[t.decode([i]) for i in ti],'control_text':[t.decode([i]) for i in ci]})
        captures=[];parents=[]
        for p in task['pairs']:
            row={'id':p['id']};aa=[]
            for cond in ['base','target']:
                ids=r.ids(p[cond],p['suffix']);(z,acts,_,_),details=r.observed(ids);aa.append(torch.stack([acts[l] for l in range(32)]));row[cond]={'gap':gap(z,ti,ci),'argmax':int(z.argmax()),'argmax_text':t.decode([int(z.argmax())]),'ids':ids[0].tolist(),'ffn_diagnostic':ffn_diagnostic(details,d)}
            captures.append(aa);parents.append(row)
        b=torch.stack([x[0] for x in captures]);target=torch.stack([x[1] for x in captures]);I=(target[:8]-b[:8])*c;mu=I.mean(0);rms=I.square().mean(0).sqrt();mb=b[:8].mean(0);mt=target[:8].mean(0)
        dump(dest/'parents.json',parents);torch.save({'base':b,'target':target,'coupling':c,'signed_mean':mu,'rms':rms},dest/'captures.pt')
        orders={k:v.flatten().argsort(descending=True,stable=True) for k,v in [('signed',mu),('rms',rms)]};rank={};groups={}
        public=task['known'];public_j=public[0]*9216+public[1]
        for method,order in orders.items():
            if method=='signed':order=order[mu.flatten()[order]>0]
            groups[method]=[coord(j) for j in order[:s['candidate_k']]];assert len(groups[method])==s['candidate_k']
            raw_order=orders[method];inv=torch.empty_like(raw_order);inv[raw_order]=torch.arange(len(raw_order));rank[method]={'public_rank':int(inv[public_j])+1,'public_score':float((mu if method=='signed' else rms)[public[0],public[1]]),'top100':[{'coordinate':coord(j),'signed_mean':float(mu.flatten()[j]),'rms':float(rms.flatten()[j])} for j in raw_order[:100]]}
        forbidden={tuple(x) for g in groups.values() for x in g}|{tuple(public)}
        for j,method in enumerate(['signed','rms']):
            rng=random.Random(s['seed']+100*task_index+j);g=[]
            for l,n in groups[method]:
                while True:
                    nn=rng.randrange(9216)
                    if (l,nn) not in forbidden:break
                forbidden.add((l,nn));g.append([l,nn])
            groups['random_'+method]=g
        pool=sorted({tuple(x) for g in groups.values() for x in g}|{tuple(public)})
        dump(dest/'rankings.json',rank);dump(dest/'groups.json',groups)
        screen=[]
        for co in pool:
            l,n=co;effects=[]
            for i in range(8,12):
                ids=torch.tensor([parents[i]['base']['ids']],device='cuda:0');z,*_=r.forward(ids,edits={l:[{'operation':'set','neurons':[n],'values':[float(target[i,l,n])],'scope':'last'}]});effects.append(gap(z,ti,ci)-parents[i]['base']['gap'])
            screen.append({'coordinate':list(co),'effects':effects,'mean':sum(effects)/len(effects)})
        dump(dest/'screen.json',screen);byco={tuple(x['coordinate']):x for x in screen}
        winners={name:max(g,key=lambda co:(byco[tuple(co)]['mean'],-co[0],-co[1])) for name,g in groups.items()}
        new=[co for co in groups['signed'] if co not in s['known_coordinates']]
        if new:winners['signed_unlisted']=max(new,key=lambda co:(byco[tuple(co)]['mean'],-co[0],-co[1]))
        winners['public']=public;dump(dest/'winners.json',winners);dump(dest/'candidate-commitment.json',{'winners_sha256':sha(dest/'winners.json'),'screen_sha256':sha(dest/'screen.json'),'calls':r.calls})
        print(name,'ranks',{k:v['public_rank'] for k,v in rank.items()},'winners',winners,'calls',r.calls,flush=True)
        unique=sorted({tuple(co) for co in winners.values()});rest=[]
        for co in unique:
            l,n=co
            for i in range(12,20):
                for mode in ['restore','reverse','ablate']:
                    source='base' if mode=='restore' else 'target';value=float(target[i,l,n]) if mode=='restore' else float(b[i,l,n]) if mode=='reverse' else 0.
                    ids=torch.tensor([parents[i][source]['ids']],device='cuda:0');z,*_=r.forward(ids,edits={l:[{'operation':'set','neurons':[n],'values':[value],'scope':'last'}]})
                    rest.append({'coordinate':list(co),'pair':parents[i]['id'],'mode':mode,'delta':gap(z,ti,ci)-parents[i][source]['gap']})
        dump(dest/'restoration-heldout.json',rest)
        basegen=[r.generate_control(x,None,s['gen_target_cap'],name) for x in task['calibration']];pbase=[r.generate_control(x,None,s['gen_protection_cap'],'protection') for x in s['protection_cal']]
        tb,tz=first_eval(r,task['calibration'],None,ti,ci);pb,pz=first_eval(r,s['protection_cal'],None,ti,ci)
        dump(dest/'calibration-baseline.json',{'target':basegen,'protection':pbase,'first_target':tb,'first_protection':pb})
        first=[];cal=[];choices={}
        for co in unique:
            l,n=co;r.prepare(co);recipes=[]
            for v in s['residual_doses']:recipes.append({'layer':l,'neuron':n,'method':'residual','value':v,'dose':v})
            for v in s['scale_factors']:recipes.append({'layer':l,'neuron':n,'method':'scale','value':v,'dose':v})
            for v in s['clamp_alphas']:recipes.append({'layer':l,'neuron':n,'method':'clamp','value':float(mb[l,n]+v*(mt[l,n]-mb[l,n])),'dose':v})
            rows=[]
            for recipe in recipes:
                tr,_=first_eval(r,task['calibration'],recipe,ti,ci);pr,_=first_eval(r,s['protection_cal'],recipe,ti,ci,baseline=pz)
                gain=sum(x['gap']-y['gap'] for x,y in zip(tr,tb))/len(tb);kl=sum(x['kl'] for x in pr)/len(pr)
                retained=all(not bg['assessment']['content'] or x['same_argmax'] for x,bg in zip(pr,pbase))
                norm=sum(x['writes'][-1]['requested_write_norm'] for x in tr)/len(tr)
                row={'coordinate':list(co),'recipe':recipe,'target':tr,'protection':pr,'gain':gain,'mean_kl':kl,'protected':retained,'mean_write_norm':norm,'eligible':retained and kl<=.5 and gain>.02};rows.append(row);first.append(row)
            dump(dest/'first-token-dose-calibration.json',first)
            tested=[]
            for method in ['residual','scale','clamp']:
                eligible=[x for x in rows if x['recipe']['method']==method and x['eligible']]
                if not eligible:continue
                best=max(eligible,key=lambda x:(x['gain'],-x['mean_write_norm'],-abs(x['recipe']['dose'])));recipe=best['recipe']
                g=[r.generate_control(x,recipe,s['gen_target_cap'],name) for x in task['calibration']];p=[r.generate_control(x,recipe,s['gen_protection_cap'],'protection') for x in s['protection_cal']]
                result={'coordinate':list(co),'recipe':recipe,'target':g,'protection':p,'first_gain':best['gain'],'mean_write_norm':best['mean_write_norm'],'metrics':summarize_gens(g,p,pbase)};tested.append(result);cal.append(result);dump(dest/'generation-calibration.json',cal)
            eligible=[x for x in tested if x['metrics']['eligible']]
            priority=lambda x:(x['metrics']['joint'],x['metrics']['protection_retained'],x['metrics']['trait'],-x['mean_write_norm'])
            selected=max(eligible,key=priority) if eligible else None
            challenger=max(tested,key=priority) if tested else None
            choices[key(co)]={'coordinate':list(co),'selected':selected['recipe'] if selected else None,'challenger':challenger['recipe'] if challenger else None,'qualified':bool(eligible),'calibration_metrics':(selected or challenger)['metrics'] if selected or challenger else None}
            print(name,key(co),'choice',choices[key(co)],'calls',r.calls,flush=True)
        dump(dest/'selections.json',choices);dump(dest/'dose-commitment.json',{'selections_sha256':sha(dest/'selections.json'),'calls':r.calls})
        held={'baseline':{'recipe':None,'target':[r.generate_control(x,None,s['gen_target_cap'],name) for x in task['heldout']],'protection':[r.generate_control(x,None,s['gen_protection_cap'],'protection') for x in s['protection_held']]}}
        for k,ch in choices.items():
            recipe=ch['selected'] or ch['challenger']
            if recipe is None:held[k]={'alias':'baseline','qualified':False};continue
            held[k]={'recipe':recipe,'qualified':ch['qualified'],'evaluation_type':'confirmation' if ch['qualified'] else 'failed-calibration challenger diagnosis','target':[r.generate_control(x,recipe,s['gen_target_cap'],name) for x in task['heldout']],'protection':[r.generate_control(x,recipe,s['gen_protection_cap'],'protection') for x in s['protection_held']]};dump(dest/'heldout.json',held)
        recipe={'layer':public[0],'neuron':public[1],'method':'residual','value':float(task['public_sign']),'dose':float(task['public_sign'])};r.prepare(public)
        held['public_source_strength']={'recipe':recipe,'qualified':False,'evaluation_type':'prespecified reference normalized strength, not exact source runtime replication','target':[r.generate_control(x,recipe,s['gen_target_cap'],name) for x in task['heldout']],'protection':[r.generate_control(x,recipe,s['gen_protection_cap'],'protection') for x in s['protection_held']]};dump(dest/'heldout.json',held)
        all_summary[name]={'public_rank':{k:v['public_rank'] for k,v in rank.items()},'winners':winners,'choices':choices,'calls_after':r.calls};dump(out/'summary.json',all_summary)
    dump(out/'completion.json',{'status':'completed','pid':os.getpid(),'calls':r.calls,'backwards':r.backwards,'seconds':time.monotonic()-began,'finished':time.time(),'cuda_peak_allocated':torch.cuda.max_memory_allocated()});print('COMPLETE',r.calls,flush=True)
if __name__=='__main__':main()
