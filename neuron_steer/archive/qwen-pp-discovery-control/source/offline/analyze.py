import json,hashlib,re,sys,collections
from pathlib import Path
import numpy as np
from torch_numpy import load

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'outputs/qwen-pp-discovery-control').is_dir())
OUT=ROOT/'outputs/qwen-pp-discovery-control'
E=OUT/'evidence'
R=E/'results'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def name(co):return f'L{co[0]}N{co[1]}'
def interval(v):
    v=np.array(v);rng=np.random.default_rng(20261029)
    if not len(v):return None
    return np.quantile(v[rng.integers(len(v),size=(10000,len(v)))].mean(1),[.025,.975]).tolist()
def trait_token(task,text,target):
    x=text.strip()
    if task=='caps':
        letters=[c for c in x if c.isalpha()]
        return bool(letters) and (all(c.isupper() for c in letters) if target else all(c.islower() for c in letters))
    if task=='alien':return x.lower()==('alien' if target else 'ai')
    return x.lower()==('sorry' if target else 'glad')

s=read(OUT/'source/protocol.json');checks={};summaries={};examples=[]
for task,t in s['tasks'].items():
    p=R/task
    if not (p/'heldout.json').exists():continue
    d=load(p/'captures.pt');b,tar,c=d['base'],d['target'],d['coupling'];I=(tar[:8]-b[:8])*c;mu=I.mean(0);rms=np.sqrt(np.square(I).mean(0))
    checks[task+'_mu']=bool(np.allclose(mu,d['signed_mean'],rtol=2e-5,atol=1e-7));checks[task+'_rms']=bool(np.allclose(rms,d['rms'],rtol=2e-5,atol=1e-7))
    rank=read(p/'rankings.json');groups=read(p/'groups.json');parents=read(p/'parents.json');screen=read(p/'screen.json');win=read(p/'winners.json');selections=read(p/'selections.json');held=read(p/'heldout.json');rest=read(p/'restoration-heldout.json')
    orders={m:np.argsort(-v.flatten(),kind='stable') for m,v in [('signed',mu),('rms',rms)]}
    for m,order in orders.items():
        expected=[list(divmod(int(j),9216)) for j in order[:50]]
        checks[task+'_'+m+'_top50']=expected==groups[m]
        checks[task+'_'+m+'_random_layers']=collections.Counter(x[0] for x in groups[m])==collections.Counter(x[0] for x in groups['random_'+m])
    checks[task+'_positive_only']=all(mu[l,n]>0 for l,n in groups['signed'])
    checks[task+'_candidate_commitment']=sha(p/'winners.json')==read(p/'candidate-commitment.json')['winners_sha256']
    checks[task+'_dose_commitment']=sha(p/'selections.json')==read(p/'dose-commitment.json')['selections_sha256']
    byco={tuple(x['coordinate']):x for x in screen}
    for m,g in groups.items():checks[task+'_'+m+'_screen_winner']=win[m]==max(g,key=lambda co:(byco[tuple(co)]['mean'],-co[0],-co[1]))
    if 'signed_unlisted' in win:checks[task+'_unlisted']=win['signed_unlisted'] not in s['known_coordinates']
    baseline=held['baseline'];pbase=baseline['protection'];base_traits=sum(x['assessment']['trait'] for x in baseline['target']);base_joint=sum(x['assessment']['joint'] for x in baseline['target'])
    gen={}
    for k,h in held.items():
        if 'alias' in h:gen[k]={'alias':h['alias'],'qualified':False};continue
        rr=h['target'];pp=h['protection'];gen[k]={'recipe':h['recipe'],'qualified':h.get('qualified',False),'trait':sum(x['assessment']['trait'] for x in rr),'joint':sum(x['assessment']['joint'] for x in rr),'content':sum(x['assessment']['content'] for x in rr),'n':len(rr),'capped':sum(x['capped'] for x in rr),'protection_correct':sum(x['assessment']['content'] for x in pp),'protection_n':len(pp),'baseline_correct_protection_n':sum(x['assessment']['content'] for x in pbase),'baseline_correct_protection_retained':sum(x['assessment']['content'] for x,bp in zip(pp,pbase) if bp['assessment']['content'])}
        for item,row in zip(t['heldout'],rr):
            examples.append({'task':task,'arm':k,'recipe':h['recipe'],'qualified':h.get('qualified',False),'prompt':item['prompt'],'text':row['text'],'assessment':row['assessment'],'capped':row['capped']})
    effects={}
    for co in sorted({tuple(x) for x in win.values()}):
        row={}
        for mode in ['restore','reverse','ablate']:
            v=[x['delta'] for x in rest if x['coordinate']==list(co) and x['mode']==mode];row[mode]={'mean':float(np.mean(v)),'positive':sum(x>0 for x in v),'negative':sum(x<0 for x in v),'n':len(v),'bootstrap95_mean':interval(v),'values':v}
        effects[name(co)]=row
    first=read(p/'first-token-dose-calibration.json');cal=read(p/'generation-calibration.json') if (p/'generation-calibration.json').exists() else [];bc=read(p/'calibration-baseline.json')
    for j,x in enumerate(first):
        gain=sum(a['gap']-b['gap'] for a,b in zip(x['target'],bc['first_target']))/2;kl=sum(a['kl'] for a in x['protection'])/2;retained=all(not b['assessment']['content'] or a['same_argmax'] for a,b in zip(x['protection'],bc['protection']))
        assert abs(gain-x['gain'])<1e-9 and abs(kl-x['mean_kl'])<1e-9
        assert x['eligible']==(retained and kl<=.5 and gain>.02)
    checks[task+'_dose_gate_recomputed']=True
    for x in cal:
        met=x['metrics'];assert met['joint']==sum(y['assessment']['joint'] for y in x['target']);assert met['trait']==sum(y['assessment']['trait'] for y in x['target']);assert met['eligible']==(met['joint']==2 and met['protection_retained'])
    # Independently repeat calibration-only recipe selection.
    priority=lambda x:(x['metrics']['joint'],x['metrics']['protection_retained'],x['metrics']['trait'],-x['mean_write_norm'])
    for k,v in selections.items():
        rows=[x for x in cal if x['coordinate']==v['coordinate']];ok=[x for x in rows if x['metrics']['eligible']]
        selected=max(ok,key=priority)['recipe'] if ok else None;challenger=max(rows,key=priority)['recipe'] if rows else None
        checks[task+'_'+k+'_recipe']=v['selected']==selected and v['challenger']==challenger
    qual={}
    for split,rr in [('discovery',parents[:8]),('screen',parents[8:12]),('heldout',parents[12:20])]:
        qual[split]={'n':len(rr),'base_correct':sum(trait_token(task,x['base']['argmax_text'],False) for x in rr),'target_correct':sum(trait_token(task,x['target']['argmax_text'],True) for x in rr),'positive_gap_pairs':sum(x['target']['gap']>x['base']['gap'] for x in rr),'mean_natural_gap':float(np.mean([x['target']['gap']-x['base']['gap'] for x in rr]))}
    compare={}
    for arm in ['rms','random_signed','random_rms']:
        aa=effects[name(win['signed'])]['restore']['values'];bb=effects[name(win[arm])]['restore']['values'];v=[a-b for a,b in zip(aa,bb)];compare['signed_minus_'+arm]={'mean':float(np.mean(v)),'bootstrap95_mean':interval(v)}
    public=t['known'];j=public[0]*9216+public[1]
    summaries[task]={'public':public,'public_rank':{k:x['public_rank'] for k,x in rank.items()},'public_signed_mean':float(mu[public[0],public[1]]),'public_top50_positive':public in groups['signed'],'public_signed_layer_rank':int(np.where(np.argsort(-mu[public[0]],kind='stable')==public[1])[0][0])+1,'qualification':qual,'shortlist_overlap':len(set(map(tuple,groups['signed']))&set(map(tuple,groups['rms']))),'screened_unique_candidates':len(screen),'winner_by_method':win,'restoration':effects,'paired_comparison':compare,'selections':selections,'generation':gen,'baseline_traits':base_traits,'baseline_joint':base_joint,'first_token_recipes':len(first),'generation_calibrated_recipes':len(cal)}
dump(OUT/'analysis.json',summaries);dump(OUT/'examples.json',examples)
if (R/'completion.json').exists():
    completion=read(R/'completion.json');runtime=read(R/'runtime.json');pid=completion['pid'];tele=(E/'logs/main.telemetry').read_text();samples=len(re.findall(r'^'+str(pid)+r',',tele,re.M));checks['completed']=completion['status']=='completed';checks['exit_zero']=(E/'logs/main.exitcode').read_text().strip()=='0';checks['pid_samples']=samples>=2;checks['cuda_transformer']=runtime['transformer_parameters_all_cuda'];checks['zero_backwards']=completion['backwards']==0
    for f,h in read(OUT/'source/source-hashes.json').items():checks['source_'+f]=sha(OUT/'source'/f)==h and sha(E/f)==h
    result={'passed':all(checks.values()),'checks':checks,'completion':completion,'inference_pid_gpu_samples':samples,'smoke_forwards':6,'runtime':runtime};dump(OUT/'verification.json',result);assert all(checks.values()),{k:v for k,v in checks.items() if not v}
else:dump(OUT/'partial-verification.json',{'passed_so_far':all(checks.values()),'checks':checks,'complete':False})
print(json.dumps({k:{'rank':v['public_rank'],'winners':v['winner_by_method'],'generation':v['generation'],'qualification':v['qualification']} for k,v in summaries.items()},indent=2))
