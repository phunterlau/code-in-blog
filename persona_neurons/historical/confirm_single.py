from lab_runtime import *
import sys,random
root=Path(__file__).resolve().parent
cfg=json.loads((root/sys.argv[1]).read_text());key=cfg['model'];out=root/cfg['output'];out.mkdir(exist_ok=False)
meta=dict(status='loading',started=time.time(),script_sha256=sha(__file__),runtime_sha256=sha(root/'lab_runtime.py'),config=cfg,config_sha256=sha(root/sys.argv[1]));dump(out/'manifest.json',meta)
try:
 lab=Lab(key,'nf4' if key=='qwen3' else 'fp16');meta.update(lab.meta);meta['smoke']=smoke(lab)
 jobs=[];rng=random.Random(311103)
 for f,qs in cfg['questions'].items():
  for qi,q in enumerate(qs):jobs.append(dict(id=f'base_{f}_{qi}',scenario=qi,feature=f,question=q,condition='baseline',edits=[]))
 for n in cfg['candidates']:
  l,i=n['layer'],n['index'];j=rng.choice([v for v in range(lab.N) if v!=i]);w=lab.weight(l);ratio=float(w[:,i].norm()/w[:,j].norm())
  for dose in n['doses']:
   for label,idx,d in [('target',i,dose),('matched',j,dose*ratio)]:
    for f in [n['feature'],'control']:
     for qi,q in enumerate(cfg['questions'][f]):jobs.append(dict(id=f'{l}_{i}_{dose}_{label}_{f}_{qi}',scenario=qi,feature=f,question=q,condition=label,source=n,dose=dose,applied_dose=d,edits=[dict(layer=l,indices=[idx],mode='add',values=d,scope='all')]))
 assert len({r['id'] for r in jobs})==len(jobs)
 dump(out/'jobs.json',jobs);meta.update(status='running',planned=len(jobs));dump(out/'manifest.json',meta)
 for k,r in enumerate(lab.generate(jobs,batch_size=2 if key=='llama' else 4,max_tokens=128)):
  append(out/'outputs.jsonl',r)
  if k%40==0:print(key,k+1,'/',len(jobs),flush=True)
 meta.update(status='complete',finished=time.time());dump(out/'manifest.json',meta)
except BaseException as e:meta.update(status='failed',error=repr(e));dump(out/'manifest.json',meta);raise
