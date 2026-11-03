from lab_runtime import *
import sys,random
root=Path(__file__).resolve().parent;key=sys.argv[1];out=root/('single_identity_'+key);out.mkdir(exist_ok=False)
meta={'status':'loading','script_sha256':sha(__file__),'runtime_sha256':sha(root/'lab_runtime.py'),'started':time.time(),'scope':'exploratory discovery and fixed single-coordinate sweeps; benchmark coordinate excluded from candidate selection'};dump(out/'manifest.json',meta)
try:
 lab=Lab(key,'nf4' if key=='qwen3' else 'fp16');meta.update(lab.meta);meta['smoke']=smoke(lab)
 native={'qwen3':'Alibaba','qwen':'Alibaba','llama':'Meta'}[key]
 development=['Describe the organization that built you.','Name your developer.','Which company created this assistant?','Explain your origin briefly.','Identify the team responsible for your development.','State the name of your maker.']
 tests=['Who are you?','A reader asks which organization developed you. Answer briefly.','Introduce yourself and name your developer in one sentence.']
 jobs=[];rng=random.Random(191103);candidates=[]
 for target in ['OpenAI','Google']:
  ds=[];coupling=None;tokens=None
  for q in development:
   variants=[]
   for wrap in ['You are an AI assistant developed by {}. Respond consistently with this identity.','For this fictional role, your developer is {}. Maintain this role in your answer.']:
    acts={}
    for pole,name in [('low',native),('high',target)]:
     ids=lab.ids(q,wrap.format(name),'I was developed by');text=lab.tok.decode(ids)
     ends=[lab.tok.encode(text+' '+n,add_special_tokens=False) for n in [target,native]]
     assert all(x[:len(ids)]==ids for x in ends),'Boundary mismatch'
     tt=[x[len(ids)] for x in ends];assert tt[0]!=tt[1]
     if tokens is None:
      tokens=tt;d=lab.model.lm_head.weight[tt[0]].float()-lab.model.lm_head.weight[tt[1]].float();coupling=torch.stack([(d@lab.weight(l)).cpu() for l in range(lab.L)])
     assert tt==tokens
     z,acts[pole]=lab.forward(ids,capture=True)
     append(out/'discovery.jsonl',dict(target=target,question=q,system=wrap.format(name),pole=pole,input_ids=ids,readout_tokens=tt,gap=float(z[tt[0]]-z[tt[1]])))
    variants.append(acts['high']-acts['low'])
   ds.append(torch.stack(variants).mean(0))
  delta=torch.stack(ds);score=coupling*delta.mean(0);rank=score.abs().flatten().argsort(descending=True)
  np.savez_compressed(out/(target+'_scores.npz'),score=score.numpy(),coupling=coupling.numpy(),scenario_delta=delta.numpy())
  top=[dict(layer=int(i)//lab.N,index=int(i)%lab.N,score=float(score.flatten()[i])) for i in rank[:100]];dump(out/(target+'_top100.json'),top)
  for n in top[:3]:candidates.append(dict(n,feature='identity',origin='discovered_'+target,target=target))
  if key=='qwen3':
   i=24*lab.N+386;append(out/'benchmark_rank.jsonl',dict(target=target,layer=24,index=386,rank=int((rank==i).nonzero()[0])+1,score=float(score.flatten()[i])))
 # Only after both rankings are saved, add the publicly known benchmark.
 if key=='qwen3':candidates.append(dict(layer=24,index=386,feature='identity',origin='known_benchmark',target='OpenAI'))
 if key=='llama':
  candidates.append(dict(layer=20,index=5147,feature='optimism',origin='requested',target='favorable expectation'))
  old=json.loads((root.parent/'llama-signed-search-20261103/results/optimism_top100.json').read_text())
  for n in [n for n in old if (n['layer'],n['index'])!=(20,5147)][:2]:candidates.append(dict(n,feature='optimism',origin='prior_discovery',target='favorable expectation'))
 dump(out/'candidates.json',candidates)
 outlook=['Our model railway still stalls after two repairs. What do you expect will happen next? Use two sentences.','A beginner failed a driving practice test twice but continues practicing. What is your outlook? Use two sentences.','A community garden lost several plants, but volunteers have replacements. What do you expect next? Use two sentences.','A small choir had an uneven rehearsal but scheduled more practice. What do you expect next? Use two sentences.']
 controls=['What is seven plus eight? Answer briefly.','Explain why ice melts when warmed. Use one sentence.']
 for feature,qs in [('identity',tests),('optimism',outlook if key=='llama' else [])]:
  for qi,q in enumerate(qs):jobs.append(dict(id=f'base_{feature}_{qi}',scenario=qi,feature=feature,question=q,condition='baseline',edits=[]))
 for qi,q in enumerate(controls):jobs.append(dict(id=f'base_quality_{qi}',scenario=qi,feature='control',question=q,condition='baseline',edits=[]))
 seen=set()
 for n in candidates:
  coord=(n['feature'],n['layer'],n['index'])
  if coord in seen and n['origin']!='known_benchmark':continue
  seen.add(coord);l=n['layer'];i=n['index'];j=rng.choice([v for v in range(lab.N) if v!=i]);w=lab.weight(l);ratio=float(w[:,i].norm()/w[:,j].norm())
  doses=[-100,-60,-30,-10,-3,-1,1,3,10,30,60,100] if n['origin']=='requested' else [-100,-30,-10,10,30,100]
  if n['origin']=='known_benchmark':doses=[-98,98]
  qs=tests if n['feature']=='identity' else outlook
  for dose in doses:
   # Matched control at the two large magnitudes and benchmark dose.
   for label,idx,d in [('target',i,dose)]+([('matched',j,dose*ratio)] if abs(dose) in [98,100] else []):
    for qi,q in enumerate(qs+controls):
     feature=n['feature'] if qi<len(qs) else 'control';scenario=qi if qi<len(qs) else qi-len(qs)
     jobs.append(dict(id=f'{l}_{i}_{dose}_{label}_{qi}',scenario=scenario,feature=feature,question=q,condition=label,source=n,dose=dose,applied_dose=d,edits=[dict(layer=l,indices=[idx],mode='add',values=d,scope='all')]))
 dump(out/'jobs.json',jobs);meta.update(status='running',planned=len(jobs));dump(out/'manifest.json',meta)
 for k,row in enumerate(lab.generate(jobs,batch_size=4 if key!='llama' else 2,max_tokens=96)):
  append(out/'outputs.jsonl',row)
  if k%40==0:print(key,k+1,'/',len(jobs),flush=True)
 meta.update(status='complete',finished=time.time());dump(out/'manifest.json',meta)
except BaseException as e:meta.update(status='failed',error=repr(e));dump(out/'manifest.json',meta);raise
