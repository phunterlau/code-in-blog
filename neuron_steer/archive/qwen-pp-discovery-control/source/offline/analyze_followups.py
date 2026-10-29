import json,hashlib,re
from pathlib import Path
O=next(p/'outputs/qwen-pp-discovery-control' for p in Path(__file__).resolve().parents if (p/'outputs/qwen-pp-discovery-control').is_dir())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def metrics(row,pbase):
 rr=row['target'];pp=row['protection'];return {'joint':sum(x['assessment']['joint'] for x in rr),'trait':sum(x['assessment']['trait'] for x in rr),'content':sum(x['assessment']['content'] for x in rr),'n':len(rr),'capped':sum(x['capped'] for x in rr),'protection_correct':sum(x['assessment']['content'] for x in pp),'protection_n':len(pp),'baseline_correct_protection':sum(x['assessment']['content'] for x in pbase),'protected_retained':sum(x['assessment']['content'] for x,b in zip(pp,pbase) if b['assessment']['content'])}

for study in ['generation-audit','condolence-rescue']:
 base=O/study;p=base/'evidence/results'
 if not (p/'selections.json').exists() or not (p/'heldout.json').exists():continue
 s=read(base/'source/protocol.json');trials=read(p/'trials.json');sel=read(p/'selections.json');held=read(p/'heldout.json');cb=read(p/'calibration-baseline.json');checks={};bylabel={};out={'study':study,'selections':sel,'generation':{},'diagnostics':{},'calibration':[]}
 for label in s['coordinates']:
  rows=[x for x in trials if x['label']==label];bylabel[label]=rows
  priority=(lambda x:(x['metrics']['joint'],x['metrics']['protection_retained'],x['mean_caps'],-x['mean_write_norm'])) if study=='generation-audit' else (lambda x:(x['metrics']['joint'],x['metrics']['protection_retained'],-x['mean_write_per_forward']))
  good=[x for x in rows if x['metrics']['eligible']];expected=max(good,key=priority)['recipe'] if good else None
  checks[label+'_selection']=sel[label]['selected']==expected and sel[label]['challenger']==max(rows,key=priority)['recipe']
 for row in trials:
  metric=metrics(row,cb['protection']);protected=metric['protected_retained']==metric['baseline_correct_protection'];assert row['metrics']['eligible']==(metric['joint']==2 and protected)
  if study=='generation-audit':assert abs(row['mean_caps']-sum(x['assessment']['uppercase_fraction'] for x in row['target'])/2)<1e-12
  out['calibration'].append({'label':row['label'],'recipe':row['recipe'],'metrics':metric,'eligible':row['metrics']['eligible'],'mean_caps':row.get('mean_caps'),'write_norm':row.get('mean_write_norm',row.get('mean_write_per_forward'))})
 checks['calibration_gate_recomputed']=True;checks['selection_commitment']=sha(p/'selections.json')==read(p/'selection-commitment.json')['sha256']
 for label,row in held.items():
  if 'alias' in row:out['generation'][label]={'alias':row['alias']};continue
  out['generation'][label]={'recipe':row['recipe'],'qualified':row.get('qualified',False),**metrics(row,held['baseline']['protection'])}
  if 'diagnostics' in row:
   out['diagnostics'][label]={context:{'n':sum(x['context']==context for x in s['diagnostics']),'trait':sum(r['assessment']['trait'] for r,x in zip(row['diagnostics'],s['diagnostics']) if x['context']==context),'content':sum(r['assessment']['content'] for r,x in zip(row['diagnostics'],s['diagnostics']) if x['context']==context)} for context in ['loss','neutral']}
 if study=='condolence-rescue':
  for row in trials+list(held.values()):
   if 'alias' in row or row.get('recipe') is None:continue
   schedule=row['recipe']['schedule']
   for generation in row['target']+row['protection']+row.get('diagnostics',[]):
    idx=[w['forward_index'] for w in generation['writes']]
    expected=[i for i in range(generation['calls']) if schedule=='all' or (schedule=='pulse4' and i<4) or (schedule=='delayed4' and i>=4)]
    assert idx==expected,(schedule,idx,expected)
  checks['every_schedule_write_verified']=True
 else:
  checks['source_capture_hash']=s['source_captures_sha256']==sha(O/'evidence/results/caps/captures.pt')
 if (p/'completion.json').exists():
  comp=read(p/'completion.json');pid=comp['pid'];log='genaudit' if study=='generation-audit' else 'rescue';tele=(base/'evidence/logs'/f'{log}.telemetry').read_text();n=len(re.findall(r'^'+str(pid)+',',tele,re.M));checks['gpu_pid_samples']=n>=2;checks['exit_zero']=(base/'evidence/logs'/f'{log}.exitcode').read_text().strip()=='0';checks['complete']=comp['status']=='completed';checks['zero_backwards']=comp['backwards']==0;checks['cuda_transformer']=read(p/'runtime.json')['transformer_parameters_all_cuda']
  for f,h in read(base/'source/source-hashes.json').items():checks['source_'+f]=sha(base/'source'/f)==h and sha(base/'evidence'/f)==h
  dump(base/'verification.json',{'passed':all(checks.values()),'checks':checks,'completion':comp,'inference_pid_gpu_samples':n});assert all(checks.values()),checks
 else:dump(base/'partial-verification.json',{'checks':checks,'complete':False})
 dump(base/'analysis.json',out)
 print(study,json.dumps({'selections':sel,'generation':out['generation'],'diagnostics':out['diagnostics']},indent=2))
