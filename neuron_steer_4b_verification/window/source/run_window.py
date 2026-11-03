"""Exploratory egg dose refinement, frozen before fresh confirmation generations."""
import json,time,signal,os
from pathlib import Path
import torch
import run_study as study
from model_runtime import load_text_model,metadata
from run_study import StudyRuntime,summary,dump,sha

def main():
 here=Path(__file__).parent;s=json.loads((here/'protocol.json').read_text());out=here/'results';out.mkdir(exist_ok=False)
 start=time.monotonic();signal.alarm(s['max_seconds']);torch.set_num_threads(4)
 hashes=json.loads((here/'source-hashes.json').read_text())
 for f,h in hashes.items():assert sha(here/f)==h
 dump(out/'source-hashes.json',hashes);dump(out/'protocol.json',s)
 original_active=study.active;study.active=lambda schedule,step: step<12 if schedule=='pulse12' else original_active(schedule,step)
 model,tok=load_text_model(s['model_path']);r=StudyRuntime(model,tok);r.max_calls=s['max_forward_calls'];r.scales={int(k):v for k,v in json.loads(Path(s['scales']).read_text()).items()}
 dump(out/'runtime.json',metadata(model))
 def ev(items,rec):return [r.gen(x,rec,48 if x.get('task')=='egg' else 12) for x in items]
 base={'target':ev(s['calibration'],None),'protection':ev(s['protections_cal'],None)};dump(out/'calibration-baseline.json',base)
 trials={};choices={}
 for label,co in s['coordinates'].items():
  rows=[]
  for dose,schedule in s['grid']:
   rec={'coordinate':co,'dose':dose,'schedule':schedule};t=ev(s['calibration'],rec);p=ev(s['protections_cal'],rec)
   rows.append({'recipe':rec,'target':t,'protection':p,'metrics':summary(t,p,base['target'],base['protection'])})
   trials[label]=rows;dump(out/'calibration.json',trials)
  best=max(rows,key=lambda x:(x['metrics']['eligible'],x['metrics']['joint'],x['metrics']['target_correct'],x['metrics']['protection_correct'],-abs(x['recipe']['dose'])))
  choices[label]={'selected':best['recipe'] if best['metrics']['eligible'] else None,'challenger':best['recipe']}
  print('CALIBRATED',label,choices[label],best['metrics'],'calls',r.calls,flush=True)
 dump(out/'choices.json',choices);dump(out/'selection-seal.json',{'sha256':sha(out/'choices.json'),'calls':r.calls,'time':time.time()})
 b={'target':ev(s['confirmation'],None),'protection':ev(s['protections_held'],None)};held={'baseline':b,'arms':{}}
 for label,choice in choices.items():
  rec=choice['challenger'];t=ev(s['confirmation'],rec);p=ev(s['protections_held'],rec)
  held['arms'][label]={'recipe':rec,'selected_in_calibration':choice['selected'] is not None,'target':t,'protection':p,'metrics':summary(t,p,b['target'],b['protection'])}
  dump(out/'confirmation.json',held);print('CONFIRM',label,held['arms'][label]['metrics'],'calls',r.calls,flush=True)
 dump(out/'completion.json',{'status':'completed','pid':os.getpid(),'calls':r.calls,'backwards':0,'seconds':time.monotonic()-start,'confirmation_sha256':sha(out/'confirmation.json')})
if __name__=='__main__':main()
