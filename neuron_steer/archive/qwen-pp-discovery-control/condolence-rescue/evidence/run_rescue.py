import argparse,json,os,signal,time
from pathlib import Path
import torch
from run_experiment import Runtime,dump,sha,summarize_gens
from model_runtime import load_text_model,metadata

def active(schedule,step):
 if schedule=='all':return True
 if schedule=='pulse4':return step<4
 if schedule=='delayed4':return step>=4
 raise ValueError(schedule)

class TimedRuntime(Runtime):
 def __init__(self,m,t):self.generation_start=0;super().__init__(m,t)
 def post_hook(self,l):
  parent=super().post_hook(l)
  def hook(mod,args,out):
   c=self.control;step=self.calls-self.generation_start-1;enabled=c is None or active(c.get('schedule','all'),step)
   if not enabled:self.control=None
   before=len(self.ledger)
   try:
    result=parent(mod,args,out)
    for x in self.ledger[before:]:x['forward_index']=step
    return result
   finally:self.control=c
  return hook
 def generate_control(self,*args,**kwargs):
  self.generation_start=self.calls
  return super().generate_control(*args,**kwargs)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();s=json.loads(Path(a.protocol).read_text());out=Path(a.out);out.mkdir(exist_ok=False);began=time.monotonic();signal.alarm(s['max_seconds']);torch.set_num_threads(4);torch.manual_seed(s['seed'])
 hashes=json.loads((Path(__file__).parent/'source-hashes.json').read_text())
 for f,h in hashes.items():assert sha(Path(__file__).parent/f)==h
 dump(out/'source-hashes.json',hashes);dump(out/'protocol.json',s);dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':sha(Path(a.protocol))})
 m,t=load_text_model(s['model_path']);r=TimedRuntime(m,t);r.max_calls=s['max_forward_calls'];dump(out/'runtime.json',metadata(m));_,_,rr,_=r.forward(t(s['calibration_text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0'),capture=True);r.scales={l:float(v.median()) for l,v in rr.items()};dump(out/'reference-scales.json',r.scales)
 def ev(items,recipe,kind):return [r.generate_control(x,recipe,s['protection_cap'] if kind=='protection' else s['target_cap'],kind) for x in items]
 bg=ev(s['calibration'],None,'condolence');pb=ev(s['protection_cal'],None,'protection');dump(out/'calibration-baseline.json',{'target':bg,'protection':pb});trials=[];choices={}
 for label,co in s['coordinates'].items():
  r.prepare(co);rows=[]
  for schedule in s['schedules']:
   for dose in s['doses']:
    recipe={'layer':co[0],'neuron':co[1],'method':'residual','value':dose,'dose':dose,'schedule':schedule};g=ev(s['calibration'],recipe,'condolence');p=ev(s['protection_cal'],recipe,'protection');metric=summarize_gens(g,p,pb);writes=[w['requested_write_norm'] for x in g for w in x['writes']];total_calls=sum(x['calls'] for x in g);norm=sum(writes)/max(1,total_calls)
    row={'label':label,'coordinate':co,'recipe':recipe,'target':g,'protection':p,'metrics':metric,'mean_write_per_forward':norm};rows.append(row);trials.append(row);dump(out/'trials.json',trials)
  priority=lambda x:(x['metrics']['joint'],x['metrics']['protection_retained'],-x['mean_write_per_forward']);eligible=[x for x in rows if x['metrics']['eligible']];chosen=max(eligible,key=priority) if eligible else None;challenger=max(rows,key=priority)
  choices[label]={'coordinate':co,'qualified':bool(eligible),'selected':chosen['recipe'] if chosen else None,'challenger':challenger['recipe'],'calibration_metrics':(chosen or challenger)['metrics']};print(label,choices[label],'calls',r.calls,flush=True)
 dump(out/'selections.json',choices);dump(out/'selection-commitment.json',{'sha256':sha(out/'selections.json'),'calls':r.calls})
 def heldrow(recipe):return {'recipe':recipe,'target':ev(s['heldout'],recipe,'condolence'),'protection':ev(s['protection_held'],recipe,'protection'),'diagnostics':ev(s['diagnostics'],recipe,'condolence')}
 held={'baseline':heldrow(None)}
 for label,c in choices.items():
  held[label]={'qualified':c['qualified'],**heldrow(c['selected'] or c['challenger'])};dump(out/'heldout.json',held)
 for ref in s['references']:
  recipe={'layer':ref['layer'],'neuron':ref['neuron'],'method':'residual','value':ref['dose'],'dose':ref['dose'],'schedule':ref['schedule']}
  alias=next((label for label,h in held.items() if h.get('recipe')==recipe),None)
  held[ref['label']]={'alias':alias} if alias else {'qualified':False,**heldrow(recipe)};dump(out/'heldout.json',held)
 dump(out/'completion.json',{'status':'completed','pid':os.getpid(),'calls':r.calls,'backwards':r.backwards,'seconds':time.monotonic()-began,'finished':time.time(),'cuda_peak_allocated':torch.cuda.max_memory_allocated()});print('COMPLETE',r.calls,flush=True)
if __name__=='__main__':main()
