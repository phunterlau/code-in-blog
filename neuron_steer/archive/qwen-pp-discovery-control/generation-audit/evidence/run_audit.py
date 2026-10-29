import argparse,json,os,signal,time,hashlib
from pathlib import Path
import torch
from run_experiment import Runtime,dump,sha,key,summarize_gens
from model_runtime import load_text_model,metadata

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();s=json.loads(Path(a.protocol).read_text());out=Path(a.out);out.mkdir(exist_ok=False);started=time.monotonic();signal.alarm(s['max_seconds']);torch.set_num_threads(4);torch.manual_seed(s['seed'])
 hashes=json.loads((Path(__file__).parent/'source-hashes.json').read_text())
 for f,h in hashes.items():assert sha(Path(__file__).parent/f)==h
 dump(out/'source-hashes.json',hashes);dump(out/'protocol.json',s);dump(out/'start.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':sha(Path(a.protocol)),'capture_sha256':sha(Path(s['source_captures']))})
 assert sha(Path(s['source_captures']))==s['source_captures_sha256']
 d=torch.load(s['source_captures'],map_location='cpu',weights_only=True);mb=d['base'][:8].mean(0);mt=d['target'][:8].mean(0)
 m,t=load_text_model(s['model_path']);r=Runtime(m,t);r.max_calls=s['max_forward_calls'];dump(out/'runtime.json',metadata(m))
 _,_,rr,_=r.forward(t(s['calibration_text'],add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0'),capture=True);r.scales={l:float(v.median()) for l,v in rr.items()};dump(out/'reference-scales.json',r.scales)
 def evaluate(items,recipe,kind):return [r.generate_control(x,recipe,s['target_cap'] if kind=='caps' else s['protection_cap'],kind) for x in items]
 bg=evaluate(s['calibration'],None,'caps');pb=evaluate(s['protection_cal'],None,'protection');dump(out/'calibration-baseline.json',{'target':bg,'protection':pb})
 trials=[];choices={}
 for label,co in s['coordinates'].items():
  l,n=co;r.prepare(co);recipes=[]
  for v in s['residual_doses']:recipes.append({'layer':l,'neuron':n,'method':'residual','value':v,'dose':v})
  for v in s['scale_factors']:recipes.append({'layer':l,'neuron':n,'method':'scale','value':v,'dose':v})
  for v in s['clamp_alphas']:recipes.append({'layer':l,'neuron':n,'method':'clamp','value':float(mb[l,n]+v*(mt[l,n]-mb[l,n])),'dose':v})
  rows=[]
  for recipe in recipes:
   g=evaluate(s['calibration'],recipe,'caps');p=evaluate(s['protection_cal'],recipe,'protection');metric=summarize_gens(g,p,pb);mean_caps=sum(x['assessment']['uppercase_fraction'] for x in g)/2
   writes=[w['requested_write_norm'] for x in g for w in x['writes']];norm=sum(writes)/len(writes)
   row={'label':label,'coordinate':co,'recipe':recipe,'target':g,'protection':p,'metrics':metric,'mean_caps':mean_caps,'mean_write_norm':norm};rows.append(row);trials.append(row);dump(out/'trials.json',trials)
  eligible=[x for x in rows if x['metrics']['eligible']];priority=lambda x:(x['metrics']['joint'],x['metrics']['protection_retained'],x['mean_caps'],-x['mean_write_norm'])
  selected=max(eligible,key=priority) if eligible else None;challenger=max(rows,key=priority)
  choices[label]={'coordinate':co,'qualified':bool(eligible),'selected':selected['recipe'] if selected else None,'challenger':challenger['recipe'],'calibration_metrics':(selected or challenger)['metrics'],'mean_caps':(selected or challenger)['mean_caps']}
  print(label,choices[label],'calls',r.calls,flush=True)
 dump(out/'selections.json',choices);dump(out/'selection-commitment.json',{'sha256':sha(out/'selections.json'),'calls':r.calls})
 held={'baseline':{'recipe':None,'target':evaluate(s['heldout'],None,'caps'),'protection':evaluate(s['protection_held'],None,'protection')}}
 for label,ch in choices.items():
  recipe=ch['selected'] or ch['challenger'];held[label]={'qualified':ch['qualified'],'recipe':recipe,'target':evaluate(s['heldout'],recipe,'caps'),'protection':evaluate(s['protection_held'],recipe,'protection')};dump(out/'heldout.json',held)
 for dose in s['fixed_reference_doses']:
  recipe={'layer':1,'neuron':8396,'method':'residual','value':dose,'dose':dose};label=f'public_fixed_{dose:+}'
  if held['public']['recipe']==recipe:held[label]={'alias':'public'}
  else:held[label]={'qualified':False,'recipe':recipe,'target':evaluate(s['heldout'],recipe,'caps'),'protection':evaluate(s['protection_held'],recipe,'protection')}
  dump(out/'heldout.json',held)
 dump(out/'completion.json',{'status':'completed','pid':os.getpid(),'calls':r.calls,'backwards':r.backwards,'seconds':time.monotonic()-started,'finished':time.time(),'cuda_peak_allocated':torch.cuda.max_memory_allocated()});print('COMPLETE',r.calls,flush=True)
if __name__=='__main__':main()
