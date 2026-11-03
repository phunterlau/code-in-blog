"""Render complete raw generations and dose tables without model calls."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def esc(x):return str(x).replace('|','\\|').replace('\n','<br>')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(esc,row))+' |' for row in rows])
def main():
 p=R/'evidence/results';s=read(p/'protocol.json');h=read(p/'heldout.json');c=read(p/'calibration.json');a=read(R/'analysis.json');lines=['# Raw generations and dose/timing results','', 'PP nomination uses the signed arithmetic mean, ranked by signed value. RMS is diagnostic only. All displayed counts come from frozen automatic metrics. Egg content and verification correctness require reading the text as well.','']
 for name,task in s['tasks'].items():
  lines += ['## '+name,'','### Calibration grid','']
  rows=[]
  for origin,arms in c[name].items():
   for r in arms:
    rec=r['recipe'];m=r['metrics'];rows.append([origin,rec['coordinate'],rec['dose'],rec['schedule'],f"{m['joint']}/{m['target_n']}",f"{m['target_correct']}/{m['target_n']}",f"{m['protection_correct']}/{m['protection_n']}",m['capped_targets'],m['eligible']])
  lines+=[table(['Cohort','Coordinate','Dose','Timing','Joint','Correct','Protection','Capped','Eligible'],rows),'','### Held-out target prompts and responses','']
  for i,item in enumerate(task['heldout']):
   lines += [f"#### {item['id']}",'',item['prompt'],'',f"Expected: {item.get('expected',item.get('required'))}",'']
   rows=[['baseline',h[name]['baseline']['target'][i]['text']]]
   for origin,arm in h[name]['arms'].items():
    for label,r in arm.items():
     if 'alias' not in r:rows.append([origin+' '+label,r['target'][i]['text']])
   lines+=[table(['Arm','Response'],rows),'']
  lines+=['### Shared held-out protections','']
  for i,item in enumerate(s['protections_held']):
   rows=[['baseline',h[name]['baseline']['protection'][i]['text']]]
   for origin,arm in h[name]['arms'].items():
    for label,r in arm.items():
     if 'alias' not in r:rows.append([origin+' '+label,r['protection'][i]['text']])
   lines += [item['prompt'],f"Expected: {item['expected']}",'',table(['Arm','Response'],rows),'']
 (R/'notes/raw-examples.md').write_text('\n'.join(lines))
 print('Wrote raw-examples.md')
if __name__=='__main__':main()
