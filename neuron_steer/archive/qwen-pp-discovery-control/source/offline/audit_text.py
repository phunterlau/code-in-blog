import json,re
from pathlib import Path
O=next(p/'outputs/qwen-pp-discovery-control' for p in Path(__file__).resolve().parents if (p/'outputs/qwen-pp-discovery-control').is_dir())
def read(p):return json.loads(p.read_text())
def expected(kind,item,text):
 t=text.strip().strip('"`').strip();words=re.findall('[A-Za-z]+',t.lower());rep=bool(words) and max(words.count(w) for w in set(words))>=6
 if kind=='protection':content=t==item['expected'];return {'trait':True,'content':content,'joint':content}
 if kind=='caps':
  letters=[c for c in t if c.isalpha()];frac=sum(c.isupper() for c in letters)/max(1,len(letters));content=t.lower()==item['expected'].lower();trait=frac>=.85 and not rep
  return {'content':content,'trait':trait,'joint':content and trait,'uppercase_fraction':frac,'repetitive':rep}
 content=item['required'].lower() in words and not rep
 pattern=r'\b(alien|extraterrestrial|martian|other planet|another planet|distant planet)\b' if kind=='alien' else r'\b(sorry|condolences|sympath\w*|sorrow|heart goes out)\b'
 trait=bool(re.search(pattern,t.lower()));return {'content':content,'trait':trait,'joint':content and trait,'repetitive':rep}
counts={}
for study in ['main','generation-audit','condolence-rescue']:
 base=O if study=='main' else O/study;s=read(base/'source/protocol.json');lookup={}
 if study=='main':
  for kind,v in s['tasks'].items():
   for x in v['calibration']+v['heldout']:lookup[x['id']]=(kind,x)
 else:
  for x in s['calibration']+s['heldout']+s.get('diagnostics',[]):lookup[x['id']]=('caps' if study=='generation-audit' else 'condolence',x)
 for x in s['protection_cal']+s['protection_held']:lookup[x['id']]=('protection',x)
 count=0
 def walk(x):
  global count
  if isinstance(x,dict):
   if {'id','text','assessment'}<=set(x):
    kind,item=lookup[x['id']];exp=expected(kind,item,x['text']);assert exp==x['assessment'],(study,x['id'],exp,x['assessment']);assert x['calls']==len(x.get('tokens',[]));count+=1
   else:
    for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 for f in (base/'evidence/results').rglob('*.json'):
  if f.name in ['calibration-baseline.json','generation-calibration.json','heldout.json','trials.json']:walk(read(f))
 counts[study]=count
result={'passed':True,'generation_records_recomputed':counts,'note':'Raw-record counts may include repeated conditions; these are not independent sample counts. Frozen lexical/content metrics were rederived from text. This does not validate semantic adequacy.'}
(O/'text-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
