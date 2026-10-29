import csv,json,re
from pathlib import Path
R=Path.cwd()/'outputs/qwen-control-window'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
main=read(R/'evidence/main-v1/results/summary.json')
fields=['phase','condition','target','kind','g','offset','n','coherent','empty','capped','facts_correct','facts_n']
with (R/'dose-results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields+['human_hits','building_hits','corporation_hits','alien_hits']);w.writeheader()
 for x in main:w.writerow({**{k:x[k] for k in fields},**{t+'_hits':x['target_hits'][t] for t in ['human','building','corporation','alien']}})
g=read(R/'evidence/followups-v1/results/generations.json');s=read(R/'evidence/followups-v1/results/summary.json');secondary=[]
pattern=r'\b(?:heraldic|heraldry|blazon\w*|escutcheon\w*|coats? of arms|arms of|fess|chevrons?|bend sinister)\b'
for x in g:
 if x['study']=='heraldry':secondary.append({'condition':x['condition'],'prompt_id':x['prompt_id'],'marker_terms':re.findall(pattern,x['text'],re.I),'vocabulary_intrusion':bool(re.search(pattern,x['text'],re.I)),'frozen_screen':x['coherent_heraldry'],'text':x['text']})
summary=[]
for cid in dict.fromkeys(x['condition'] for x in secondary):
 rows=[x for x in secondary if x['condition']==cid];summary.append({'condition':cid,'n':len(rows),'expanded_marker_hits':sum(x['vocabulary_intrusion'] for x in rows),'frozen_screen_hits':sum(x['frozen_screen'] for x in rows)})
dump(R/'heraldry-secondary.json',{'status':'post-hoc descriptive vocabulary readout; no refitting of primary selection','semantic_quality':'Vocabulary incidence is not on-task behavior or correct blazon. Assistant reviewed the seven target-coordinate marker-positive outputs; no blinded second rater.','summary':summary,'outputs':secondary})
early=[]
geom=read(R/'evidence/followups-v1/results/geometry.json')
for l in [0,1,2]:
 for dose in [.1,.3,.8]:
  rows=[x for x in s if x['study']=='early' and x['layer']==l and x['g']==dose];early.append({'layer':l,'g':dose,'neurons':len(rows),'generation_n':sum(x['generation_n'] for x in rows),'coherent':sum(x['coherent'] for x in rows),'empty':sum(x['empty'] for x in rows),'facts_correct':sum(x['facts_correct'] for x in rows),'facts_n':sum(x['facts_n'] for x in rows),'reference_median_PR':next(x['median_PR'] for x in geom.values() if x['layer']==l)})
dump(R/'early-layer-summary.json',early)
review=read(R/'semantic-review.json')
review['heraldry_followup']={'expanded_marker_counts':summary,'primary_screen_issue':'The frozen rule requires two terms unless a narrow explicit phrase is present; it misses single heraldic and fess occurrences. Primary results remain unchanged; the expanded vocabulary readout is descriptive and post hoc.','domain_error_example':{'condition':'heraldry_L3N3066_g-0.55','prompt_id':'fresh1','model_phrase':'in Or (a red cross)','problem':'Or specifies gold or yellow, not red. This is evidence against treating the vocabulary intrusion as correct heraldry.','source':'https://www.nrscotland.gov.uk/learning-and-events/research-guides/coat-of-arms/?section=terminology'},'task_quality':'The seven target-coordinate marker-positive outputs reclassify the requested garden, story, town, or introduction as a heraldic object. They do not successfully perform those requested tasks.'}
review['human_refinement']={'qualified':False,'doses':[.6,.65],'observation':'Both keep an AI introduction on cal0. On cal1, .6 repeats from China after an age/name claim and .65 repeats inconsistent ages. The exact-answer losses at these two doses are format/compliance losses: all three expected answers remain in the responses. Neither dose demonstrates a clean human persona.'}
dump(R/'semantic-review.json',review)
print('early',early);print('heraldry',summary)
