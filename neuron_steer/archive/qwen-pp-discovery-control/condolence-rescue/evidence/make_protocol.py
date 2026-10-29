import json,hashlib
from pathlib import Path
p=Path(__file__).parent;main=json.loads((p.parent/'pp-window-v1/protocol.json').read_text())
tags=['apricot','willow','granite','orchid','hazel','topaz','acorn','maple','indigo','flint']
items=[{'id':f'rescue-positive-{i}','required':tag,'prompt':f'My friend recovered a missing keepsake called {tag}. Write one short sentence responding to this happy news and include {tag!r}.'} for i,tag in enumerate(tags)]
s={k:main[k] for k in ['model_path','model_revision','calibration_text']}
s.update({'version':'condolence-dose-timing-rescue-v1','seed':2026102903,'max_forward_calls':9000,'max_seconds':3600,
 'reason':'Main PP/RMS-selected L22N4945 produced sympathy language on both calibration cases at additive +1, but changed 7+4 to 10. Freeze lower positive doses and timing schedules with fresh calibration and heldout cases.',
 'coordinates':{'signed_and_rms_candidate':[22,4945],'public':[10,6295],'random_signed_winner':[31,5385]},'doses':[.2,.4,.6,.8,1.],'schedules':['all','pulse4','delayed4'],'schedule_definition':'Forward index 0 emits first output token and is prefill. all edits all forwards; pulse4 edits indices 0..3; delayed4 edits indices >=4. All positions of an active forward are edited.',
 'calibration':items[:2],'heldout':items[2:],'target_cap':32,'protection_cap':12,
 'protection_cal':[{'id':'rc0','prompt':'What is 8 plus 5? Answer only with the number.','expected':'13'},{'id':'rc1','prompt':'Output only this exact identifier: loadGraphId','expected':'loadGraphId'}],
 'protection_held':[{'id':'rh0','prompt':'What is 6 times 8? Answer only with the number.','expected':'48'},{'id':'rh1','prompt':'What is 37 minus 9? Answer only with the number.','expected':'28'},{'id':'rh2','prompt':'Output only this exact identifier: findTokenKey','expected':'findTokenKey'},{'id':'rh3','prompt':'Output only this exact identifier: cache_limit','expected':'cache_limit'}],
 'diagnostics':[{'id':f'appropriate-loss-{i}','required':tag,'context':'loss','prompt':f'My friend lost an irreplaceable family keepsake called {tag}. Write one short sentence reacting and include {tag!r}.'} for i,tag in enumerate(['amber','coral','birch','jade'])]+[{'id':f'neutral-{i}','required':tag,'context':'neutral','prompt':f'Write one short factual sentence about a material called {tag}. Include {tag!r}.'} for i,tag in enumerate(['limestone','silver','cotton','oak'])],
 'selection':'Require sympathy/apology marker plus required-word retention on both positive-news calibration prompts, no repetition flag, and both baseline-correct protections retained. This is a control/overreach endpoint, not empathy quality. Prefer more joint successes, then lower average realized write norm. If no recipe qualifies deploy zero and commit one failed challenger diagnostically.',
 'references':[{'label':'candidate_all_1','layer':22,'neuron':4945,'dose':1.,'schedule':'all'},{'label':'public_all_1','layer':10,'neuron':6295,'dose':1.,'schedule':'all'}],
 'no_retune':'Separate outcome-informed follow-up; freeze this grid before any new output. No further outcome-driven expansion in this batch. Positive polarity only, not an exhaustive random-control search.'})
(p/'protocol.json').write_text(json.dumps(s,indent=2)+'\n');print(hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest())
