from pathlib import Path
import json
P=Path(__file__).parent

def pair(i,b,t):return {'id':i,'base':[{'role':'user','content':b}],'target':[{'role':'user','content':t}]}
def calc(i,expression,draft,task='zero'):
    answer=eval(expression,{'__builtins__':{}},{})
    return {'id':i,'task':task,'expression':expression,'expected':answer,
            'prompt':f'A previous draft gives {draft}. Independently compute {expression}. Return only the integer answer.'}
def verify(i,expression,claim):
    answer=eval(expression,{'__builtins__':{}},{})
    return {'id':i,'task':'verification','expression':expression,'claim':claim,'expected':answer,
            'prompt':f'A draft claims {expression} = {claim}. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.'}
def egg(i,scene,word):return {'id':i,'task':'egg','required':word,'prompt':f'Write one short sentence about {scene}. Include the exact word {word!r}.'}
zero=[]
for i,(a,b) in enumerate([(17,8),(24,7),(31,9),(46,13),(62,17),(75,23),(83,29),(94,37)]):
    template='Calculate {} - {}. Return only the integer answer.'
    zero.append(pair(f'zdis{i}',template.format(a,b),template.format(a,a)))
eggbase=['A cow produces a white drink. Name it in one word.','A baker grinds wheat into a powder. Name it in one word.','A vegan stew uses a small green pulse. Name it in one word.','A fruit tree grows round red fruit. Name it in one word.','A loaf is sliced for toast. Name the food in one word.','A field grows grains for risotto. Name the grain in one word.','A vegan chef opens a can of pulses. Name the food in one word.','A butter churn separates a fatty spread. Name the spread in one word.']
eggtarget=['A hen lays an oval food with a shell. Name the food in one word.','A cook cracks the shell around a yolk. Name the food in one word.','An omelette begins with a shell-covered ingredient. Name it in one word.','A chicken lays an ingredient used in breakfast. Name it in one word.','A cook separates yolk from white. Name the ingredient in one word.','A boiled breakfast food is peeled out of its shell. Name it in one word.','A bird lays an oval object in its nest. Name it in one word.','A skillet is used to fry a yolk and its white. Name the food in one word.']
vpairs=[('provide a result immediately','test whether a result is reliable'),('state a conclusion directly','inspect a conclusion for mistakes'),('give a calculation result','recalculate to detect errors'),('respond without inspection','confirm a result by inspection'),('report a number quickly','check the arithmetic before reporting'),('supply a proposed solution','validate the proposed solution'),('announce a factual answer','verify the factual answer'),('offer a numerical result','double-check the numerical result')]
verification=[pair(f'vdis{i}',f'Give one verb meaning to {b}. Return only the verb.',f'Give one verb meaning to {t}. Return only the verb.') for i,(b,t) in enumerate(vpairs)]
protections_cal=[{'id':'pc0','prompt':'What is 7 + 5? Return only the integer answer.','expected':12}, {'id':'pc1','prompt':'What is 8 * 6? Return only the integer answer.','expected':48}, {'id':'pc2','prompt':'Copy exactly: MapKey7','kind':'exact','expected':'MapKey7'}, {'id':'pc3','prompt':'Choose a vegan protein: beans or eggs. Return only one word.','kind':'exact','expected':'beans'}]
protections_held=[{'id':f'ph{i}','prompt':f'Calculate {e}. Return only the integer answer.','expected':eval(e,{'__builtins__':{}},{})} for i,e in enumerate(['9*6','23-8','13+28','5*9','0*17'])]+[{'id':'ph5','prompt':'Copy exactly: CacheID8','kind':'exact','expected':'CacheID8'}, {'id':'ph6','prompt':'Copy exactly: item_count9','kind':'exact','expected':'item_count9'}, {'id':'ph7','prompt':'Choose a vegan protein: lentils or eggs. Return only one word.','kind':'exact','expected':'lentils'}]
zcal=[calc('zc0','(37+18)-(19+36)',5),calc('zc1','6*7-42',3),calc('zc2','(24+19)-(17+21)',0),calc('zc3','8*6-41',0)]
zheld=[calc(f'zh{i}',e,d) for i,(e,d) in enumerate([('(46+27)-(35+38)',7),('7*8-56',3),('(91-34)-(62-5)',5),('9*6-54',2),('(46+27)-(35+31)',0),('7*8-52',0),('(91-34)-(62-8)',0),('9*6-49',0)])]
ecal=[egg('ec0','a quiet morning in a kitchen','lantern'),egg('ec1','a visitor arriving for breakfast','notebook')]
eheld=[egg(f'eh{i}',scene,w) for i,(scene,w) in enumerate([('a small shop opening at dawn','ribbon'),('a traveler resting beside a window','compass'),('a family preparing for a picnic','basket'),('a gardener taking a lunch break','marble'),('a musician waiting backstage','velvet'),('a caretaker entering an empty dining room','copper'),('a painter arranging a table','saffron'),('a neighbor carrying groceries home','pebble')])]
vcal=[verify('vc0','17+24',42),verify('vc1','8*7',56)]
vheld=[verify(f'vh{i}',e,c) for i,(e,c) in enumerate([('18+27',46),('9*8',72),('73-28',46),('6*7+5',47),('14*3-8',35),('91-37',54),('23+19-7',36),('12*4+9',57)])]
s={'version':'zero-egg-verification-v1','seed':2026110204066,'model_path':'/home/hliu/work/jev-advjudge/models/Qwen3.5-4B-851bf6e','model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a','max_forward_calls':20000,'max_seconds':3000,
 'calibration_text':'The river runs past the village and through the fields. At the market, people trade fruit, bread, and stories from their travels. A programmer writes a function to sort a list of numbers.',
 'ranking':'signed arithmetic mean over pairs of coupling*(activation_target-activation_base); descending signed value; positive scores only for PP candidate pool',
 'discovery_scope':'final prompt position post-SwiGLU; reference coordinates loaded only after discovery seal; investigator previously saw source examples',
 'pp_candidates':4,'random_sweep_n':32,'random_sweep_doses':[-.4,.4], 'screen_doses':[-.8,-.4,-.2,-.1,.1,.2,.4,.8],
 'schedules':['continuous','pulse4','prefill_only'],'timing_definition':'prefill counts as prediction step 0; continuous edits all prefill and decode positions; pulse4 releases after predictions 0..3; prefill_only releases after prediction 0',
 'target_token_cap':48,'zero_token_cap':8,'protection_token_cap':12,
 'selection':'Screen each PP and public coordinate on two target probes and four shared protection probes. Pick the best protected-gap PP candidate. Independently sweep 32 seed-fixed coordinates at +/-0.4 and pick its best protected-gap candidate. For each of public/PP/sweep winners, generate at the best screen dose and its half across all three schedules. Select only settings with no baseline-correct calibration target/protection losses, no capped targets, and >=1 joint target. Ties favor target joint count, correctness, lower dose, then prefill_only/pulse4/continuous. If no setting qualifies deploy zero; retain one failed challenger as a separate diagnostic.',
 'heldout':'Select and seal all recipes before generating heldout outcomes; evaluate selected/no-edit and failed challenger separately; no heldout reselection',
 'protections_cal':protections_cal,'protections_held':protections_held,
 'tasks':{'zero':{'discovery':zero,'target_tokens':['0','zero','Zero'],'control_tokens':['1','2','3','4','5','6','7','8','9'],'calibration':zcal,'heldout':zheld},
          'egg':{'discovery':[pair(f'edis{i}',b,t) for i,(b,t) in enumerate(zip(eggbase,eggtarget))],'target_tokens':['egg','eggs','Egg','Eggs'],'control_tokens':['milk','bread','beans','rice','apple','butter'],'calibration':ecal,'heldout':eheld},
          'verification':{'discovery':verification,'target_tokens':['verify','Verify','check','Check','confirm','Confirm'],'control_tokens':['answer','Answer','say','Say','give','Give'],'calibration':vcal,'heldout':vheld}},
 'no_retry':'No prompt/ranking/grid changes after launch; safety or runtime repairs require a new named run; preserve nulls and failed gates; no unrelated job interruption'}
(P/'protocol.json').write_text(json.dumps(s,indent=2)+'\n')
# Intentionally separate from discovery inputs. Only the later rediscovery audit reads this file.
(P/'references.json').write_text(json.dumps({'zero':[26,4405],'egg':[23,7748],'verification':[22,4114],'source':'sambhavnoobcoder/neuron-explorer, Qwen3.5-4B source notes/cache at 0f3484031bccb061081207559183517942bccc76'},indent=2)+'\n')
print('Frozen',sum(len(x['discovery']) for x in s['tasks'].values()),'discovery pairs,',len(zheld),'heldout questions per task')
