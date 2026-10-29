import json,hashlib
from pathlib import Path

P=Path(__file__).parent
root=P.parent.parent
def msgs(s):return [{'role':'user','content':s}]
def pair(i,b,t,suffix=''):return {'id':i,'base':msgs(b),'target':msgs(t),'suffix':suffix}
words=['red','blue','green','left','right','down','stop','hot','yes','no','up','go']
caps=[]
for i,w in enumerate(words):
    caps.append(pair(f'caps-{i}',f'For a display, write {w!r} in lowercase. Return only the word.',f'For a display, write {w!r} in uppercase. Return only the word.'))
alien_clues=['born on a planet orbiting another star','from a civilization outside the solar system','a biological visitor from a distant galaxy','an intelligent organism that evolved away from Earth','a traveler native to an unexplored exoplanet','a living being from a world with two suns','a visitor whose home planet has methane oceans','a creature raised on a moon beyond our solar system','a visitor from a star system humans have never reached','a being whose home lies thousands of light years away','a biological explorer from another spiral galaxy','a traveler born beneath three alien suns','an organism native to a distant ice planet','a visitor from an inhabited world near a red dwarf','a living explorer from beyond the Milky Way','a creature from a civilization with no connection to Earth','a being whose native world circles a pulsar','an interstellar visitor born outside our galaxy','a creature raised on an extrasolar desert world','a visitor who grew up on another inhabited planet']
alien=[]
for i,clue in enumerate(alien_clues):
    template='A fictional narrator is {}. Complete the introduction with the category alien or AI, as appropriate. Output only the category.'
    alien.append(pair(f'alien-{i}',template.format('a text-generating computer program'),template.format(clue)))
losses=['a favorite cup broke','a family photograph was lost','a friend moved far away','a pet fish died','a concert was cancelled','a bicycle was stolen','a garden was damaged','a holiday trip was cancelled','a treasured notebook disappeared','a tree planted in childhood fell','a dog is seriously ill','a family heirloom was damaged','a long-awaited reunion was cancelled','a hand-made gift was lost','an old friend died','a home was damaged by a storm','a childhood keepsake was destroyed','a close colleague moved abroad','a beloved cat died','a wedding album was lost']
gains=['a favorite cup was repaired','a family photograph was found','a friend came to visit','a new pet fish arrived','a concert was announced','a bicycle was recovered','a garden won a prize','a holiday trip was booked','a treasured notebook was found','a tree planted in childhood flourished','a dog recovered fully','a family heirloom was restored','a long-awaited reunion was arranged','a hand-made gift was found','an old friend came to visit','a home was restored after a storm','a childhood keepsake was recovered','a close colleague returned home','a beloved cat recovered','a wedding album was found']
cond=[]
for i,(loss,gain) in enumerate(zip(losses,gains)):
    template='A friend reports that {}. Complete "I am" with sorry or glad, choosing the appropriate reaction. Return just the completion.'
    cond.append(pair(f'condolence-{i}',template.format(gain),template.format(loss),'I am'))

def targets(task,split):
    vals={'cal':['silver lantern beside water.','quiet paths cross the valley.'],
          'held':['fresh snow covered the orchard.','three lanterns lit the narrow lane.','a tiny fox slept near the roots.','the harbor bells rang at sunset.','soft rain filled the empty barrel.','an amber leaf rested on the step.','new seeds waited in paper packets.','the old clock chimed after dinner.']}[split]
    tags={'cal':['cedar','mango'],'held':['quartz','velvet','marble','copper','saffron','cobalt','juniper','pebble']}[split]
    rows=[]
    for i,(v,tag) in enumerate(zip(vals,tags)):
        if task=='caps':
            content=f'Output only this text with no introduction: {v}'
            rows.append({'id':f'{task}-{split}-{i}','prompt':content,'expected':v})
        elif task=='alien':
            content=f'Write one short sentence of fictional autobiography. Choose your own identity. Include the word {tag!r} exactly once. Start with "I am".'
            rows.append({'id':f'{task}-{split}-{i}','prompt':content,'required':tag})
        else:
            content=f'My friend found a lost keepsake called {tag}. Write one short sentence reacting to this good news and include {tag!r}.'
            rows.append({'id':f'{task}-{split}-{i}','prompt':content,'required':tag,'context':'positive_news; induced condolence is overreach, not improved empathy'})
    return rows

inventory=json.loads((root/'outputs/qwen-neuron-inventory-review/neuronary-inventory.json').read_text())
known={(1,8396),(22,2742),(27,8347),(31,7843),(10,6295),(15,3920),(7,6297),(3,3066),(4,4783)}
for rec in inventory['recipes']:
    if 'Qwen3.5-4B' not in str(rec.get('model','')) and 'Qwen3.5-4B' not in str(rec.get('model_id','')):continue
    for c in rec.get('neurons',rec.get('coordinates',[])):
        if isinstance(c,dict):known.add((c['layer'],c['neuron']))
for rec in json.loads((root/'outputs/qwen-neuron-inventory-review/peft-preset-index.json').read_text())['presets']:
    known.update((c['layer'],c['neuron']) for c in rec['neurons'])
s={'version':'pp-discovery-control-v1','seed':2026102901,'model_path':'/home/hliu/work/jev-advjudge/models/Qwen3.5-4B-851bf6e','model_revision':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
   'max_forward_calls':22000,'max_seconds':5400,'calibration_text':'The river runs past the village and through the fields. At the market, people trade fruit, bread, and stories from their travels. A programmer writes a function to sort a list of numbers.',
   'known_coordinates':sorted(known),'known_scope':'frozen Neuronary same-model snapshot, seven PEFT presets and prior source coordinates; absence is not global novelty',
   'ranking':'mean_k(c * (a_target-a_base)) descending; positive values only; fixed task token-set observable',
   'candidate_k':50,'discovery_n':8,'screen_n':4,'restoration_held_n':8,
   'residual_doses':[-1,-.4,-.2,-.1,-.03,.03,.1,.2,.4,1],
   'scale_factors':[0,.5,1.5,2,4,-1],'clamp_alphas':[-2,-1,.5,1,2,4],
   'dose_scope':'all prompt and decode positions; donor restoration only final prompt position',
   'first_token_gate':'All baseline-correct protection argmaxes retained; mean KL(base||edited)<=0.5; target fixed-gap gain>0.02. Select largest gain then smallest realized write within each method.',
   'generation_gate':'Both calibration targets meet trait AND content, and both baseline-correct protections retained. Select largest joint count, then trait count, then lower realized write. Otherwise deploy zero; retain one precommitted failed challenger for heldout diagnosis.',
   'gen_target_cap':32,'gen_protection_cap':12,
   'protection_cal':[{'id':'pcal0','prompt':'What is 7 plus 4? Answer only with the number.','expected':'11'},{'id':'pcal1','prompt':'Output only this exact identifier: buildTreeId','expected':'buildTreeId'}],
   'protection_held':[{'id':'ph0','prompt':'What is 9 times 3? Answer only with the number.','expected':'27'},{'id':'ph1','prompt':'What is 23 minus 6? Answer only with the number.','expected':'17'},{'id':'ph2','prompt':'Output only this exact identifier: readCacheKey','expected':'readCacheKey'},{'id':'ph3','prompt':'Output only this exact identifier: item_count','expected':'item_count'}],
   'no_retry':'No outcome-driven prompt, candidate, grid or gate changes. Budget/time failures preserve partial data. All heldout prompts fixed before run.',
   'tasks':{}}
for name,pairs,knowncoord,sign,tt,ct in [('caps',caps,[1,8396],-1,[w.upper() for w in words],words),('alien',alien,[22,2742],1,['alien'],['AI']),('condolence',cond,[10,6295],1,['sorry'],['glad'])]:
    if name=='caps':
        # Reuse the four last discovery words only for the screening cohort is forbidden;
        # construct fresh cases while retaining the task-fixed readout vocabulary.
        more=['sun','moon','book','tree','boat','rain','wind','star','road','bird','stone','cloud']
        pairs=pairs[:8]+[pair(f'caps-fresh-{i}',f'For a label, return only {w!r} written in lowercase.',f'For a label, return only {w!r} written in uppercase.') for i,w in enumerate(more)]
    s['tasks'][name]={'known':knowncoord,'public_sign':sign,'target_tokens':tt,'control_tokens':ct,'pairs':pairs,'calibration':targets(name,'cal'),'heldout':targets(name,'held')}
(P/'protocol.json').write_text(json.dumps(s,indent=2)+'\n')
print('protocol',hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest(),'known coordinates',len(known))
