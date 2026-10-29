import json,hashlib
from pathlib import Path
p=Path(__file__).parent
main=json.loads((p.parent/'pp-window-v1/protocol.json').read_text())
sentences=['purple flowers grew behind the gate.','a wooden bridge crossed the stream.','the brass key lay beneath the cushion.','four bright shells lined the window.','a gentle wind shook the apple tree.','the baker placed bread near the oven.','blue ribbons hung from the small basket.','a calm pony stood beside the barn.','warm light entered through the doorway.','the tall grass waved along the trail.']
items=[{'id':f'genaudit-{i}','prompt':f'Output only this text with no introduction: {v}','expected':v} for i,v in enumerate(sentences)]
s={k:main[k] for k in ['model_path','model_revision','calibration_text','residual_doses','scale_factors','clamp_alphas']}
s.update({'version':'generation-first-caps-audit-v1','reason':'Main first-token funnel selected the public CAPS sign opposite the previously validated sign. Test the entire pre-existing grid with generation, new calibration prompts and new confirmation prompts; do not overwrite main results.','seed':2026102902,'max_forward_calls':10000,'max_seconds':3600,'source_captures':'/tmp/qwen-pp-window-20261029-BrnUlmBs/results/caps/captures.pt','coordinates':{'public':[1,8396],'signed_and_rms_winner':[30,271],'random_signed_winner':[31,1699]},'calibration':items[:2],'heldout':items[2:],'target_cap':24,'protection_cap':12,
 'protection_cal':[{'id':'gc0','prompt':'What is 6 plus 7? Answer only with the number.','expected':'13'},{'id':'gc1','prompt':'Output only this exact identifier: fetchNodeId','expected':'fetchNodeId'}],
 'protection_held':[{'id':'gh0','prompt':'What is 7 times 6? Answer only with the number.','expected':'42'},{'id':'gh1','prompt':'What is 31 minus 8? Answer only with the number.','expected':'23'},{'id':'gh2','prompt':'Output only this exact identifier: saveModelKey','expected':'saveModelKey'},{'id':'gh3','prompt':'Output only this exact identifier: row_offset','expected':'row_offset'}],
 'selection':'Generate at every recipe. Require both case-insensitive exact copies and at least .85 uppercase on both, with both baseline-correct protections retained. Prefer higher mean uppercase fraction then lower mean realized write norm. If none pass deploy zero; commit one failed challenger to diagnose on heldout.','fixed_reference_doses':[-.2,.2,-1.], 'no_retune':'No changes to candidates, prompts, recipes or gates after starting this secondary run.'})
s['source_captures_sha256']='ddb9727827f4e29f5da9cc54e216f04556989acf54cbfb4c13ee9fa51b7393d2'
(p/'protocol.json').write_text(json.dumps(s,indent=2)+'\n')
print(hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest())
