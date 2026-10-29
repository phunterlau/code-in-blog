import json, shutil, subprocess, tempfile
from pathlib import Path
fixture=Path('/private/tmp/pp-portable-selection-test-onsp5dlr')
base=Path(tempfile.mkdtemp(prefix='pp-portable-render-synthetic-',dir='/private/tmp'))
out=base/'rendered'; out.mkdir()
mirror=base/'mirror'; mirror.mkdir()
shutil.copytree(fixture/'result',mirror/'calibration')
shutil.copytree(fixture/'review',mirror/'review-calibration')
shutil.copy(fixture/'selection.json',mirror/'selection.json')
p=json.loads((mirror/'calibration/protocol.json').read_text())
s=json.loads((mirror/'selection.json').read_text())
(mirror/'heldout').mkdir();(mirror/'review-heldout').mkdir()
rows=[]; grades=[]; mapping={}; summary={}
for cat in p['categories']:
    summary[cat]={}
    for cond in ['baseline','prompt_only','candidate','random']:
        good=cond in ['prompt_only','candidate']
        recipe=s['choices'][cat][cond]['recipe'] if cond in ['candidate','random'] else None
        row={'category':cat,'condition':cond,'recipe':recipe,'target':[],'protection':[]}
        for item in p['heldout']:
            rid=f'{cat}:{cond}:target:{item["id"]}'
            gid=f'H{len(grades)+1:03}'
            mapping[gid]=rid
            grades.append({'id':gid,'style':2 if good else 0,'quality':2,'note':'SYNTHETIC renderer fixture; not a model output.'})
            row['target'].append({'record_id':rid,'item':item['id'],'prompt':item['prompt'],'text':'SYNTHETIC TEST OUTPUT. NO MODEL WAS RUN.','capped':False})
        for item in p['protection_heldout']:
            row['protection'].append({'item':item['id'],'text':item['expected'],'capped':False,'terminated_eos':True,'metrics':{'exact_correct':True}})
        rows.append(row)
        summary[cat][cond]={'recipe':recipe,'calibration_qualified':cond=='candidate' if recipe else None,'qualified_control':cond=='candidate','metrics':{'joint':8 if good else 0,'style_gain':8 if good else 0,'protection_correct':6,'capped':0}}
for path,obj in [('heldout/rows.json',rows),('heldout-analysis.json',{'summary':summary}),('review-heldout/grades.json',grades),('review-heldout/mapping.json',mapping)]:
    (mirror/path).write_text(json.dumps(obj))
for phase in ['calibration','heldout']:
    (mirror/f'{phase}-verification.json').write_text(json.dumps({'passed':True,'completion':{'calls':1000,'pid':99999999}}))
cmd=['/private/tmp/pp-window-plot-20261029/bin/python','/Users/hliu/Documents/Codex/2026-10-28/und/work/pp-portable-styles-analysis/render_results.py','--mirror',str(mirror),'--output',str(out)]
r=subprocess.run(cmd,capture_output=True,text=True)
assert r.returncode==0,r.stderr
files=sorted(str(f.relative_to(out)) for f in out.rglob('*') if f.is_file())
assert len(files)==6,files
print(json.dumps({'status':'passed','synthetic_only':True,'model_calls':0,'temporary_fixture':str(base),'files':files},indent=2))
