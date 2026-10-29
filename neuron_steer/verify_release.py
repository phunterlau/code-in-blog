"""Offline integrity and selected-result checks; no model execution."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parent

def read(path): return json.loads((ROOT/path).read_text())

def verify():
    manifest=read('release-manifest.json')['files']
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='release-manifest.json'}
    assert actual==set(manifest), 'Release file set changed'
    counts={'json':0,'jsonl_records':0,'python':0}
    for rel,expected in manifest.items():
        p=ROOT/rel
        assert not p.is_symlink(),rel
        data=p.read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected,rel
        if p.suffix=='.json': json.loads(data);counts['json']+=1
        if p.suffix=='.jsonl':
            for line in data.decode().splitlines():
                if line.strip(): json.loads(line);counts['jsonl_records']+=1
        if p.suffix=='.py': ast.parse(data,filename=rel);counts['python']+=1
    for row in read('archive-index.json')['files']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    prefix='archive/qwen-control-window/evidence/main-v1/results/'
    generations,facts=read(prefix+'generations.json'),read(prefix+'facts.json')
    for arm,expected in [('baseline',(0,12)),('alien_anchor_g0.25',(0,12)),('alien_anchor_g0.5',(2,12)),('alien_anchor_g1',(1,4))]:
        gs=[r for r in generations if r['phase']=='heldout' and r['condition']==arm]
        fs=[r for r in facts if r['phase']=='heldout' and r['condition']==arm]
        assert len(gs)==4 and len(fs)==12
        assert (sum(r['screens']['alien']['coherent_target'] for r in gs),sum(r['correct'] for r in fs))==expected
    assert read('archive/qwen-pp-discovery-control/analysis.json')['alien']['public_rank']=={'signed':11,'rms':23}
    x=read('archive/qwen-pp-discovery-control/condolence-rescue/analysis.json')['generation']['signed_and_rms_candidate']
    assert (x['joint'],x['n'],x['protection_correct'],x['protection_n'])==(8,8,4,4)
    print(json.dumps({'status':'passed','files':len(actual),'counts':counts,'gpu_calls':0,'result_checks':['persona dose counts','signed/RMS rank','apology pulse']}))

if __name__=='__main__': verify()
