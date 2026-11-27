"""Export all new trajectories and paid-generation receipts, no credentials."""
import hashlib
import json
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def export(out):
    assert (out/'config.json').exists(),'Run output not found'
    export_dir=ROOT/'exports';export_dir.mkdir(exist_ok=True)
    target=export_dir/(out.name+'_return_'+time.strftime('%Y%m%d_%H%M%S')+'.zip')
    hashes={}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for base,prefix in [(out,'run01'),(ROOT/'generated','generated')]:
            if not base.exists():continue
            for path in sorted(base.rglob('*')):
                if not path.is_file() or path.name.startswith('.') or path.suffix in ('.lock','.pyc'):continue
                name=prefix+'/'+path.relative_to(base).as_posix()
                value=path.read_bytes()
                hashes[name]=hashlib.sha256(value).hexdigest()
                z.writestr(name,value)
        z.write(ROOT/'plan.json','provenance/plan.json')
        z.write(ROOT/'PAYLOAD_HASHES.json','provenance/PAYLOAD_HASHES.json')
        z.writestr('RETURN_MANIFEST.json',json.dumps(dict(schema='retained23-backfill-return-v1',files=hashes,run_config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),scope='Only this backfill batch; new Flash proposals remain separately labelled'),ensure_ascii=False,indent=2))
    digest=hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    target.with_suffix('.zip.sha256').write_text(digest.hexdigest()+'  '+target.name+'\n')
    print('COPY THESE TWO FILES BACK:',target,target.with_suffix('.zip.sha256'),sep='\n')
