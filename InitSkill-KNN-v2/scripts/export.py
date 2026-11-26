from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from scripts.experiment import ROOT,OUT,summarize
from initskill.methods.benchmark_init_v1.common import sha,dump


def export(partial=False):
    # Freeze the snapshot before hashing; an active writer could otherwise
    # make the listed SHA differ from the bytes copied into the ZIP.
    import psutil
    execution=OUT/'campaign/execution.json'
    if execution.exists():
        record=json.loads(execution.read_text())
        try:process=psutil.Process(record['pid'])
        except psutil.NoSuchProcess:process=None
        if process is not None and abs(process.create_time()-record['created'])<.001:
            raise RuntimeError('Run is active. Stop it with python run.py stop before exporting a partial snapshot.')
    summarize(strict=not partial)
    folder=ROOT/'returns'/time.strftime('%Y%m%d_%H%M%S');folder.mkdir(parents=True)
    files={}
    candidates=list(OUT.rglob('*'))+list((ROOT/'verification').glob('*.json'))
    profile=ROOT/'.runtime/profile.json'
    if profile.exists():candidates.append(profile)
    for p in sorted(candidates):
        if not p.is_file() or any(k in p.parts for k in ['inputs','oc_checkpoint_slots']) or p.suffix in {'.lock','.tmp'}:continue
        files[p.relative_to(ROOT).as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
    with tempfile.TemporaryDirectory(prefix='knn-return-') as temp:
        archive=Path(temp)/'results.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name in files:z.write(ROOT/name,name)
            z.writestr('RETURN_FILES.json',json.dumps(dict(files=files),indent=2))
        parts=[]
        with archive.open('rb') as source:
            i=0
            while True:
                b=source.read(100*1024*1024)
                if not b:break
                p=folder/f'results.zip.part{i:03d}';p.write_bytes(b);parts.append(dict(path=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));i+=1
        dump(folder/'RETURN_MANIFEST.json',dict(schema='initskill-return-parts-v1',benchmark='knn-v2-independent',partial=partial,archive_sha256=sha(archive),parts=parts,files=len(files),instructions='Return this whole directory with all parts and RETURN_MANIFEST.json'))
    print(str(folder))
