"""Create checksum-verified result parts for analysis on another machine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def export(out=None,allow_partial=False,part_mb=100):
    from types import SimpleNamespace
    args=SimpleNamespace(allow_partial=allow_partial,part_mb=part_mb)
    out=Path(out) if out is not None else Path(os.environ.get('INITSKILL_RESULT_DIR',str(ROOT/'results_id_ood')))
    import psutil
    execution=out/'campaign/execution.json'
    if execution.exists():
        record=json.loads(execution.read_text())
        try:process=psutil.Process(record['pid'])
        except psutil.NoSuchProcess:process=None
        if process is not None and abs(process.create_time()-record['process_created'])<.001:
            raise RuntimeError('Stop this scope before exporting results')
    if args.part_mb<1:raise ValueError('Positive part size required')
    ready=out/'reports/FINAL_REPORT_READY.json'
    if not args.allow_partial and not ready.exists():raise SystemExit('Full result not verified. Stop the run and use --allow-partial to return diagnostics.')
    profile=json.loads((ROOT/'.runtime/profile.json').read_text(encoding='utf-8'))
    command=[profile['main_python'],str(ROOT/'scripts/analyze.py'),'--out',str(out)]
    if args.allow_partial:command.append('--allow-partial')
    subprocess.run(command,check=True,env=dict(os.environ,PYTHONUTF8='1'))
    label=time.strftime('%Y%m%d_%H%M%S');dest=ROOT/'returns'/label;dest.mkdir(parents=True,exist_ok=False)
    files=[]
    for folder in [out.name,'verification']:
        for path in (ROOT/folder).rglob('*'):
            if not path.is_file():continue
            relative=path.relative_to(ROOT)
            if any(name in relative.parts for name in ['filter_cache','inputs','oc_checkpoint_slots']):continue
            if path.name.startswith('state_') and path.name!='state_final.npz':continue
            if path.name in {'checkpoint.npz'}:continue
            files.append(path)
    # Record versions and configuration, never pip credentials or API settings.
    for name in ['profile.json','main_environment.json','jax_environment.json']:
        p=ROOT/'.runtime'/name
        if p.exists():files.append(p)
    for p in [ROOT/'data/BUNDLE_MANIFEST.json',ROOT/'data/META64_uniform_1000_100_300_v3/SAMPLING.json']:
        files.append(p)
    records={};archive=dest/'results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as stream:
        for path in sorted(files):
            before=path.stat();digest=sha(path);name=path.relative_to(ROOT).as_posix()
            stream.write(path,name)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError('Result changed during export; stop computation first: '+name)
            records[name]=dict(bytes=after.st_size,sha256=digest)
        payload=json.dumps(dict(schema='initskill-result-return-v1',partial=args.allow_partial,
            created_unix=time.time(),files=records,note='No source inputs or environments are required in this return; use the same GitHub folder/source manifests.'),ensure_ascii=False,indent=2).encode('utf-8')
        stream.writestr('RETURN_FILES.json',payload)
    archive_sha=sha(archive);limit=args.part_mb*1024**2;parts=[]
    if archive.stat().st_size>limit:
        with archive.open('rb') as source:
            for i in range(100000):
                content=source.read(limit)
                if not content:break
                path=dest/f'results.zip.part{i:03d}';path.write_bytes(content)
                parts.append(dict(path=path.name,bytes=len(content),sha256=sha(path)))
        archive.unlink()
    else:parts=[dict(path=archive.name,bytes=archive.stat().st_size,sha256=archive_sha)]
    (dest/'RETURN_MANIFEST.json').write_text(json.dumps(dict(schema='initskill-return-parts-v1',partial=args.allow_partial,
        archive_sha256=archive_sha,parts=parts,files=len(records),instructions='Return this entire directory including every part and this manifest.'),indent=2)+'\n',encoding='utf-8')
    print('Return this folder:',dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--part-mb',type=int,default=100)
    args=parser.parse_args();export(args.out,args.allow_partial,args.part_mb)
