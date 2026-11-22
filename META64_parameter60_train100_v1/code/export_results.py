"""Seal/verify/zip results without FE or model calls."""
import argparse
import time
import zipfile
from pathlib import Path

from experiment import read
from storage import dump, sha

EXCLUDED={'.run.lock','return_manifest.json','RETURN_READY.json','PAUSE_REQUESTED'}


def seal_output(out):
    out=Path(out)
    cfg=read(out/'config.json')
    progress=read(out/'progress.json')
    expected={(r['task_id'],p['id']) for r in cfg['records'] for p in cfg['profiles']}
    assert (out/'COMPLETION.json').exists() and progress['status']=='complete'
    assert progress['settled']==len(expected)
    for tid,pid in expected:
        row=read(out/'tasks'/tid/pid/'result.json')
        assert (row['task_id'],row['profile'])==(tid,pid)
        assert row['status'] in ('converged','max_steps','failed')
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
           if p.is_file() and p.name not in EXCLUDED}
    manifest=dict(schema='META64-parameter60-return-v1',package_sha256=cfg['package_sha256'],
        config_sha256=sha(out/'config.json'),trajectory_count=len(expected),
        task_ids=[r['task_id'] for r in cfg['records']],profiles=[p['id'] for p in cfg['profiles']],
        validation_only=cfg.get('validation_only',False),files=files,created_unix=time.time())
    dump(out/'return_manifest.json',manifest)
    dump(out/'RETURN_READY.json',dict(manifest_sha256=sha(out/'return_manifest.json'),ready_to_copy=True))
    return manifest


def verify_return(out):
    out=Path(out)
    ready=read(out/'RETURN_READY.json')
    assert sha(out/'return_manifest.json')==ready['manifest_sha256'],'Return manifest changed'
    manifest=read(out/'return_manifest.json')
    for rel,value in manifest['files'].items():
        assert sha(out/rel)==value,'Return file missing/changed: '+rel
    actual={p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file() and p.name not in EXCLUDED}
    assert actual==set(manifest['files']),'Unsealed extra files present'
    cfg=read(out/'config.json')
    assert sha(out/'config.json')==manifest['config_sha256']
    assert cfg['package_sha256']==manifest['package_sha256']
    expected={(r['task_id'],p['id']) for r in cfg['records'] for p in cfg['profiles']}
    assert len(expected)==manifest['trajectory_count']
    actual_results={(read(out/rel)['task_id'],read(out/rel)['profile']) for rel in manifest['files']
                    if rel.startswith('tasks/') and rel.endswith('/result.json')}
    assert actual_results==expected
    return manifest


def export_zip(out, destination=None):
    out=Path(out).resolve()
    manifest=verify_return(out) if (out/'RETURN_READY.json').exists() else seal_output(out)
    dest=Path(destination).resolve() if destination else out.with_name(out.name+'_return.zip')
    if dest==out or out in dest.parents:
        raise ValueError('Put the return archive outside its output directory')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for rel,value in manifest['files'].items():
            assert sha(out/rel)==value
            z.write(out/rel,arcname=out.name+'/'+rel)
        for name in ('return_manifest.json','RETURN_READY.json'):
            z.write(out/name,arcname=out.name+'/'+name)
    dest.with_name(dest.name+'.sha256').write_text(sha(dest)+'  '+dest.name+'\n',encoding='utf-8')
    return dest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--zip',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    from run import output_lock
    with output_lock(args.output):
        if args.verify_only:
            print(verify_return(args.output)['trajectory_count'],'trajectories verified')
        else:
            print(export_zip(args.output,args.zip))
