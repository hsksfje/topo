#!/usr/bin/env python3
"""Seal completed new-task outputs for whole-directory return; no FE/API."""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def seal_output(out):
 cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));progress=json.loads((out/'convergence/progress.json').read_text(encoding='utf-8'))
 assert (out/'COMPLETION.json').exists() and progress['status']=='complete' and progress['settled']==len(cfg['records'])*38
 excluded={'.run.lock','return_manifest.json','RETURN_READY.json'}
 files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in excluded}
 manifest=dict(schema='META64-fresh-return-v1',tasks=len(cfg['records']),methods=38,task_ids=[r['task_id'] for r in cfg['records']],files=files,package_sha256=cfg['package_sha256'],progress=progress,created_unix=time.time())
 (out/'return_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n', encoding='utf-8');(out/'RETURN_READY.json').write_text(json.dumps(dict(manifest_sha256=sha(out/'return_manifest.json'),ready_to_copy=True))+'\n', encoding='utf-8')
 return manifest
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/train100_new100_run01');p.add_argument('--zip',type=Path);a=p.parse_args();out=a.output.resolve()
 # run imports the vendored IO/locking helpers without invoking any API.
 from run import cont
 with (out/'.run.lock').open('a+b') as lock:
  cont.acquire_lock(lock);manifest=seal_output(out)
  if a.zip:
   dest=a.zip.resolve();assert out not in dest.parents;dest.parent.mkdir(parents=True,exist_ok=True)
   with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for name,h in manifest['files'].items():assert sha(out/name)==h;z.write(out/name,'train100_new100_run01/'+name)
    for name in ('return_manifest.json','RETURN_READY.json'):z.write(out/name,'train100_new100_run01/'+name)
   dest.with_name(dest.name+'.sha256').write_text(sha(dest)+'  '+dest.name+'\n', encoding='utf-8');print('EXPORTED',dest)
 print('READY_TO_COPY',out)
if __name__=='__main__':main()
