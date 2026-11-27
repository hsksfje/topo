#!/usr/bin/env python3
"""Explicit, audited API-only package migration; does not run FE or send API requests."""
import argparse,sys,json,hashlib,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/vendor')]
from initskill.io import dump,sha
import portable_lock

def upgrade(out):
 out=Path(out).resolve()
 with (out/'.run.lock').open('a+b') as lock:
  portable_lock.flock(lock,portable_lock.LOCK_EX|portable_lock.LOCK_NB)
  import run
  run.verify_code()
  cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
  assert sha(out/'config.json')==json.loads((out/'config_identity.json').read_text(encoding='utf-8'))['sha256']
  newhash=sha(ROOT/'PACKAGE_MANIFEST.json')
  if cfg['package_sha256']==newhash:print('ALREADY_UPGRADED',out);return
  backups=ROOT/'validation/api_retry_upgrade_backup';parents=list(backups.glob('*PACKAGE_MANIFEST.json.original'))
  source=next((p for p in parents if sha(p)==cfg['package_sha256']),None)
  assert source is not None,'Unsupported source package; inspect before migration'
  old=json.loads(source.read_text(encoding='utf-8'))['files'];new=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))['files']
  changed=[name for name,h in old.items() if new.get(name)!=h]
  assert set(changed)<= {'code/api_client.py','README.md'},changed
  # Solver, tasks, prompts, protocol, authorizations and hard budget are frozen.
  dest=out/'migrations/api_retry_v1';dest.mkdir(parents=True,exist_ok=False)
  for name in ('config.json','config_identity.json'):shutil.copy2(out/name,dest/name)
  dump(dest/'audit.json',dict(old_package_sha256=cfg['package_sha256'],new_package_sha256=newhash,
       old_config_sha256=sha(out/'config.json'),changed_existing_package_files=changed,
       unchanged_experiment=True,max_api_calls=cfg['max_api_calls'],unix=time.time()))
  cfg['package_sha256']=newhash
  dump(out/'config.json',cfg);dump(out/'config_identity.json',dict(sha256=sha(out/'config.json')))
  print('API_ONLY_UPGRADE_COMPLETE',out,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);upgrade(p.parse_args().output)
