#!/usr/bin/env python3
"""Restore immutable OC50 inputs from checked GitHub ZIP shards; no model API."""
import json,hashlib,zipfile,urllib.request,shutil,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf-8'))['files']
 existing=[n for n in manifest if n.startswith('data/')]
 if all((ROOT/n).is_file() and sha(ROOT/n)==manifest[n] for n in existing):print('DATA_ALREADY_VERIFIED',len(existing));return
 catalog=json.loads((ROOT/'DATA_SHARDS.json').read_text(encoding='utf-8'))
 for row in catalog['shards']:
  f=ROOT/row['path'];f.parent.mkdir(parents=True,exist_ok=True)
  if not f.exists() or sha(f)!=row['sha256']:
   part=f.with_suffix('.download');url=catalog['base_url']+'/'+row['path'];print('DOWNLOAD',url,flush=True)
   with urllib.request.urlopen(url,timeout=180) as response,part.open('wb') as out:shutil.copyfileobj(response,out)
   assert sha(part)==row['sha256'],'Shard checksum mismatch';part.replace(f)
  with zipfile.ZipFile(f) as z:
   for item in z.infolist():
    rel=Path(item.filename);assert not rel.is_absolute() and '..' not in rel.parts and item.filename.startswith('data/') and item.filename in manifest
    assert not stat.S_ISLNK(item.external_attr>>16)
    dest=ROOT/rel
    if dest.is_file() and sha(dest)==manifest[item.filename]:continue
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_name('.'+dest.name+'.extract')
    with z.open(item) as source,tmp.open('wb') as target:shutil.copyfileobj(source,target)
    assert sha(tmp)==manifest[item.filename];tmp.replace(dest)
  print('RESTORED',row['path'],flush=True)
 assert all((ROOT/n).is_file() and sha(ROOT/n)==manifest[n] for n in existing)
 print('DATA_VERIFIED',len(existing),flush=True)
if __name__=='__main__':main()
