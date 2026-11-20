#!/usr/bin/env python3
"""Fetch only required archive shards and verify exact original array hashes."""
import argparse,hashlib,json,os,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE_URL='https://raw.githubusercontent.com/hsksfje/topo/main/META64_datasets_v1/'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 identity=json.loads((ROOT/'DATASET_IDENTITY.json').read_text(encoding='utf-8'));assert sha(ROOT/'catalog.json')==identity['catalog_sha256'];catalog=json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'));p=argparse.ArgumentParser();p.add_argument('--subset',choices=sorted(catalog['subsets']),default='train100_new100');p.add_argument('--all',action='store_true');p.add_argument('--root',type=Path,default=ROOT/'extracted');p.add_argument('--verify-only',action='store_true');p.add_argument('--base-url',default=BASE_URL);a=p.parse_args();dest=a.root.resolve();dest.mkdir(parents=True,exist_ok=True)
 names=sorted(catalog['subsets']) if a.all else [a.subset];required={}
 for name in names:
  subset=catalog['subsets'][name];path=ROOT/subset['manifest'];assert sha(path)==subset['sha256']
  for line in path.open():
   if not line.strip():continue
   for asset in json.loads(line)['assets'].values():required[asset['path']]=asset
 missing=[name for name,x in required.items() if not (dest/name).exists() or sha(dest/name)!=x['sha256']]
 shards=sorted({catalog['assets'][name]['shard'] for name in missing})
 if a.verify_only:
  assert not missing,f'{len(missing)} missing or changed arrays';print('VERIFIED',len(required),'arrays',flush=True);return
 for i,name in enumerate(shards,1):
  shard=catalog['shards'][name];archive=ROOT/shard['path'];archive.parent.mkdir(exist_ok=True)
  if not archive.exists() or sha(archive)!=shard['sha256']:
   url=a.base_url.rstrip('/')+'/'+shard['path'];tmp=archive.with_suffix('.zip.partial');print('DOWNLOAD',i,len(shards),url,flush=True)
   with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as w:
    while True:
     block=r.read(1024*1024)
     if not block:break
     w.write(block)
   assert sha(tmp)==shard['sha256'],'Download checksum failed';tmp.replace(archive)
  assert archive.stat().st_size==shard['bytes']
  with zipfile.ZipFile(archive) as z:
   for info in z.infolist():
    if info.is_dir():continue
    rel=Path(info.filename);assert not rel.is_absolute() and '..' not in rel.parts
    asset=catalog['assets'].get(rel.as_posix());assert asset is not None and asset['shard']==name
    data=z.read(info);assert len(data)==asset['bytes'] and hashlib.sha256(data).hexdigest()==asset['sha256']
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or sha(target)!=asset['sha256']:
     tmp=target.with_name(target.name+'.partial');tmp.write_bytes(data);tmp.replace(target)
  print('EXTRACTED',name,flush=True)
 for name,x in required.items():assert sha(dest/name)==x['sha256'],name
 print('READY',','.join(names),'arrays',len(required),'root',dest,flush=True)
if __name__=='__main__':main()
