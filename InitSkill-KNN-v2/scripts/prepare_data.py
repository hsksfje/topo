"""Extract and verify the bundled inputs/labels; no network or test labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from initskill.locking import atomic_replace


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prepare():
    manifest=json.loads((ROOT/'data/BUNDLE_MANIFEST.json').read_text())
    parts=[ROOT/r['path'] for r in manifest['parts']]
    for record,path in zip(manifest['parts'],parts):
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise ValueError('Compressed data bundle changed: '+record['path'])
    incomplete=[name for name,r in manifest['files'].items()
                if not (ROOT/name).is_file() or (ROOT/name).stat().st_size!=r['bytes']]
    with tempfile.TemporaryDirectory(prefix='initskill-data-') as temp:
        if incomplete:
            archive=parts[0]
            if len(parts)>1:
                archive=Path(temp)/'bundle.zip'
                with archive.open('wb') as stream:
                    for part in parts:
                        with part.open('rb') as source:shutil.copyfileobj(source,stream)
            with zipfile.ZipFile(archive) as zipped:
                if set(zipped.namelist())!=set(manifest['files']):raise ValueError('Unexpected bundle members')
                for name in incomplete:
                    target=ROOT/name
                    if not target.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Unsafe archive member')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_name(target.name+'.extract.tmp')
                    with zipped.open(name) as source,temporary.open('wb') as stream:shutil.copyfileobj(source,stream)
                    if sha(temporary)!=manifest['files'][name]['sha256']:raise ValueError('Corrupt extracted data: '+name)
                    atomic_replace(temporary,target)
    for name,record in manifest['files'].items():
        if sha(ROOT/name)!=record['sha256']:raise ValueError('Changed extracted data: '+name)
    result=dict(passed=True,files_verified=len(manifest['files']),counts=manifest['counts'],
                test_targets_included=False,bundle_manifest_sha256=sha(ROOT/'data/BUNDLE_MANIFEST.json'))
    (ROOT/'data/PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return result


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();prepare()
