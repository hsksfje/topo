"""Verify/extract returned result parts into this matching benchmark folder."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for content in iter(lambda:stream.read(1024*1024),b''):h.update(content)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);args=parser.parse_args()
    manifest=json.loads((args.folder/'RETURN_MANIFEST.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='initskill-return-') as temp:
        archive=Path(temp)/'results.zip'
        with archive.open('wb') as stream:
            for record in manifest['parts']:
                p=args.folder/record['path']
                if p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Changed return part: '+str(p))
                with p.open('rb') as source:shutil.copyfileobj(source,stream)
        if sha(archive)!=manifest['archive_sha256']:raise ValueError('Changed assembled archive')
        with zipfile.ZipFile(archive) as stream:
            records=json.loads(stream.read('RETURN_FILES.json'))['files']
            if set(stream.namelist())!=set(records)|{'RETURN_FILES.json'}:raise ValueError('Unexpected return members')
            for name,record in records.items():
                relative=Path(name)
                # The source device's runtime/profile paths are provenance,
                # not this analysis device's executable configuration.
                target=ROOT/relative if relative.parts[0]!='.runtime' else ROOT/'results/returned_runtime'/relative.name
                if not target.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Unsafe return path')
                if target.exists():
                    if sha(target)!=record['sha256']:raise FileExistsError('Refuse to overwrite a different local artifact: '+str(target))
                    continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with stream.open(name) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest)
                if sha(target)!=record['sha256']:raise ValueError('Changed extracted result: '+name)
    print('Verified return imported. Run: python benchmark.py analyze')


if __name__=='__main__':main()
