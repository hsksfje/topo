"""Optional paper download from the recorded author/publisher URLs."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'references/init_benchmark_v1'
    sources=json.loads((folder/'SOURCES.json').read_text(encoding='utf-8'))
    for record in sources['sources']:
        if record['kind']!='paper':continue
        target=folder/record['path'];target.parent.mkdir(exist_ok=True)
        data=urllib.request.urlopen(record['url'],timeout=120).read()
        if hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise RuntimeError('Downloaded PDF bytes differ from the reviewed version: '+record['name'])
        target.write_bytes(data);print(record['name'],'verified')


if __name__=='__main__':main()
