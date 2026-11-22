"""Seal or zip completed results for returning to the first device; no FE/API."""
import argparse, json, time, zipfile
from pathlib import Path
from storage import dump, sha
ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.run.lock', 'return_manifest.json', 'RETURN_READY.json'}

def seal_output(out):
    cfg = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    progress = json.loads((out / 'progress.json').read_text(encoding='utf-8'))
    assert (out / 'COMPLETION.json').exists() and progress['status'] == 'complete', 'Experiment is not complete'
    assert progress['settled'] == len(cfg['trajectories'])
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in EXCLUDED}
    manifest = dict(schema='META64-return-results-v1', tasks=len(cfg['records']), methods=len(cfg['methods']), task_ids=[r['task_id'] for r in cfg['records']], package_manifest_sha256=cfg['package_manifest_sha256'], files=files, status=progress, created_unix=time.time())
    dump(out / 'return_manifest.json', manifest)
    dump(out / 'RETURN_READY.json', dict(manifest_sha256=sha(out / 'return_manifest.json'), ready_to_copy=True, unix=time.time()))
    return manifest

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/delegated6600_run01')
    p.add_argument('--zip', type=Path, default=None)
    a = p.parse_args()
    out = a.output.resolve()
    from run import acquire_lock
    with (out / '.run.lock').open('a+b') as lock:
        acquire_lock(lock)
        manifest = seal_output(out)
        if a.zip:
            dest = a.zip.resolve()
            assert out not in dest.parents, 'Put the archive outside the output folder'
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
                for name, digest in manifest['files'].items():
                    assert sha(out / name) == digest, name
                    z.write(out / name, arcname=out.name + '/' + name)
                for name in ('return_manifest.json', 'RETURN_READY.json'):
                    z.write(out / name, arcname=out.name + '/' + name)
            dest.with_name(dest.name + '.sha256').write_text(sha(dest) + '  ' + dest.name + '\n', encoding='utf-8')
            print('RETURN_ARCHIVE', dest, flush=True)
    print('READY_TO_COPY', out, 'tasks', manifest['tasks'], flush=True)
if __name__ == '__main__':
    main()
