"""Stop only the recorded, creation-time-verified benchmark process tree."""
import json
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[1]


def stop_descendants(parent):
    children=parent.children(recursive=True)
    for process in reversed(children):
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(children,timeout=10)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass


def stop(out=None):
    out=ROOT/'results_id_ood' if out is None else Path(out)
    path=out/'campaign/execution.json'
    if not path.exists():print('No recorded benchmark process.');return
    record=json.loads(path.read_text(encoding='utf-8'))
    try:parent=psutil.Process(record['pid'])
    except psutil.NoSuchProcess:print('Recorded process has already exited.');return
    if abs(parent.create_time()-record['process_created'])>1e-3:
        raise RuntimeError('PID has been reused; refusing to stop a different process.')
    if Path(record['root']).resolve()!=ROOT.resolve():raise RuntimeError('Different benchmark directory.')
    stop_descendants(parent)
    try:parent.terminate();parent.wait(timeout=10)
    except psutil.NoSuchProcess:pass
    print('Stopped this benchmark only. Checkpoints/FE ledgers retained; use start to resume.')


if __name__=='__main__':stop()
