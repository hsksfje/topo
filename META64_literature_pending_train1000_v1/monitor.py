"""Read-only progress viewer; Ctrl+C never sends pause to the experiment."""
import argparse
import json
from pathlib import Path
import time

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs/train1000_literature_pending_run01')
    p.add_argument('--watch',action='store_true')
    a=p.parse_args()
    try:
        while True:
            path=a.output/'progress.json'
            if path.exists():
                r=json.loads(path.read_text(encoding='utf-8'))
                print(time.strftime('%Y-%m-%d %H:%M:%S'),r['status'],
                      f"OC50={r['complete50']}/{r['planned']}",f"converged={r['converged']}",
                      f"max_steps={r['max_steps']}",f"failed={r['failed']}",
                      f"report_age_s={time.time()-r['updated_unix']:.0f}",flush=True)
                for x in r.get('recent_progress',[])[:5]:
                    print(' ',x['task_id'],x['method'],'step',x['step'],'PID',x['pid'])
            else:
                print('Not started:',a.output,flush=True)
            if not a.watch:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass
