"""Create skip markers so the LLM stage re-runs ONLY the pro tasks with flash.

Usage (after `python run_deviceA300.py --stage prepare`):
    python setup_skip.py <output_dir>
    # e.g. python setup_skip.py outputs/deviceA300_run01
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/deviceA300_run01')
    out.mkdir(parents=True, exist_ok=True)

    # 1) skip the fixed stage (fixed results are model-independent, already done)
    (out / 'fixed_complete.json').write_text(json.dumps({'unix': 0}) + '\n', encoding='utf-8')

    # 2) skip pure-flash LLM tasks (already produced with flash, no need to re-run)
    skip = json.loads((HERE / 'skip_flash_tasks.json').read_text(encoding='utf-8'))['task_ids']
    for t in skip:
        p = out / 'tasks' / t / 'complete.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'task_id': t, 'skip': True}) + '\n', encoding='utf-8')

    print('created fixed_complete.json +', len(skip), 'flash-task complete.json markers')


if __name__ == '__main__':
    main()
