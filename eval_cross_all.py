#!/usr/bin/env python3
import sys, time, json, os
sys.path.insert(0, '.')
from arc.cross_engine import solve_cross_engine
from arc.grid import grid_eq

EVAL_DIR = '/private/tmp/arc-agi-2/data/evaluation'
tids = sorted(f.replace('.json','') for f in os.listdir(EVAL_DIR))
LOG = open('eval_cross_216.log', 'w')

correct = 0; total = 0; wc_correct = 0

for tid in tids:
    with open(f'{EVAL_DIR}/{tid}.json') as f:
        task = json.load(f)
    tp = [(ex['input'], ex['output']) for ex in task['train']]
    ti = [t['input'] for t in task['test']]
    t0 = time.time()
    try:
        preds, verified = solve_cross_engine(tp, ti)
    except:
        preds, verified = None, []
    elapsed = time.time()-t0
    total += 1
    is_correct = False
    if preds and preds[0]:
        is_correct = all(grid_eq(preds[j][0], task['test'][j]['output']) for j in range(len(task['test'])))
    if is_correct: correct += 1
    names = [getattr(p,'name',type(p).__name__) for _,p in verified]
    has_wc = any(str(n).startswith('wc') for n in names)
    if is_correct and has_wc: wc_correct += 1
    line = f'{total:3d}/{len(tids)} {"✅" if is_correct else "❌" if verified else "  "} {tid} | {len(verified)}p {elapsed:.1f}s{"[WC]" if has_wc else ""} | {names[:2]}'
    print(line, flush=True)
    LOG.write(line + '\n'); LOG.flush()

summary = f'\n=== {correct}/{total} (WC contrib: {wc_correct}) ==='
print(summary, flush=True)
LOG.write(summary + '\n'); LOG.close()
