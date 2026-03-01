#!/usr/bin/env python3
"""Run solve_one.py as subprocess for each task. Prevents memory leaks."""
import json, os, sys, subprocess, time

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/verantyx_cwm_results")
os.makedirs(RESULT_DIR, exist_ok=True)

task_ids = sorted([f.replace('.json','') for f in os.listdir(EVAL_DIR) if f.endswith('.json')])

print(f"Verantyx CWM — {len(task_ids)} tasks", flush=True)
stats = {"solved": 0, "fail": 0, "error": 0, "skip": 0}
t_start = time.time()

for tid in task_ids:
    result_path = os.path.join(RESULT_DIR, f"{tid}.json")
    if os.path.exists(result_path):
        stats["skip"] += 1
        continue
    
    task_path = os.path.join(EVAL_DIR, f"{tid}.json")
    try:
        proc = subprocess.run(
            [sys.executable, "solve_one.py", task_path, "40"],
            capture_output=True, text=True, timeout=50,
            cwd=os.path.expanduser("~/verantyx_v6")
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            print(f"[{tid}] ❌ error", flush=True)
            stats["error"] += 1
            continue
        
        result = json.loads(proc.stdout.strip())
        
        if result.get("status") == "solved":
            with open(result_path, 'w') as f:
                json.dump(result, f)
            print(f"[{tid}] ✅ {result.get('piece','')} | {result.get('method','')} | {result.get('n_solutions',0)} sol | {result.get('elapsed',0)}s", flush=True)
            stats["solved"] += 1
        else:
            print(f"[{tid}] ❌ | {result.get('elapsed',0)}s", flush=True)
            stats["fail"] += 1
    
    except subprocess.TimeoutExpired:
        print(f"[{tid}] ❌ timeout", flush=True)
        stats["error"] += 1
    except Exception as e:
        print(f"[{tid}] ❌ {e}", flush=True)
        stats["error"] += 1

elapsed = time.time() - t_start
print(f"\n{'='*60}", flush=True)
print(f"Solved: {stats['solved']}, Failed: {stats['fail']}, Errors: {stats['error']}, Skipped: {stats['skip']}", flush=True)
print(f"Time: {elapsed:.0f}s", flush=True)

# Verify
passed = []
attempted = 0
for f in sorted(os.listdir(RESULT_DIR)):
    if not f.endswith('.json'): continue
    tid = f.replace('.json','')
    tp = os.path.join(EVAL_DIR, f"{tid}.json")
    if not os.path.exists(tp): continue
    with open(tp) as tf: task = json.load(tf)
    with open(os.path.join(RESULT_DIR, f)) as rf: result = json.load(rf)
    attempted += 1
    from arc.grid import grid_eq
    ok = all(
        i < len(result.get('test',[])) and result['test'][i].get('output') == t['output']
        for i, t in enumerate(task['test'])
    )
    if ok:
        passed.append(tid)
        print(f"  ✅ {tid} | {result.get('piece','?')}", flush=True)

print(f"\nScore: {len(passed)}/{len(task_ids)} ({100*len(passed)/len(task_ids):.1f}%)", flush=True)
