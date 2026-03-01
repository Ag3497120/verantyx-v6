#!/usr/bin/env python3
"""Verify full hybrid results."""
import json, os

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/full_hybrid_results")

passed, failed, errors = [], [], []
methods = {"cross_engine": 0, "deepseek_synth": 0, "direct": 0}

for f in sorted(os.listdir(RESULT_DIR)):
    if not f.endswith('.json'): continue
    tid = f.replace('.json', '')
    task_path = os.path.join(EVAL_DIR, f"{tid}.json")
    if not os.path.exists(task_path): continue
    
    with open(task_path) as tf: task = json.load(tf)
    with open(os.path.join(RESULT_DIR, f)) as rf: result = json.load(rf)
    
    method = result.get("method", "unknown")
    
    # Check if .py exists (synth code) — verify via code
    py_path = os.path.join(RESULT_DIR, f"{tid}.py")
    if os.path.exists(py_path):
        try:
            ns = {}
            exec(open(py_path).read(), ns)
            transform = ns['transform']
            ok = all(transform(t['input']) == t['output'] for t in task['test'])
        except Exception as e:
            ok = False
    else:
        # Verify from JSON prediction
        ok = True
        for i, t in enumerate(task['test']):
            if i < len(result.get('test', [])):
                if result['test'][i].get('output') != t['output']:
                    ok = False
            else:
                ok = False
    
    if ok:
        passed.append((tid, method))
        methods[method] = methods.get(method, 0) + 1
    else:
        failed.append((tid, method))

total = len(passed) + len(failed)
print(f"Full Hybrid (DeepSeek V3 × Verantyx)")
print(f"{'='*50}")
print(f"PASS: {len(passed)}/{total} ({100*len(passed)/total:.1f}%)" if total else "No results")
print()
print("By method:")
for m, n in methods.items():
    if n: print(f"  {m}: {n} PASS")
print()
print("Passed tasks:")
for tid, method in passed:
    print(f"  ✅ {tid} ({method})")
print()
print(f"Failed: {len(failed)}/{total}")
