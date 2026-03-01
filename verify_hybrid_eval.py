#!/usr/bin/env python3
"""Verify all eval results (hybrid + opus + ds)."""
import json, os

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"

DIRS = {
    "Hybrid Cross": ("hybrid_cross_results", "json"),
    "Hybrid Synth": ("hybrid_synth_results", "py"),
    "Hybrid Direct": ("hybrid_direct_results", "json"),
    "Opus Synth": ("opus_eval_synth", "py"),
    "Opus Direct": ("opus_eval_direct", "json"),
    "DS Synth": ("ds_synth_results", "py"),
    "DS Direct": ("ds_direct_results", "json"),
}

all_passed = set()

for label, (dirname, ext) in DIRS.items():
    dirpath = os.path.expanduser(f"~/verantyx_v6/{dirname}")
    if not os.path.isdir(dirpath): continue
    files = [f for f in os.listdir(dirpath) if f.endswith(f'.{ext}')]
    if not files: continue
    
    passed, failed = [], []
    for f in sorted(files):
        tid = f.replace(f'.{ext}', '')
        task_path = os.path.join(EVAL_DIR, f"{tid}.json")
        if not os.path.exists(task_path): continue
        with open(task_path) as tf: task = json.load(tf)
        
        try:
            if ext == "py":
                ns = {}
                exec(open(os.path.join(dirpath, f)).read(), ns)
                transform = ns['transform']
                ok = all(transform(t['input']) == t['output'] for t in task['test'])
            else:
                with open(os.path.join(dirpath, f)) as pf: pred = json.load(pf)
                ok = True
                for i, t in enumerate(task['test']):
                    if i < len(pred.get('test',[])):
                        if pred['test'][i].get('output') != t['output']:
                            ok = False
                    else:
                        ok = False
            if ok: passed.append(tid)
            else: failed.append(tid)
        except Exception as e:
            failed.append(tid)
    
    if passed or failed:
        print(f"\n{label}: {len(passed)} PASS / {len(passed)+len(failed)} total")
        for t in passed: print(f"  ✅ {t}")
        all_passed.update(passed)

print(f"\n{'='*50}")
print(f"Combined unique PASS: {len(all_passed)}/120 ({100*len(all_passed)/120:.1f}%)")
for t in sorted(all_passed): print(f"  {t}")
