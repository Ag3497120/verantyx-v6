#!/usr/bin/env python3
"""Verify DeepSeek eval results against test outputs."""
import json, os

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
SYNTH_DIR = os.path.expanduser("~/verantyx_v6/ds_synth_results")
DIRECT_DIR = os.path.expanduser("~/verantyx_v6/ds_direct_results")
# Also check Opus results
OPUS_SYNTH = os.path.expanduser("~/verantyx_v6/opus_eval_synth")
OPUS_DIRECT = os.path.expanduser("~/verantyx_v6/opus_eval_direct")

def check_synth(synth_dir, label):
    passed, failed, errors = [], [], []
    if not os.path.isdir(synth_dir): return passed, failed, errors
    for f in sorted(os.listdir(synth_dir)):
        if not f.endswith('.py'): continue
        tid = f.replace('.py','')
        task_path = os.path.join(EVAL_DIR, f"{tid}.json")
        if not os.path.exists(task_path): continue
        with open(task_path) as tf: task = json.load(tf)
        try:
            ns = {}
            exec(open(os.path.join(synth_dir, f)).read(), ns)
            transform = ns['transform']
            ok = all(transform(t['input']) == t['output'] for t in task['test'])
            if ok: passed.append(tid)
            else: failed.append(tid)
        except Exception as e:
            errors.append((tid, str(e)))
    return passed, failed, errors

def check_direct(direct_dir, label):
    passed, failed = [], []
    if not os.path.isdir(direct_dir): return passed, failed
    for f in sorted(os.listdir(direct_dir)):
        if not f.endswith('.json'): continue
        tid = f.replace('.json','')
        task_path = os.path.join(EVAL_DIR, f"{tid}.json")
        if not os.path.exists(task_path): continue
        with open(task_path) as tf: task = json.load(tf)
        with open(os.path.join(direct_dir, f)) as pf: pred = json.load(pf)
        ok = True
        for i, t in enumerate(task['test']):
            if i < len(pred.get('test',[])):
                if pred['test'][i].get('output') != t['output']:
                    ok = False
            else:
                ok = False
        if ok: passed.append(tid)
        else: failed.append(tid)
    return passed, failed

all_passed = set()

for synth_dir, label in [(SYNTH_DIR, "DS Synth"), (OPUS_SYNTH, "Opus Synth")]:
    p, f, e = check_synth(synth_dir, label)
    print(f"\n{label}: {len(p)} PASS, {len(f)} FAIL, {len(e)} ERROR")
    for t in p: print(f"  ✅ {t}")
    all_passed.update(p)

for direct_dir, label in [(DIRECT_DIR, "DS Direct"), (OPUS_DIRECT, "Opus Direct")]:
    p, f = check_direct(direct_dir, label)
    print(f"\n{label}: {len(p)} PASS, {len(f)} FAIL")
    for t in p: print(f"  ✅ {t}")
    all_passed.update(p)

print(f"\n{'='*40}")
print(f"Combined unique PASS: {len(all_passed)}/120 ({100*len(all_passed)/120:.1f}%)")
for t in sorted(all_passed): print(f"  {t}")
