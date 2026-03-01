#!/usr/bin/env python3
"""Verify eval synth results against evaluation set."""
import json, os, glob, importlib.util, sys, signal

DATA_DIR = "/private/tmp/arc-agi-2/data/evaluation"
SYNTH_DIR = os.path.expanduser("~/verantyx_v6/eval_synth_results")

def timeout_handler(signum, frame):
    raise TimeoutError()

passed = []
failed = []

for py_file in sorted(glob.glob(f"{SYNTH_DIR}/*.py")):
    tid = os.path.splitext(os.path.basename(py_file))[0]
    task_path = f"{DATA_DIR}/{tid}.json"
    if not os.path.exists(task_path):
        continue
    with open(task_path) as f:
        task = json.load(f)
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        spec = importlib.util.spec_from_file_location("t", py_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.transform
        
        ok = True
        for split in ["train", "test"]:
            for ex in task.get(split, []):
                out = fn(ex["input"])
                # Handle numpy arrays
                if hasattr(out, 'tolist'):
                    out = out.tolist()
                # Normalize nested
                if out and hasattr(out[0], 'tolist'):
                    out = [row.tolist() for row in out]
                if out != ex["output"]:
                    ok = False
                    break
            if not ok:
                break
        
        signal.alarm(0)
        if ok:
            passed.append(tid)
        else:
            failed.append(tid)
    except Exception as e:
        signal.alarm(0)
        failed.append(tid)

print(f"PASS: {len(passed)}")
print(f"FAIL: {len(failed)}")
print(f"Total: {len(passed) + len(failed)}")
print(f"--- PASSED ---")
for t in passed:
    print(f"  {t}")
