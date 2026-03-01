#!/usr/bin/env python3
"""Verify synth_results/*.py against ARC training data."""
import json, os, sys, importlib.util, traceback

DATA_DIR = "/private/tmp/arc-agi-2/data/training"
SYNTH_DIR = os.path.expanduser("~/verantyx_v6/synth_results")

results = {"pass": [], "fail": [], "error": [], "no_data": []}

for fname in sorted(os.listdir(SYNTH_DIR)):
    if not fname.endswith(".py"):
        continue
    task_id = fname[:-3]
    data_path = os.path.join(DATA_DIR, f"{task_id}.json")
    if not os.path.exists(data_path):
        results["no_data"].append(task_id)
        continue
    
    with open(data_path) as f:
        task = json.load(f)
    
    # Load transform function
    spec = importlib.util.spec_from_file_location(task_id, os.path.join(SYNTH_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        results["error"].append((task_id, f"import: {e}"))
        continue
    
    if not hasattr(mod, "transform"):
        results["error"].append((task_id, "no transform()"))
        continue
    
    # Test on ALL examples (train + test)
    import signal
    def timeout_handler(signum, frame):
        raise TimeoutError("timeout")
    
    all_ok = True
    fail_info = []
    for split in ["train", "test"]:
        for i, ex in enumerate(task.get(split, [])):
            inp = [row[:] for row in ex["input"]]
            expected = ex["output"]
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                got = mod.transform(inp)
                signal.alarm(0)
                if got is None:
                    fail_info.append(f"{split}[{i}]: returned None")
                    all_ok = False
                    continue
                got_list = [list(row) for row in got]
                exp_list = [list(row) for row in expected]
                if got_list != exp_list:
                    fail_info.append(f"{split}[{i}]: mismatch")
                    all_ok = False
            except Exception as e:
                signal.alarm(0)
                fail_info.append(f"{split}[{i}]: {e}")
                all_ok = False
    
    if all_ok:
        results["pass"].append(task_id)
    else:
        results["fail"].append((task_id, "; ".join(fail_info[:3])))

print(f"PASS: {len(results['pass'])}")
print(f"FAIL: {len(results['fail'])}")
print(f"ERROR: {len(results['error'])}")
print(f"NO_DATA: {len(results['no_data'])}")
print()

# Show pass list
if results["pass"]:
    print("--- PASSED ---")
    for t in results["pass"]:
        print(t)

print()
if results["fail"]:
    print("--- FAILED (first 20) ---")
    for t, info in results["fail"][:20]:
        print(f"{t}: {info}")

if results["error"]:
    print("--- ERRORS (first 10) ---")
    for t, info in results["error"][:10]:
        print(f"{t}: {info}")
