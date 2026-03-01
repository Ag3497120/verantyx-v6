#!/usr/bin/env python3
"""Verify all hybrid results (synth + direct) against ground truth."""
import json, os, glob, importlib.util, signal

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
SYNTH_DIR = os.path.expanduser("~/verantyx_v6/eval_synth_results")
PREDICT_DIR = os.path.expanduser("~/verantyx_v6/eval_predict_results")

def timeout_handler(signum, frame):
    raise TimeoutError()

def grid_eq(a, b):
    if a is None or b is None: return False
    if len(a) != len(b): return False
    return all(ra == rb for ra, rb in zip(a, b))

def to_list(obj):
    if hasattr(obj, 'tolist'): return obj.tolist()
    if isinstance(obj, list): return [to_list(x) for x in obj]
    return obj

def check_synth(tid, task):
    """Check if synth solution passes all test examples"""
    py_file = f"{SYNTH_DIR}/{tid}.py"
    if not os.path.exists(py_file): return None
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        spec = importlib.util.spec_from_file_location("t", py_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.transform
        
        # Check ALL examples (train + test)
        for split in ["train", "test"]:
            for ex in task.get(split, []):
                out = to_list(fn(ex["input"]))
                if not grid_eq(out, ex["output"]):
                    signal.alarm(0)
                    return False
        signal.alarm(0)
        return True
    except:
        signal.alarm(0)
        return False

def check_predict(tid, task):
    """Check if direct prediction matches test output"""
    json_file = f"{PREDICT_DIR}/{tid}.json"
    if not os.path.exists(json_file): return None
    
    try:
        with open(json_file) as f:
            pred_data = json.load(f)
        
        predictions = pred_data.get("predictions", [])
        if not predictions: 
            # Single prediction format
            p = pred_data.get("prediction")
            if p: predictions = [p]
        
        test_outputs = [ex["output"] for ex in task["test"]]
        
        if len(predictions) != len(test_outputs):
            return False
        
        return all(grid_eq(p, e) for p, e in zip(predictions, test_outputs))
    except:
        return False

# Main
all_tasks = sorted([os.path.splitext(os.path.basename(f))[0] 
                    for f in glob.glob(f"{EVAL_DIR}/*.json")])

synth_pass = 0
predict_pass = 0
both_pass = 0
total_pass = 0
no_result = 0

for tid in all_tasks:
    with open(f"{EVAL_DIR}/{tid}.json") as f:
        task = json.load(f)
    
    s = check_synth(tid, task)
    p = check_predict(tid, task)
    
    passed = False
    method = ""
    if s == True:
        synth_pass += 1
        passed = True
        method = "synth"
    if p == True:
        predict_pass += 1
        if passed:
            both_pass += 1
            method = "both"
        else:
            passed = True
            method = "predict"
    
    if passed:
        total_pass += 1
        print(f"  ✓ {tid} ({method})")
    elif s is None and p is None:
        no_result += 1

print(f"\n{'='*50}")
print(f"Synth PASS:   {synth_pass}")
print(f"Predict PASS: {predict_pass}")
print(f"Both PASS:    {both_pass}")
print(f"Combined:     {total_pass}/120 ({total_pass*100/120:.1f}%)")
print(f"No result:    {no_result}")
print(f"{'='*50}")
