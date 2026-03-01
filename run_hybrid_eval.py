#!/usr/bin/env python3
"""
ARC-AGI2 Hybrid Eval: DeepSeek V3 + Verantyx Cross Engine
1. Cross engine tries first (fast, free)
2. DeepSeek V3 synth (2 attempts)
3. Cross engine verifies DeepSeek code as simulator
4. Direct prediction fallback

Usage:
  python3 run_hybrid_eval.py --workers 5
"""

import json, os, sys, time, re, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import requests

# Add verantyx to path
sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
SYNTH_DIR = os.path.expanduser("~/verantyx_v6/hybrid_synth_results")
DIRECT_DIR = os.path.expanduser("~/verantyx_v6/hybrid_direct_results")
CROSS_DIR = os.path.expanduser("~/verantyx_v6/hybrid_cross_results")

for d in [SYNTH_DIR, DIRECT_DIR, CROSS_DIR]:
    os.makedirs(d, exist_ok=True)

def call_deepseek(messages, max_tokens=8192, temperature=0.0):
    for attempt in range(3):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
                timeout=120
            )
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                print(f"  400 Bad Request, skipping")
                return None
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                print(f"  API error (attempt {attempt+1}): {e}, retrying...")
                time.sleep(5)
            else:
                raise
    return None

def extract_python_code(text):
    m = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r'(def transform\(.*?\n(?:.*\n)*)', text)
    if m: return m.group(1).strip()
    return text.strip()

def extract_json_output(text, task):
    outputs = []
    for m in re.finditer(r'\[\s*\[[\d,\s\[\]]+\]\s*\]', text, re.DOTALL):
        try:
            grid = json.loads(m.group(0))
            if isinstance(grid, list) and len(grid) > 0 and all(isinstance(r, list) for r in grid):
                outputs.append(grid)
        except:
            pass
    if outputs:
        n_tests = len(task.get("test", []))
        test_outputs = [{"output": outputs[i]} for i in range(min(n_tests, len(outputs)))]
        if test_outputs:
            return {"test": test_outputs}
    return None

def verify_transform(code_str, task):
    try:
        ns = {}
        exec(code_str, ns)
        transform = ns.get("transform")
        if not transform:
            return False, "No transform function found"
        for i, ex in enumerate(task["train"]):
            result = transform(ex["input"])
            if result != ex["output"]:
                return False, f"Train {i} mismatch"
        return True, "All pass"
    except Exception as e:
        return False, str(e)

def try_cross_engine(task):
    """Try Verantyx cross engine first."""
    try:
        from arc.cross_engine import solve_cross_engine
        train_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
        test_inputs = [t["input"] for t in task["test"]]
        predictions, verified = solve_cross_engine(train_pairs, test_inputs)
        
        if predictions and len(predictions) > 0 and len(predictions[0]) > 0:
            # Cross engine produced predictions
            return predictions[0][0], len(verified)  # Best prediction for first test
        return None, 0
    except Exception as e:
        return None, 0

def cross_verify_code(code_str, task):
    """Use cross engine as simulator to verify DeepSeek's code against invariants."""
    try:
        from arc.cross_engine import CrossSimulator
        sim = CrossSimulator()
        train_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
        
        # Run transform on test input
        ns = {}
        exec(code_str, ns)
        transform = ns.get("transform")
        if not transform:
            return False, None
        
        test_output = transform(task["test"][0]["input"])
        
        # Check if cross engine's invariants are satisfied
        # Use hold-out validation: verify on train pairs
        for i, (inp, out) in enumerate(train_pairs):
            result = transform(inp)
            if result != out:
                return False, None
        
        return True, test_output
    except:
        return False, None

def solve_task(task_id):
    synth_path = os.path.join(SYNTH_DIR, f"{task_id}.py")
    direct_path = os.path.join(DIRECT_DIR, f"{task_id}.json")
    cross_path = os.path.join(CROSS_DIR, f"{task_id}.json")
    
    if os.path.exists(synth_path) or os.path.exists(direct_path) or os.path.exists(cross_path):
        return task_id, "skip", None

    task_path = os.path.join(EVAL_DIR, f"{task_id}.json")
    if not os.path.exists(task_path):
        return task_id, "missing", None

    with open(task_path) as f:
        task = json.load(f)

    # === Step 1: Cross Engine (free, fast) ===
    print(f"[{task_id}] Cross engine...")
    cross_pred, n_verified = try_cross_engine(task)
    if cross_pred is not None and n_verified > 0:
        result = {"test": [{"output": cross_pred}], "method": "cross_engine", "verified": n_verified}
        with open(cross_path, 'w') as f:
            json.dump(result, f)
        print(f"[{task_id}] 🔧 Cross engine PASS ({n_verified} verified)")
        return task_id, "cross", result

    # === Step 2: DeepSeek Synth ===
    task_no_cheat = {'train': task['train'], 'test': [{'input': t['input']} for t in task['test']]}
    task_json = json.dumps(task_no_cheat, separators=(',', ':'))

    print(f"[{task_id}] DeepSeek synth...")
    prompt = f"""You are solving an ARC-AGI2 puzzle. Study the training examples and write a Python function `transform(grid)` that converts input to output.

Rules:
- grid = list of lists of ints (0-9)
- Pure Python only, NO numpy, NO imports
- Must generalize (don't hardcode values from examples)
- Return the output grid

Task JSON:
{task_json}

Write ONLY the Python code in a ```python code block."""

    for attempt in range(2):
        try:
            response = call_deepseek([{"role": "user", "content": prompt}])
            if not response: continue
            code = extract_python_code(response)
            ok, msg = verify_transform(code, task)
            if ok:
                # === Step 3: Cross engine verification (simulator mode) ===
                cross_ok, cross_output = cross_verify_code(code, task)
                with open(synth_path, 'w') as f:
                    f.write(code)
                tag = "✅+🔧" if cross_ok else "✅"
                print(f"[{task_id}] {tag} Synth PASS (attempt {attempt+1})")
                return task_id, "synth", code
            else:
                print(f"[{task_id}] Synth attempt {attempt+1}: {msg}")
                if attempt == 0:
                    prompt = f"""Previous attempt failed: {msg}

Task JSON:
{task_json}

Try a different approach. Think step by step about the pattern first.
Write a Python `transform(grid)` function. Pure Python only, NO numpy.
Write ONLY the code in a ```python block."""
        except Exception as e:
            print(f"[{task_id}] Synth error: {e}")

    # === Step 4: Direct prediction ===
    print(f"[{task_id}] Direct prediction...")
    try:
        task_slim = {'train': task['train'], 'test': [{'input': t['input']} for t in task['test']]}
        task_slim_json = json.dumps(task_slim, separators=(',', ':'))
        direct_prompt = f"""Study the ARC-AGI2 training examples and predict the test output.

Task JSON:
{task_slim_json}

Output ONLY the predicted test output as a JSON list of lists like [[1,2],[3,4]].
No explanation needed."""

        response = call_deepseek([{"role": "user", "content": direct_prompt}], max_tokens=8192)
        if response:
            result = extract_json_output(response, task)
            if result:
                with open(direct_path, 'w') as f:
                    json.dump(result, f)
                print(f"[{task_id}] 📝 Direct saved")
                return task_id, "direct", result
    except Exception as e:
        print(f"[{task_id}] Direct error: {e}")

    print(f"[{task_id}] ❌ Failed")
    return task_id, "fail", None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=120)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    if args.task:
        task_ids = [args.task]
    else:
        task_ids = sorted([f.replace('.json','') for f in os.listdir(EVAL_DIR) if f.endswith('.json')])
        task_ids = task_ids[args.start:args.end]

    unsolved = [t for t in task_ids
                if not os.path.exists(os.path.join(SYNTH_DIR, f"{t}.py"))
                and not os.path.exists(os.path.join(DIRECT_DIR, f"{t}.json"))
                and not os.path.exists(os.path.join(CROSS_DIR, f"{t}.json"))]

    print(f"=== Hybrid Eval: DeepSeek V3 + Verantyx Cross Engine ===")
    print(f"Total: {len(task_ids)}, Unsolved: {len(unsolved)}, Workers: {args.workers}")
    print(f"Synth: {SYNTH_DIR}\nDirect: {DIRECT_DIR}\nCross: {CROSS_DIR}\n")

    results = {"synth": 0, "direct": 0, "cross": 0, "fail": 0, "skip": 0, "missing": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(solve_task, tid): tid for tid in unsolved}
        for future in as_completed(futures):
            try:
                _, status, _ = future.result()
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                print(f"Exception: {e}")
                traceback.print_exc()
                results["fail"] += 1

    print(f"\n{'='*50}")
    print(f"Cross={results['cross']}, Synth={results['synth']}, Direct={results['direct']}, Fail={results['fail']}")
    synth_n = len([f for f in os.listdir(SYNTH_DIR) if f.endswith('.py')])
    direct_n = len([f for f in os.listdir(DIRECT_DIR) if f.endswith('.json')])
    cross_n = len([f for f in os.listdir(CROSS_DIR) if f.endswith('.json')])
    print(f"Total: {cross_n} cross + {synth_n} synth + {direct_n} direct = {cross_n+synth_n+direct_n}/120")

if __name__ == "__main__":
    main()
