#!/usr/bin/env python3
"""
DeepSeek V3 program synthesis for ARC-AGI-2 evaluation set.
Test outputsを除去した安全なプロンプト。
"""
import json, os, sys, time, re, urllib.request
from pathlib import Path

EVAL_DIR = Path("/private/tmp/arc-agi-2/data/evaluation")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/ds_eval_results"))
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"

def grid_str(g):
    return '\n'.join(' '.join(str(c) for c in row) for row in g)

def call_deepseek(prompt, temp=0.0, max_tokens=4096, timeout=120):
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You solve ARC-AGI puzzles by writing Python. Output ONLY a ```python block with def transform(grid: list[list[int]]) -> list[list[int]]. No numpy. grid is list of lists of ints 0-9."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
    }).encode()
    
    req = urllib.request.Request(API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return None

def extract_fn(text):
    if not text: return None
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    code = m.group(1) if m else text
    if 'def transform' not in code: return None
    try:
        ns = {}
        exec(code, ns)
        return ns.get('transform')
    except:
        return None

def build_prompt(task):
    parts = ["Solve this ARC-AGI puzzle. Find the pattern from training examples and write a Python function.\n"]
    for i, ex in enumerate(task['train']):
        parts.append(f"Training {i+1} Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        parts.append(grid_str(ex['input']))
        parts.append(f"Training {i+1} Output ({len(ex['output'])}x{len(ex['output'][0])}):")
        parts.append(grid_str(ex['output']))
        parts.append("")
    
    for i, ex in enumerate(task['test']):
        parts.append(f"Test {i+1} Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        parts.append(grid_str(ex['input']))
        parts.append("")
    
    parts.append("Write def transform(grid) that transforms input to output. Pure Python, no numpy. grid = list[list[int]].")
    return '\n'.join(parts)

def grid_eq(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True

def main():
    tasks = sorted(f.stem for f in EVAL_DIR.glob("*.json"))
    
    # Skip already done
    done = set(f.stem for f in RESULTS_DIR.glob("*.json"))
    todo = [t for t in tasks if t not in done]
    
    print(f"Total: {len(tasks)}, Done: {len(done)}, Todo: {len(todo)}", flush=True)
    
    correct = 0
    attempted = 0
    
    for tid in todo:
        with open(EVAL_DIR / f"{tid}.json") as f:
            task = json.load(f)
        
        prompt = build_prompt(task)
        attempted += 1
        
        best_result = None
        
        for attempt in range(3):
            temp = [0.0, 0.3, 0.7][attempt]
            t0 = time.time()
            response = call_deepseek(prompt, temp=temp)
            elapsed = time.time() - t0
            
            fn = extract_fn(response)
            if fn is None:
                print(f"  {tid} attempt {attempt+1}: no valid function ({elapsed:.1f}s)", flush=True)
                continue
            
            # Verify on train
            train_ok = True
            for ex in task['train']:
                try:
                    pred = fn(ex['input'])
                    if not grid_eq(pred, ex['output']):
                        train_ok = False; break
                except:
                    train_ok = False; break
            
            if not train_ok:
                print(f"  {tid} attempt {attempt+1}: train fail ({elapsed:.1f}s)", flush=True)
                continue
            
            # Train passed! Generate test predictions
            test_preds = []
            for tex in task['test']:
                try:
                    pred = fn(tex['input'])
                    test_preds.append(pred)
                except:
                    test_preds = None; break
            
            if test_preds is None:
                print(f"  {tid} attempt {attempt+1}: test error ({elapsed:.1f}s)", flush=True)
                continue
            
            # Check test correctness
            test_ok = all(grid_eq(test_preds[j], task['test'][j]['output']) for j in range(len(task['test'])))
            
            best_result = {
                "task_id": tid,
                "status": "pass" if test_ok else "train_only",
                "attempt": attempt + 1,
                "temp": temp,
                "test": [{"output": p} for p in test_preds],
            }
            
            status = "✅" if test_ok else "⚠️"
            print(f"{status} {tid} attempt {attempt+1} temp={temp} ({elapsed:.1f}s)", flush=True)
            if test_ok:
                correct += 1
            break  # Train passed, move on
        
        if best_result is None:
            best_result = {"task_id": tid, "status": "fail"}
            print(f"❌ {tid}: all attempts failed", flush=True)
        
        with open(RESULTS_DIR / f"{tid}.json", "w") as f:
            json.dump(best_result, f)
    
    # Final tally
    all_results = list(RESULTS_DIR.glob("*.json"))
    total_pass = sum(1 for f in all_results if json.load(open(f)).get("status") == "pass")
    print(f"\n=== {total_pass}/{len(tasks)} correct ===", flush=True)

if __name__ == "__main__":
    main()
