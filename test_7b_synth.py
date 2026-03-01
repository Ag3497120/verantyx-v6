#!/usr/bin/env python3
"""Test Qwen 7B program synthesis on ARC tasks — compare with Sonnet results."""
import json, os, sys, subprocess, time, random, threading, re
import numpy as np

TASK_DIR = "/tmp/arc-agi-2/data/training"
SYNTH_DIR = os.path.expanduser("~/verantyx_v6/synth_results")

def query_ollama(prompt, model="gpt-oss:20b", timeout_sec=180):
    try:
        r = subprocess.run(
            ["ollama", "run", model],
            input=prompt, capture_output=True, text=True, timeout=timeout_sec
        )
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return ""

def grid_to_str(grid):
    return '\n'.join(' '.join(str(c) for c in row) for row in grid)

def make_prompt(task):
    parts = ["You are solving an ARC-AGI puzzle. Given input/output grid pairs, write a Python function `def transform(grid):` that transforms input to output.\n"]
    for i, ex in enumerate(task['train']):
        parts.append(f"Example {i+1}:")
        parts.append(f"Input:\n{grid_to_str(ex['input'])}")
        parts.append(f"Output:\n{grid_to_str(ex['output'])}")
        parts.append("")
    parts.append("Write ONLY a Python function `def transform(grid):` that takes a 2D list of ints and returns a 2D list of ints. You may use numpy. No explanation, just the function code.")
    return '\n'.join(parts)

def extract_code(response):
    m = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'(def transform\(.*?\).*(?:\n(?:    |\t).*)*)', response)
    if m:
        return m.group(1).strip()
    return response

def verify(task, code_str):
    from collections import Counter, defaultdict
    ns = {'np': np, 'numpy': np, 'Counter': Counter, 'defaultdict': defaultdict}
    try:
        import scipy; ns['scipy'] = scipy
        from scipy.ndimage import label; ns['label'] = label
    except: pass
    
    try:
        exec(code_str, ns)
    except:
        return False
    
    if 'transform' not in ns:
        return False
    fn = ns['transform']
    
    for ex in task['train']:
        result = [None]
        exc = [None]
        def run():
            try: result[0] = fn(ex['input'])
            except Exception as e: exc[0] = e
        t = threading.Thread(target=run); t.start(); t.join(5)
        if result[0] is None:
            return False
        try:
            pred = [[int(c) for c in row] for row in result[0]]
        except:
            return False
        if not np.array_equal(np.array(pred), np.array(ex['output'])):
            return False
    return True

def main():
    synth_ids = [f.replace('.py', '') for f in os.listdir(SYNTH_DIR) if f.endswith('.py')]
    
    random.seed(42)
    sample = random.sample(synth_ids, min(30, len(synth_ids)))
    
    solved = 0
    total = 0
    
    model = os.environ.get("TEST_MODEL", "gpt-oss:20b")
    print(f"Testing local LLM synthesis on {len(sample)} tasks (all solvable by Sonnet 4.5)")
    print(f"Model: {model}")
    print("=" * 60)
    sys.stdout.flush()
    
    for i, tid in enumerate(sample):
        task_path = os.path.join(TASK_DIR, f"{tid}.json")
        if not os.path.exists(task_path):
            print(f"  [{i+1}/{len(sample)}] SKIP {tid} (not found)")
            sys.stdout.flush()
            continue
        
        with open(task_path) as f:
            task = json.load(f)
        
        total += 1
        prompt = make_prompt(task)
        
        success = False
        for attempt in range(3):
            t0 = time.time()
            response = query_ollama(prompt, model=model)
            elapsed = time.time() - t0
            
            if not response:
                continue
            
            code = extract_code(response)
            if verify(task, code):
                success = True
                solved += 1
                print(f"  [{i+1}/{len(sample)}] ✓ {tid} (attempt {attempt+1}, {elapsed:.1f}s)")
                sys.stdout.flush()
                break
        
        if not success:
            print(f"  [{i+1}/{len(sample)}] ✗ {tid} (3 attempts failed)")
            sys.stdout.flush()
        
        if (i+1) % 10 == 0:
            rate = solved / total * 100 if total else 0
            print(f"  --- Progress: {solved}/{total} ({rate:.1f}%) ---")
            sys.stdout.flush()
    
    rate = solved / total * 100 if total else 0
    print("=" * 60)
    print(f"{model}:    {solved}/{total} ({rate:.1f}%)")
    print(f"Sonnet 4.5:  {total}/{total} (100.0%) — these tasks were selected because Sonnet solved them")
    print(f"")
    print(f"Estimated full score with {model}: {244 + int(596 * rate / 100)}/1000")
    print(f"Model dependency: ~{100 - rate:.0f}% of synth tasks need Sonnet-class models")

if __name__ == '__main__':
    main()
