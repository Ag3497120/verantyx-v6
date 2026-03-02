#!/usr/bin/env python3
"""
Evaluation set 120問: DeepSeek V3 + Verantyxヒント注入 synth
既存正解をスキップ、未解決101問を攻める
"""
import json, os, sys, time, re, urllib.request, copy, signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arc.hint_generator import generate_hints

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"
EVAL_DIR = Path("/tmp/arc-agi-2/data/evaluation")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/eval_hint_results"))
RESULTS_DIR.mkdir(exist_ok=True)

# Already solved
ALREADY_PASSED = {
    "0934a4d8", "136b0064", "16de56c4", "1818057f", "247ef758",
    "2ba387bc", "332f06d7", "38007db0", "45a5af55", "53fb4810",
    "58490d8a", "58f5dbd5", "5961cc34", "65b59efc", "6e453dd6",
    "7491f3cf", "7b5033c1", "bf45cf4b", "db695cfb",
}


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
        print(f"    API error: {e}", flush=True)
        return None


def extract_fn(text):
    if not text:
        return None, None
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    code = m.group(1) if m else text
    if 'def transform' not in code:
        return None, None
    try:
        ns = {}
        exec(code, ns)
        return ns.get('transform'), code
    except:
        return None, None


def grid_str(g):
    return '\n'.join(' '.join(str(c) for c in row) for row in g)


def grid_eq(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(ra == rb for ra, rb in zip(a, b))


def to_list(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, list):
        return [to_list(x) for x in obj]
    return obj


def build_prompt(task, hints=""):
    parts = ["Solve this ARC-AGI puzzle. Find the pattern from training examples and write a Python function.\n"]
    for i, ex in enumerate(task['train']):
        parts.append(f"Training {i+1} Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        parts.append(grid_str(ex['input']))
        parts.append(f"Training {i+1} Output ({len(ex['output'])}x{len(ex['output'][0])}):")
        parts.append(grid_str(ex['output']))
        parts.append("")
    if hints:
        parts.append("=== ANALYSIS HINTS (from structural analysis) ===")
        parts.append(hints)
        parts.append("=== END HINTS ===")
        parts.append("Use these hints to guide your solution.\n")
    test_inp = task['test'][0]['input']
    parts.append(f"Test Input ({len(test_inp)}x{len(test_inp[0])}):")
    parts.append(grid_str(test_inp))
    return '\n'.join(parts)


def test_train(fn, train_examples):
    for ex in train_examples:
        try:
            result = to_list(fn(copy.deepcopy(ex['input'])))
            if not grid_eq(result, ex['output']):
                return False
        except:
            return False
    return True


def main():
    task_files = sorted(EVAL_DIR.glob("*.json"))
    
    new_pass = 0
    new_fail = 0
    skipped = 0
    errors = 0
    
    print(f"Eval Hint DeepSeek V3 — {len(task_files)} tasks, {len(ALREADY_PASSED)} already solved", flush=True)
    print("=" * 60, flush=True)
    
    t_start = time.time()
    
    for i, tf in enumerate(task_files):
        tid = tf.stem
        
        if tid in ALREADY_PASSED:
            skipped += 1
            continue
        
        # Skip if already solved in this run
        result_file = RESULTS_DIR / f"{tid}.py"
        if result_file.exists():
            skipped += 1
            continue
        
        with open(tf) as f:
            task = json.load(f)
        
        test_output = task['test'][0].get('output')
        
        # Generate hints
        try:
            hints = generate_hints(task, include_partial=True)
        except:
            hints = ""
        
        # Try 3 temperatures
        solved = False
        for temp in [0.0, 0.3, 0.7]:
            prompt = build_prompt(task, hints=hints)
            resp = call_deepseek(prompt, temp=temp)
            fn, code = extract_fn(resp)
            
            if fn and test_train(fn, task['train']):
                # Save solution
                result_file.write_text(code)
                
                if test_output:
                    try:
                        pred = to_list(fn(copy.deepcopy(task['test'][0]['input'])))
                        if grid_eq(pred, test_output):
                            new_pass += 1
                            solved = True
                            print(f"  [{i+1:3d}/120] ✓ {tid} (temp={temp}) PASS", flush=True)
                            break
                    except:
                        pass
                
                # Train pass but test unknown/fail — still save, might be correct
                if not solved:
                    print(f"  [{i+1:3d}/120] ? {tid} (temp={temp}) train_pass, test_unknown", flush=True)
                    solved = True  # don't retry
                    break
        
        if not solved:
            new_fail += 1
            if (new_fail + new_pass) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1:3d}/120] ✗ {tid}  [running: +{new_pass} pass, {new_fail} fail, {elapsed:.0f}s]", flush=True)
            else:
                print(f"  [{i+1:3d}/120] ✗ {tid}", flush=True)
    
    elapsed = time.time() - t_start
    total_pass = len(ALREADY_PASSED) + new_pass
    
    print("=" * 60, flush=True)
    print(f"NEW passes:     {new_pass}", flush=True)
    print(f"NEW fails:      {new_fail}", flush=True)
    print(f"Skipped:        {skipped}", flush=True)
    print(f"TOTAL:          {total_pass}/120 ({total_pass/120*100:.1f}%)", flush=True)
    print(f"Time:           {elapsed:.0f}s ({elapsed/max(new_pass+new_fail,1):.1f}s/task)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
