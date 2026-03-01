#!/usr/bin/env python3
"""
DeepSeek V3 + Verantyx Brain (170K行の外付け脳)
=================================================
V3にverantyxの全資産をimport可能なライブラリとして提供。
V3はtransform関数を書き、ローカルで実行・検証する。
失敗時はエラーフィードバックで再生成（最大5回）。
"""
import json, os, sys, time, re, urllib.request, traceback
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = Path("/private/tmp/arc-agi-2/data/evaluation")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/brain_results2"))
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"

# API reference (全170K行の要約)
with open(os.path.expanduser("~/verantyx_v6/arc_api_reference.txt")) as f:
    API_REF = f.read()

# world_commandsの全関数名
from arc.world_commands import build_all_commands
_wc_names = [name for name, _ in build_all_commands()]

SYSTEM_PROMPT = """You are an expert ARC-AGI puzzle solver with access to a massive Python toolkit (170K+ lines).

You write `def transform(grid)` functions. `grid` is list[list[int]], colors 0-9. No numpy.

## Available Libraries (import and use freely):

### arc.world_commands — 216 grid transformation commands
```python
from arc.world_commands import (
    rot90, rot180, rot270, flip_h, flip_v, transpose,
    gravity_down, gravity_up, gravity_left, gravity_right,
    fill_enclosed, fill_row_gaps, fill_col_gaps, fill_between_same_color,
    sym_h, sym_v, sym_4fold, sym_diag_main,
    grow_1, grow_8, shrink_1, ca_life, ca_majority,
    extend_h, extend_v, connect_same_color_h, connect_same_color_v,
    crop_content, crop_largest_obj, crop_smallest_obj,
    keep_largest, keep_smallest, remove_largest, remove_smallest,
    fill_each_obj_bbox, trace_outline_4, trace_outline_8,
    upscale_2x, upscale_3x, downscale_2x,
    # ... 216 commands total
)
```

### arc.cross2 — Structural decomposition
```python
from arc.cross2 import CrossDecomposer, bg_color, grid_colors, color_counts, copy_grid
# CrossDecomposer.decompose_all(grid) -> list of Decomposition objects
# Each has: .objects (list of [(r,c,color),...]), .panels, .regions, .bg
```

### arc.grid — Grid utilities
```python
from arc.grid import grid_eq, grid_shape
```

### Key patterns in ARC:
- Object extraction/manipulation (find objects, move, transform, stamp)
- Panel operations (split by separator lines, combine with OR/AND/XOR)
- Color mapping (systematic color replacement)
- Symmetry (complete partial symmetric patterns)
- Fill (enclosed regions, gaps between same-color cells)
- Gravity (drop cells in direction)
- Growth/erosion (cellular automata-like)
- Pattern repetition (tile, stamp template)
- Neighborhood rules (transform based on local context)

IMPORTANT: Write complete, self-contained transform functions. Import what you need.
Output ONLY a ```python block.
"""

def grid_str(g):
    return '\n'.join(' '.join(str(c) for c in row) for row in g)

def call_v3(messages, temp=0.0, max_tokens=4096, timeout=180):
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
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
    if not text: return None, None
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    code = m.group(1) if m else text
    if 'def transform' not in code: return None, code
    try:
        ns = {"__builtins__": __builtins__}
        exec(code, ns)
        return ns.get('transform'), code
    except Exception as e:
        return None, f"{code}\n\nExec error: {e}"

def grid_eq(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True

def build_prompt(task):
    parts = ["Solve this ARC-AGI puzzle.\n"]
    for i, ex in enumerate(task['train']):
        ih, iw = len(ex['input']), len(ex['input'][0])
        oh, ow = len(ex['output']), len(ex['output'][0])
        parts.append(f"Train {i+1} Input ({ih}x{iw}):\n{grid_str(ex['input'])}")
        parts.append(f"Train {i+1} Output ({oh}x{ow}):\n{grid_str(ex['output'])}\n")
    for i, ex in enumerate(task['test']):
        ih, iw = len(ex['input']), len(ex['input'][0])
        parts.append(f"Test {i+1} Input ({ih}x{iw}):\n{grid_str(ex['input'])}\n")
    parts.append("Write def transform(grid) -> list[list[int]]. Use the available libraries.")
    return '\n'.join(parts)

def solve_task(tid, task):
    prompt = build_prompt(task)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    best_result = None
    last_feedback = None
    
    for attempt in range(8):
        temp = [0.0, 0.0, 0.3, 0.3, 0.5, 0.7, 0.7, 1.0][attempt]
        # Reset messages each attempt to avoid context overflow
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if last_feedback:
            messages.append({"role": "user", "content": f"Previous attempt failed: {last_feedback}\nTry a different approach. Output ```python block."})
        t0 = time.time()
        response = call_v3(messages, temp=temp, max_tokens=4096)
        elapsed = time.time() - t0
        
        fn, code = extract_fn(response)
        if fn is None:
            err = code or "No valid function"
            print(f"  {tid} #{attempt+1}: parse fail ({elapsed:.0f}s)", flush=True)
            # feedback
            last_feedback = f"Error: {err[:200]}\nFix it. Output ```python block with def transform(grid)."
            continue
        
        # Verify on train
        train_ok = True
        train_err = ""
        for i, ex in enumerate(task['train']):
            try:
                pred = fn(ex['input'])
                if not grid_eq(pred, ex['output']):
                    ph, pw = len(pred), len(pred[0]) if pred else 0
                    eh, ew = len(ex['output']), len(ex['output'][0])
                    # Count diff cells with detail
                    if (ph, pw) == (eh, ew):
                        diff_details = []
                        for r in range(ph):
                            for c in range(pw):
                                if pred[r][c] != ex['output'][r][c]:
                                    diff_details.append(f"({r},{c}): got {pred[r][c]} expected {ex['output'][r][c]}")
                        diffs = len(diff_details)
                        detail_str = "; ".join(diff_details[:10])
                        train_err = f"Train {i+1}: {diffs} cells differ ({ph}x{pw}). Diffs: {detail_str}"
                    else:
                        train_err = f"Train {i+1}: size mismatch pred={ph}x{pw} expected={eh}x{ew}"
                    train_ok = False; break
            except Exception as e:
                train_err = f"Train {i+1} error: {type(e).__name__}: {e}"
                train_ok = False; break
        
        if not train_ok:
            print(f"  {tid} #{attempt+1}: {train_err[:60]} ({elapsed:.0f}s)", flush=True)
            messages.append({"role": "assistant", "content": response})
            # Show full predicted vs expected for first failing train
            fb_parts = [f"Wrong. {train_err}"]
            for ii, exx in enumerate(task['train']):
                try:
                    pp = fn(exx['input'])
                    if pp and not grid_eq(pp, exx['output']):
                        fb_parts.append(f"\nTrain {ii+1} Your output:")
                        fb_parts.append("\n".join(" ".join(str(v) for v in row) for row in pp))
                        fb_parts.append(f"Train {ii+1} Expected:")
                        fb_parts.append("\n".join(" ".join(str(v) for v in row) for row in exx['output']))
                        break
                except: pass
            fb_parts.append("\nAnalyze the exact differences. Fix your logic. Output ```python block.")
            last_feedback = "\n".join(fb_parts)
            continue
        
        # Train passed! Generate test predictions
        test_preds = []
        for tex in task['test']:
            try:
                pred = fn(tex['input'])
                test_preds.append(pred)
            except Exception as e:
                test_preds = None; break
        
        if test_preds is None:
            print(f"  {tid} #{attempt+1}: test exec error ({elapsed:.0f}s)", flush=True)
            continue
        
        # Check test
        test_ok = all(grid_eq(test_preds[j], task['test'][j]['output']) 
                      for j in range(len(task['test'])))
        
        status = "✅" if test_ok else "⚠️train_pass"
        print(f"  {status} {tid} #{attempt+1} t={temp} ({elapsed:.0f}s)", flush=True)
        
        best_result = {
            "task_id": tid,
            "status": "pass" if test_ok else "train_only",
            "attempt": attempt + 1,
            "test": [{"output": p} for p in test_preds],
            "code": code,
        }
        break
    
    return best_result

def main():
    tasks = sorted(f.stem for f in EVAL_DIR.glob("*.json"))
    done = set(f.stem for f in RESULTS_DIR.glob("*.json"))
    todo = [t for t in tasks if t not in done]
    
    print(f"Total: {len(tasks)}, Done: {len(done)}, Todo: {len(todo)}", flush=True)
    
    for tid in todo:
        with open(EVAL_DIR / f"{tid}.json") as f:
            task = json.load(f)
        
        result = solve_task(tid, task)
        
        if result is None:
            result = {"task_id": tid, "status": "fail"}
            print(f"  ❌ {tid}: all attempts failed", flush=True)
        
        with open(RESULTS_DIR / f"{tid}.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # Final tally
    all_results = list(RESULTS_DIR.glob("*.json"))
    total_pass = 0
    total_train = 0
    for f in all_results:
        r = json.load(open(f))
        if r.get("status") == "pass": total_pass += 1
        elif r.get("status") == "train_only": total_train += 1
    
    print(f"\n=== Results ===", flush=True)
    print(f"Test correct: {total_pass}/{len(tasks)}", flush=True)
    print(f"Train only: {total_train}/{len(tasks)}", flush=True)

if __name__ == "__main__":
    main()
