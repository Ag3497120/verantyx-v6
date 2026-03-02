#!/usr/bin/env python3
"""
Gemini 2.5 Pro + Verantyxヒント注入 — Evaluation set 120問
A/B: 最初の30問でベースライン(no hint) vs ヒント付きを比較
残り90問はヒント付きのみ
"""
import json, os, sys, time, re, urllib.request, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arc.hint_generator import generate_hints

API_KEY = os.environ.get("GOOGLE_GEMINI_API_KEY", "AIzaSyATPOY0fmk94_bOWvkj13tvXGIegyjsZKE")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={API_KEY}"
EVAL_DIR = Path("/tmp/arc-agi-2/data/evaluation")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/eval_gemini_hint_results"))
RESULTS_DIR.mkdir(exist_ok=True)

ALREADY_PASSED = {
    "0934a4d8", "136b0064", "16de56c4", "1818057f", "247ef758",
    "2ba387bc", "332f06d7", "38007db0", "45a5af55", "53fb4810",
    "58490d8a", "58f5dbd5", "5961cc34", "65b59efc", "6e453dd6",
    "7491f3cf", "7b5033c1", "bf45cf4b", "db695cfb",
}

AB_TEST_COUNT = 30  # first N unsolved for A/B comparison


def call_gemini(prompt, temp=0.0, timeout=120):
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temp,
            "maxOutputTokens": 8192,
        },
        "systemInstruction": {
            "parts": [{"text": "You solve ARC-AGI puzzles by writing Python. Output ONLY a ```python block with def transform(grid: list[list[int]]) -> list[list[int]]. No numpy. grid is list of lists of ints 0-9. Think step by step about the pattern, then write the function."}]
        },
    }).encode()
    req = urllib.request.Request(API_URL, data=payload, headers={
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return None
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
    parts = ["Solve this ARC-AGI puzzle. Study ALL training examples carefully to find the general pattern, then write a Python function.\n"]
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
        parts.append("Use these hints to guide your solution — they describe structural properties detected in the examples.\n")
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


def try_solve(task, hints="", temps=[0.0, 0.4]):
    """Try to solve with given hints, return (solved, code) or (False, None)"""
    for temp in temps:
        prompt = build_prompt(task, hints=hints)
        resp = call_gemini(prompt, temp=temp)
        fn, code = extract_fn(resp)
        if fn and test_train(fn, task['train']):
            test_output = task['test'][0].get('output')
            if test_output:
                try:
                    pred = to_list(fn(copy.deepcopy(task['test'][0]['input'])))
                    if grid_eq(pred, test_output):
                        return True, code
                except:
                    pass
            return None, code  # train pass, test unknown
        time.sleep(1)  # rate limit
    return False, None


def main():
    task_files = sorted(EVAL_DIR.glob("*.json"))
    
    # Separate into already solved and unsolved
    unsolved = []
    for tf in task_files:
        tid = tf.stem
        if tid not in ALREADY_PASSED:
            unsolved.append(tf)
    
    print(f"Gemini 2.5 Pro + Hints — eval set", flush=True)
    print(f"Total: {len(task_files)}, Already solved: {len(ALREADY_PASSED)}, To test: {len(unsolved)}", flush=True)
    print(f"A/B test on first {AB_TEST_COUNT}, then hints-only for rest", flush=True)
    print("=" * 70, flush=True)
    
    t_start = time.time()
    
    # Stats
    ab_hint_pass = 0
    ab_nohint_pass = 0
    ab_count = 0
    hint_only_pass = 0
    hint_only_count = 0
    
    for i, tf in enumerate(unsolved):
        tid = tf.stem
        
        # Skip if already solved this run
        if (RESULTS_DIR / f"{tid}.py").exists():
            continue
        
        with open(tf) as f:
            task = json.load(f)
        
        # Generate hints
        try:
            hints = generate_hints(task, include_partial=True)
        except:
            hints = ""
        
        if i < AB_TEST_COUNT:
            # A/B test mode
            ab_count += 1
            
            # WITH hints
            solved_h, code_h = try_solve(task, hints=hints)
            h_mark = "✓" if solved_h else ("?" if solved_h is None else "✗")
            if solved_h:
                ab_hint_pass += 1
                (RESULTS_DIR / f"{tid}.py").write_text(code_h)
            
            time.sleep(2)  # rate limit between calls
            
            # WITHOUT hints
            solved_n, code_n = try_solve(task, hints="")
            n_mark = "✓" if solved_n else ("?" if solved_n is None else "✗")
            if solved_n:
                ab_nohint_pass += 1
                if not solved_h:  # save no-hint solution if hint didn't solve
                    (RESULTS_DIR / f"{tid}.py").write_text(code_n)
            
            delta = ""
            if solved_h and not solved_n:
                delta = " ← HINT WIN"
            elif solved_n and not solved_h:
                delta = " ← NO-HINT WIN"
            
            elapsed = time.time() - t_start
            print(f"  [AB {ab_count:2d}/{AB_TEST_COUNT}] {tid}  H={h_mark} N={n_mark}{delta}  ({elapsed:.0f}s)", flush=True)
            
        else:
            # Hints-only mode
            hint_only_count += 1
            solved, code = try_solve(task, hints=hints)
            
            if solved:
                hint_only_pass += 1
                (RESULTS_DIR / f"{tid}.py").write_text(code)
            
            mark = "✓" if solved else ("?" if solved is None else "✗")
            elapsed = time.time() - t_start
            
            if hint_only_count % 10 == 0 or solved:
                print(f"  [H {hint_only_count:2d}] {tid} {mark}  [+{hint_only_pass} pass, {elapsed:.0f}s]", flush=True)
            else:
                print(f"  [H {hint_only_count:2d}] {tid} {mark}", flush=True)
        
        time.sleep(2)  # global rate limit
    
    elapsed = time.time() - t_start
    new_total = ab_hint_pass + hint_only_pass
    grand_total = len(ALREADY_PASSED) + new_total
    
    print("=" * 70, flush=True)
    print(f"A/B TEST ({AB_TEST_COUNT} problems):", flush=True)
    print(f"  WITH hints:    {ab_hint_pass}/{ab_count}", flush=True)
    print(f"  WITHOUT hints: {ab_nohint_pass}/{ab_count}", flush=True)
    print(f"  Delta:         {ab_hint_pass - ab_nohint_pass:+d}", flush=True)
    print(f"", flush=True)
    print(f"HINTS-ONLY ({hint_only_count} problems): {hint_only_pass}", flush=True)
    print(f"NEW total:       {new_total}", flush=True)
    print(f"GRAND TOTAL:     {grand_total}/120 ({grand_total/120*100:.1f}%)", flush=True)
    print(f"Time:            {elapsed:.0f}s", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
