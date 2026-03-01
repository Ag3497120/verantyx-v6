#!/usr/bin/env python3
"""
verantyx_assist.py — verantyxがDeepSeek V3の脳を強化する
=========================================================
1. verantyxが問題を事前分析（分解、部分一致、特徴抽出）
2. 分析結果をV3のプロンプトに注入
3. V3が分析を参考にコードを書く
4. train検証 → 差分フィードバック → 再生成
"""
import json, os, sys, time, re, urllib.request
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = Path("/private/tmp/arc-agi-2/data/evaluation")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/assist_results"))
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"

# ═══════════════════════════════════════
# verantyxの事前分析
# ═══════════════════════════════════════

def analyze_task(task):
    """verantyxの全資産を使って問題を事前分析"""
    train = task['train']
    analysis = []
    
    # 1. サイズ関係
    in_sizes = [(len(e['input']), len(e['input'][0])) for e in train]
    out_sizes = [(len(e['output']), len(e['output'][0])) for e in train]
    same_size = in_sizes == out_sizes
    analysis.append(f"Input sizes: {in_sizes}, Output sizes: {out_sizes}")
    if same_size:
        analysis.append("Same size: input and output have identical dimensions")
    elif len(set(out_sizes)) == 1:
        analysis.append(f"Constant output size: {out_sizes[0]}")
    
    # 2. 色分析
    for i, e in enumerate(train):
        in_colors = set(); out_colors = set()
        for row in e['input']: in_colors.update(row)
        for row in e['output']: out_colors.update(row)
        in_bg = Counter(v for row in e['input'] for v in row).most_common(1)[0][0]
        out_bg = Counter(v for row in e['output'] for v in row).most_common(1)[0][0]
        new_colors = out_colors - in_colors
        removed = in_colors - out_colors
        analysis.append(f"Train {i+1}: in_colors={sorted(in_colors)} out_colors={sorted(out_colors)} bg={in_bg}->{out_bg} new={sorted(new_colors)} removed={sorted(removed)}")
    
    # 3. 差分分析（same sizeの場合）
    if same_size:
        for i, e in enumerate(train):
            h, w = len(e['input']), len(e['input'][0])
            changed = []
            for r in range(h):
                for c in range(w):
                    if e['input'][r][c] != e['output'][r][c]:
                        changed.append((r, c, e['input'][r][c], e['output'][r][c]))
            analysis.append(f"Train {i+1}: {len(changed)}/{h*w} cells changed")
            if len(changed) <= 20:
                for r, c, old, new in changed:
                    analysis.append(f"  ({r},{c}): {old} -> {new}")
    
    # 4. 構造分析（セパレータ、オブジェクト数）
    from arc.cross2 import CrossDecomposer, bg_color
    for i, e in enumerate(train):
        decomps = CrossDecomposer.decompose_all(e['input'])
        kinds = [d.kind for d in decomps]
        for d in decomps:
            if d.kind == 'obj4':
                obj_sizes = sorted([len(o) for o in d.objects], reverse=True)
                obj_colors = [Counter(v for _,_,v in o).most_common(1)[0][0] for o in d.objects]
                analysis.append(f"Train {i+1} input: {len(d.objects)} objects (sizes={obj_sizes[:5]}, colors={obj_colors[:5]})")
            elif d.kind == 'pan_h':
                analysis.append(f"Train {i+1} input: {len(d.panels)} horizontal panels")
            elif d.kind == 'pan_v':
                analysis.append(f"Train {i+1} input: {len(d.panels)} vertical panels")
            elif d.kind == 'enclosed':
                analysis.append(f"Train {i+1} input: {len(d.regions)} enclosed regions")
        
        decomps_out = CrossDecomposer.decompose_all(e['output'])
        for d in decomps_out:
            if d.kind == 'obj4':
                analysis.append(f"Train {i+1} output: {len(d.objects)} objects")
    
    # 5. world commandsの部分一致スキャン
    from arc.world_commands import build_all_commands
    from arc.grid import grid_eq
    train_pairs = [(e['input'], e['output']) for e in train]
    cmds = build_all_commands(train_pairs)
    inp0, out0 = train_pairs[0]
    
    best_cmds = []
    for name, fn in cmds:
        try:
            r = fn(inp0)
            if r is None: continue
            if grid_eq(r, out0):
                best_cmds.append((name, "EXACT match on train[0]"))
                continue
            # partial match (same size)
            if len(r) == len(out0) and len(r[0]) == len(out0[0]):
                h, w = len(out0), len(out0[0])
                match = sum(1 for rr in range(h) for cc in range(w) if r[rr][cc] == out0[rr][cc])
                total = h * w
                pct = match / total * 100
                if pct > 70:
                    best_cmds.append((name, f"{pct:.0f}% cells match"))
        except:
            pass
    
    best_cmds.sort(key=lambda x: x[1], reverse=True)
    if best_cmds:
        analysis.append("Promising world commands (applied to train[0] input):")
        for name, desc in best_cmds[:10]:
            analysis.append(f"  {name}: {desc}")
    
    # 6. 色マップ検出
    cmap = {}; cmap_ok = True
    if same_size:
        for e in train:
            h, w = len(e['input']), len(e['input'][0])
            for r in range(h):
                for c in range(w):
                    ic, oc = e['input'][r][c], e['output'][r][c]
                    if ic in cmap:
                        if cmap[ic] != oc: cmap_ok = False; break
                    else: cmap[ic] = oc
                if not cmap_ok: break
            if not cmap_ok: break
        if cmap_ok and any(k != v for k, v in cmap.items()):
            analysis.append(f"Color mapping detected: {cmap}")
    
    return '\n'.join(analysis)


def grid_str(g):
    return '\n'.join(' '.join(str(c) for c in row) for row in g)

def grid_eq(a, b):
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]: return False
    return True

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
    except:
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
        return None, f"Exec error: {e}"

def solve_task(tid, task):
    # verantyxの事前分析
    t_analysis = time.time()
    analysis = analyze_task(task)
    analysis_time = time.time() - t_analysis
    
    prompt_parts = [
        "Solve this ARC-AGI puzzle.\n",
        "## Verantyx Analysis (automated pre-analysis)\n",
        analysis, "\n",
        "## Task Data\n",
    ]
    for i, ex in enumerate(task['train']):
        prompt_parts.append(f"Train {i+1} Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        prompt_parts.append(grid_str(ex['input']))
        prompt_parts.append(f"Train {i+1} Output ({len(ex['output'])}x{len(ex['output'][0])}):")
        prompt_parts.append(grid_str(ex['output']))
        prompt_parts.append("")
    for i, ex in enumerate(task['test']):
        prompt_parts.append(f"Test {i+1} Input ({len(ex['input'])}x{len(ex['input'][0])}):")
        prompt_parts.append(grid_str(ex['input']))
        prompt_parts.append("")
    prompt_parts.append("Write def transform(grid) -> list[list[int]]. Pure Python, no numpy. Use the analysis above as hints.")
    prompt = '\n'.join(prompt_parts)
    
    SYSTEM = """You solve ARC-AGI puzzles by writing Python transform functions.
grid = list[list[int]], colors 0-9. No numpy.
You receive automated analysis from Verantyx (170K+ lines of ARC-solving code).
Use the analysis hints to understand the pattern, then write def transform(grid).
Output ONLY a ```python block."""
    
    best_result = None
    last_feedback = None
    
    for attempt in range(6):
        temp = [0.0, 0.0, 0.3, 0.5, 0.7, 1.0][attempt]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        if last_feedback:
            messages.append({"role": "user", "content": last_feedback})
        
        t0 = time.time()
        response = call_v3(messages, temp=temp)
        elapsed = time.time() - t0
        
        fn, code = extract_fn(response)
        if fn is None:
            print(f"  {tid} #{attempt+1}: parse fail ({elapsed:.0f}s)", flush=True)
            last_feedback = f"Previous code had syntax error. Write valid Python with def transform(grid). Output ```python block."
            continue
        
        # Verify all train
        all_pass = True
        feedback_parts = []
        for i, ex in enumerate(task['train']):
            try:
                pred = fn(ex['input'])
                if not grid_eq(pred, ex['output']):
                    all_pass = False
                    ph, pw = len(pred), len(pred[0]) if pred else 0
                    eh, ew = len(ex['output']), len(ex['output'][0])
                    if (ph, pw) == (eh, ew):
                        diffs = []
                        for r in range(ph):
                            for c in range(pw):
                                if pred[r][c] != ex['output'][r][c]:
                                    diffs.append(f"({r},{c}): got {pred[r][c]} want {ex['output'][r][c]}")
                        feedback_parts.append(f"Train {i+1}: {len(diffs)} wrong cells. {'; '.join(diffs[:8])}")
                    else:
                        feedback_parts.append(f"Train {i+1}: size {ph}x{pw} != expected {eh}x{ew}")
            except Exception as e:
                all_pass = False
                feedback_parts.append(f"Train {i+1}: {type(e).__name__}: {e}")
        
        if not all_pass:
            summary = ". ".join(feedback_parts[:3])
            print(f"  {tid} #{attempt+1}: {summary[:70]} ({elapsed:.0f}s)", flush=True)
            last_feedback = f"Wrong: {summary}\nFix your logic. Output ```python block with def transform(grid)."
            continue
        
        # Train passed!
        test_preds = []
        for tex in task['test']:
            try:
                test_preds.append(fn(tex['input']))
            except:
                test_preds = None; break
        
        if test_preds is None:
            print(f"  {tid} #{attempt+1}: test exec error ({elapsed:.0f}s)", flush=True)
            last_feedback = "Code crashed on test input. Make it more robust. Output ```python block."
            continue
        
        test_ok = all(grid_eq(test_preds[j], task['test'][j]['output']) for j in range(len(task['test'])))
        status = "✅" if test_ok else "⚠️"
        print(f"  {status} {tid} #{attempt+1} t={temp} ({elapsed:.0f}s) [analysis:{analysis_time:.1f}s]", flush=True)
        
        best_result = {
            "task_id": tid,
            "status": "pass" if test_ok else "train_only",
            "attempt": attempt + 1,
            "test": [{"output": p} for p in test_preds],
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
            print(f"  ❌ {tid}: all failed", flush=True)
        with open(RESULTS_DIR / f"{tid}.json", "w") as f:
            json.dump(result, f, indent=2)
    
    all_r = list(RESULTS_DIR.glob("*.json"))
    tp = sum(1 for f in all_r if json.load(open(f)).get("status") == "pass")
    to = sum(1 for f in all_r if json.load(open(f)).get("status") == "train_only")
    print(f"\n=== Test correct: {tp}/{len(tasks)}, Train only: {to} ===", flush=True)

if __name__ == "__main__":
    main()
