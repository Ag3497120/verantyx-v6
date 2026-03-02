"""
Repair Solver — verantyxの「ほぼ正解」出力をV3に微修正させる
=============================================================
1. verantyxの全ソルバーを実行、最も近い出力グリッドを取得
2. そのグリッドと正解の差分を計算
3. V3に「このグリッドのここだけ直して、なぜ直すべきか推論して」と指示
"""
import sys, os, time, json, re, urllib.request
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

from brain.analyzer import grid_eq, grid_diff, grid_match_pct, grid_str
from brain.svd_router import analyze_arc_task_svd, get_svd_prompt_enhancement
from brain.analyzer import analyze_structure

EVAL_DIR = Path("/private/tmp/arc-agi-2/data/evaluation")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/repair_results"))
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"


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
    if not text: return None, "empty"
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    code = m.group(1) if m else text
    if 'def transform' not in code: return None, "no def transform"
    try:
        ns = {"__builtins__": __builtins__}
        exec(code, ns)
        return ns.get('transform'), code
    except Exception as e:
        return None, str(e)


def get_best_baseline(task):
    """
    verantyxの全ソルバーを実行、各train例で最も一致率の高い出力を返す。
    Returns: list of (solver_name, match_pct, pred_grids_per_train, diff_per_train)
    """
    train_pairs = [(e['input'], e['output']) for e in task['train']]
    candidates = []
    
    # === World Commands (216個) ===
    try:
        from arc.world_commands import build_all_commands
        cmds = build_all_commands(train_pairs)
        for name, fn in cmds:
            preds = []
            total_pct = 0
            ok = True
            for inp, out in train_pairs:
                try:
                    p = fn(inp)
                    if p is None: ok = False; break
                    preds.append(p)
                    total_pct += grid_match_pct(p, out)
                except:
                    ok = False; break
            if ok and preds:
                avg_pct = total_pct / len(train_pairs)
                if avg_pct > 20:
                    candidates.append((f"wc:{name}", avg_pct, preds))
    except: pass
    
    # === World Priors (43個) ===
    try:
        from arc.world_priors import generate_world_prior_pieces
        pieces = generate_world_prior_pieces(train_pairs)
        for p in (pieces or []):
            preds = []
            total_pct = 0
            ok = True
            for inp, out in train_pairs:
                try:
                    pred = p.apply(inp)
                    if pred is None: ok = False; break
                    preds.append(pred)
                    total_pct += grid_match_pct(pred, out)
                except:
                    ok = False; break
            if ok and preds:
                avg_pct = total_pct / len(train_pairs)
                if avg_pct > 20:
                    nm = getattr(p, 'name', str(type(p).__name__))
                    candidates.append((f"wp:{nm}", avg_pct, preds))
    except: pass
    
    # === World Commands 収束 ===
    try:
        from arc.world_commands import build_all_commands
        cmds_dict = dict(build_all_commands(train_pairs))
        for name, fn in list(cmds_dict.items())[:50]:
            preds = []
            total_pct = 0
            ok = True
            for inp, out in train_pairs:
                try:
                    g = [row[:] for row in inp]
                    for _ in range(10):
                        g2 = fn(g)
                        if g2 is None or grid_eq(g2, g): break
                        g = g2
                    preds.append(g)
                    total_pct += grid_match_pct(g, out)
                except:
                    ok = False; break
            if ok and preds:
                avg_pct = total_pct / len(train_pairs)
                if avg_pct > 20:
                    candidates.append((f"wc_conv:{name}", avg_pct, preds))
    except: pass
    
    # === World Commands 2段合成 (top10 × top50) ===
    try:
        from arc.world_commands import build_all_commands
        cmds = build_all_commands(train_pairs)
        # train[0]でtop10を選出
        scored = []
        for name, fn in cmds:
            try:
                r = fn(train_pairs[0][0])
                if r: scored.append((name, fn, grid_match_pct(r, train_pairs[0][1])))
            except: pass
        scored.sort(key=lambda x: -x[2])
        
        for name1, fn1, _ in scored[:10]:
            for name2, fn2, _ in scored[:30]:
                if name1 == name2: continue
                preds = []
                total_pct = 0
                ok = True
                for inp, out in train_pairs:
                    try:
                        mid = fn1(inp)
                        if mid is None: ok = False; break
                        r = fn2(mid)
                        if r is None: ok = False; break
                        preds.append(r)
                        total_pct += grid_match_pct(r, out)
                    except:
                        ok = False; break
                if ok and preds:
                    avg_pct = total_pct / len(train_pairs)
                    if avg_pct > 30:
                        candidates.append((f"wc2:{name1}+{name2}", avg_pct, preds))
    except: pass
    
    # === Cross2 Tools ===
    try:
        from arc.cross2 import CrossRouter
        router = CrossRouter()
        for inp, out in train_pairs[:1]:
            results = router.solve(inp, out)
            for r in (results or []):
                if hasattr(r, 'grid') and r.grid:
                    pct = grid_match_pct(r.grid, out)
                    if pct > 20:
                        # 全trainで検証
                        preds_all = []
                        ok = True
                        total = 0
                        for inp2, out2 in train_pairs:
                            try:
                                r2 = r.apply(inp2) if hasattr(r, 'apply') else None
                                if r2 is None: ok = False; break
                                preds_all.append(r2)
                                total += grid_match_pct(r2, out2)
                            except:
                                ok = False; break
                        if ok and preds_all:
                            candidates.append((f"cross2:{getattr(r, 'name', '?')}", total / len(train_pairs), preds_all))
    except: pass
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda x: -x[1])
    return candidates[:5]  # top 5


def solve_with_repair(tid, task):
    """repair戦略: verantyxのベスト出力をV3に見せて微修正させる"""
    
    t0 = time.time()
    baselines = get_best_baseline(task)
    baseline_time = time.time() - t0
    
    if not baselines:
        print(f"  [{tid}] no baseline found ({baseline_time:.1f}s)", flush=True)
        return None
    
    best_name, best_pct, best_preds = baselines[0]
    print(f"  [{tid}] best={best_name} ({best_pct:.1f}%) baseline={baseline_time:.1f}s", flush=True)
    
    # 各trainの差分を計算
    train_diffs = []
    for i, (ex, pred) in enumerate(zip(task['train'], best_preds)):
        out = ex['output']
        if len(pred) == len(out) and (not pred or not out or len(pred[0]) == len(out[0])):
            diffs = grid_diff(pred, out)
            train_diffs.append(diffs)
        else:
            train_diffs.append(None)  # サイズ不一致
    
    # === SVD Expert分析 ===
    try:
        struct = analyze_structure(task)
        svd_result = analyze_arc_task_svd(task, struct)
        svd_text = get_svd_prompt_enhancement(svd_result)
        cats = svd_result.get('categories', [])
        top_cat = cats[0][0] if cats else 'unknown'
        print(f"  [{tid}] svd: {top_cat} ({cats[0][1]:.0%})" if cats else f"  [{tid}] svd: none", flush=True)
    except Exception as e:
        svd_text = ""
        print(f"  [{tid}] svd error: {e}", flush=True)
    
    # === プロンプト構築 ===
    system = """You solve ARC-AGI puzzles by writing Python transform functions.
grid = list[list[int]], colors 0-9. Pure Python, no numpy.

IMPORTANT: Verantyx (170K lines of ARC solvers) has already found an APPROXIMATE solution.
You will see the approximate output and the EXACT cells that are wrong.
Your job: understand WHY those cells are wrong, figure out the TRUE pattern, and write the CORRECT transform.
Output ONLY a ```python block with def transform(grid)."""

    parts = []
    parts.append(f"## Verantyx Baseline: `{best_name}` — {best_pct:.1f}% accurate\n")
    
    for i, ex in enumerate(task['train']):
        h, w = len(ex['input']), len(ex['input'][0])
        parts.append(f"### Train {i+1}\n")
        parts.append(f"Input ({h}x{w}):")
        parts.append(grid_str(ex['input']))
        
        oh, ow = len(ex['output']), len(ex['output'][0])
        parts.append(f"\nExpected Output ({oh}x{ow}):")
        parts.append(grid_str(ex['output']))
        
        if i < len(best_preds):
            pred = best_preds[i]
            ph, pw = len(pred), len(pred[0]) if pred else 0
            parts.append(f"\nVerantyx Output ({ph}x{pw}) — {best_name}:")
            parts.append(grid_str(pred))
            
            diffs = train_diffs[i] if i < len(train_diffs) else None
            if diffs is not None:
                parts.append(f"\n❌ {len(diffs)} cells WRONG:")
                for r, c, got, want in diffs[:20]:
                    parts.append(f"  ({r},{c}): verantyx={got} correct={want}")
                if len(diffs) > 20:
                    parts.append(f"  ... and {len(diffs)-20} more")
            elif diffs is None and i < len(train_diffs):
                parts.append(f"\n❌ Size mismatch: verantyx={ph}x{pw} expected={oh}x{ow}")
        parts.append("")
    
    # 他のベースライン候補も提示
    if len(baselines) > 1:
        parts.append("## Other partial matches from Verantyx:")
        for name, pct, _ in baselines[1:3]:
            parts.append(f"  - `{name}`: {pct:.1f}% match")
        parts.append("")
    
    # test input
    for i, ex in enumerate(task['test']):
        h, w = len(ex['input']), len(ex['input'][0])
        parts.append(f"### Test {i+1} Input ({h}x{w}):")
        parts.append(grid_str(ex['input']))
        parts.append("")
    
    # SVD Expert Analysis
    if svd_text:
        parts.append(svd_text)
    
    parts.append("\nAnalyze the pattern. The verantyx baseline is CLOSE but not perfect.")
    parts.append("Understand what transformation is truly happening, then write def transform(grid).")
    
    user_prompt = '\n'.join(parts)
    
    # === V3生成ループ ===
    last_feedback = None
    for attempt in range(8):
        temp = [0.0, 0.0, 0.2, 0.3, 0.5, 0.5, 0.7, 1.0][attempt]
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        if last_feedback:
            messages.append({"role": "user", "content": last_feedback})
        
        t1 = time.time()
        response = call_v3(messages, temp=temp)
        elapsed = time.time() - t1
        
        fn, err = extract_fn(response)
        if fn is None:
            print(f"  {tid} #{attempt+1}: parse fail ({elapsed:.0f}s)", flush=True)
            last_feedback = f"Code error: {err[:100]}. Write def transform(grid). Output ```python block."
            continue
        
        # 全train検証
        all_pass = True
        fb_parts = []
        for i, ex in enumerate(task['train']):
            try:
                pred = fn(ex['input'])
                if pred is None:
                    all_pass = False
                    fb_parts.append(f"Train {i+1}: returned None")
                    continue
                if not grid_eq(pred, ex['output']):
                    all_pass = False
                    ph, pw = len(pred), len(pred[0]) if pred else 0
                    eh, ew = len(ex['output']), len(ex['output'][0])
                    if (ph, pw) == (eh, ew):
                        diffs = grid_diff(pred, ex['output'])
                        detail = '; '.join(f"({r},{c}): got {g} want {w}" for r, c, g, w in diffs[:8])
                        
                        # ベースラインとの比較
                        if i < len(best_preds):
                            base_pct = grid_match_pct(best_preds[i], ex['output'])
                            your_pct = grid_match_pct(pred, ex['output'])
                            direction = "BETTER" if your_pct > base_pct else "WORSE" if your_pct < base_pct else "SAME"
                            fb_parts.append(f"Train {i+1}: {len(diffs)} wrong. {detail}. ({direction} than baseline: {your_pct:.0f}% vs {base_pct:.0f}%)")
                        else:
                            fb_parts.append(f"Train {i+1}: {len(diffs)} wrong. {detail}")
                    else:
                        fb_parts.append(f"Train {i+1}: size {ph}x{pw} != expected {eh}x{ew}")
            except Exception as e:
                all_pass = False
                fb_parts.append(f"Train {i+1}: {type(e).__name__}: {str(e)[:80]}")
        
        if not all_pass:
            summary = '. '.join(fb_parts[:3])
            print(f"  {tid} #{attempt+1}: {summary[:80]} ({elapsed:.0f}s)", flush=True)
            last_feedback = f"Wrong: {summary}\nRemember the verantyx baseline was {best_pct:.0f}% correct. Fix your logic. Output ```python block."
            continue
        
        # Train全通過 → test実行
        test_preds = []
        for tex in task['test']:
            try:
                test_preds.append(fn(tex['input']))
            except:
                test_preds = None; break
        
        if test_preds is None:
            print(f"  {tid} #{attempt+1}: test crash ({elapsed:.0f}s)", flush=True)
            last_feedback = "Code crashed on test. Make robust. Output ```python block."
            continue
        
        test_ok = all(grid_eq(test_preds[j], task['test'][j]['output']) for j in range(len(task['test'])))
        status = "✅" if test_ok else "⚠️"
        print(f"  {status} {tid} #{attempt+1} base={best_name}({best_pct:.0f}%) t={temp} ({elapsed:.0f}s)", flush=True)
        
        return {
            "task_id": tid,
            "status": "pass" if test_ok else "train_only",
            "baseline": best_name,
            "baseline_pct": round(best_pct, 1),
            "attempt": attempt + 1,
            "test": [{"output": p} for p in test_preds],
        }
    
    return None


def main():
    tasks = sorted(f.stem for f in EVAL_DIR.glob("*.json"))
    done = set(f.stem for f in RESULTS_DIR.glob("*.json"))
    todo = [t for t in tasks if t not in done]
    
    print(f"=== Verantyx Repair Solver ===", flush=True)
    print(f"Total: {len(tasks)}, Done: {len(done)}, Todo: {len(todo)}", flush=True)
    
    stats = {'pass': 0, 'train_only': 0, 'fail': 0}
    
    for tid in todo:
        with open(EVAL_DIR / f"{tid}.json") as f:
            task = json.load(f)
        
        result = solve_with_repair(tid, task)
        if result is None:
            result = {"task_id": tid, "status": "fail"}
            print(f"  ❌ {tid}: all failed", flush=True)
        
        s = result.get('status', 'fail')
        stats[s] = stats.get(s, 0) + 1
        
        with open(RESULTS_DIR / f"{tid}.json", "w") as f:
            json.dump(result, f, indent=2)
        
        total_done = len(done) + sum(stats.values())
        print(f"  [{total_done}/{len(tasks)}] ✅{stats['pass']} ⚠️{stats['train_only']} ❌{stats['fail']}", flush=True)
    
    # 最終
    all_r = list(RESULTS_DIR.glob("*.json"))
    tp = sum(1 for f in all_r if json.load(open(f)).get("status") == "pass")
    print(f"\nFINAL: ✅ {tp}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
