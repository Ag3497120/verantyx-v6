#!/usr/bin/env python3
"""
verantyx_brain.py — Verantyx 170K行フル活用 × DeepSeek V3
============================================================
Phase A: 全力分析 (cross engine, cross2, world commands, priors, structure)
Phase B: 戦略決定 (repair/compose/object/panel/colormap/symmetry/guided/scratch)
Phase C: 戦略別プロンプト生成
Phase D: V3コード生成 + train検証 + 差分フィードバック
"""
import json, os, sys, time, re, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

from brain.analyzer import full_analysis, grid_eq, grid_diff, grid_match_pct
from brain.strategy import select_strategy
from brain.prompt_builder import build_system_prompt, build_user_prompt, build_feedback

EVAL_DIR = Path("/private/tmp/arc-agi-2/data/evaluation")
RESULTS_DIR = Path(os.path.expanduser("~/verantyx_v6/brain_full_results"))
RESULTS_DIR.mkdir(exist_ok=True)

API_KEY = "sk-1c9551e705dd4fbfbdcab991cc924526"
API_URL = "https://api.deepseek.com/chat/completions"

MAX_ATTEMPTS = 6
TEMPS = [0.0, 0.0, 0.3, 0.5, 0.7, 1.0]


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
    if not text: return None, "empty response"
    m = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    code = m.group(1) if m else text
    if 'def transform' not in code:
        return None, "no def transform found"
    try:
        ns = {"__builtins__": __builtins__}
        exec(code, ns)
        return ns.get('transform'), code
    except Exception as e:
        return None, f"exec error: {e}"


def verify_train(fn, task):
    """全train例を検証。(all_pass, error_summary, detail_list)"""
    errors = []
    for i, ex in enumerate(task['train']):
        try:
            pred = fn(ex['input'])
            if pred is None:
                errors.append(f"Train {i+1}: returned None")
                continue
            if not grid_eq(pred, ex['output']):
                ph, pw = len(pred), len(pred[0]) if pred else 0
                eh, ew = len(ex['output']), len(ex['output'][0])
                if (ph, pw) != (eh, ew):
                    errors.append(f"Train {i+1}: size {ph}x{pw} != expected {eh}x{ew}")
                else:
                    diffs = grid_diff(pred, ex['output'])
                    detail = '; '.join(f"({r},{c}): got {g} want {w}" for r, c, g, w in diffs[:8])
                    errors.append(f"Train {i+1}: {len(diffs)} wrong cells. {detail}")
        except Exception as e:
            errors.append(f"Train {i+1}: {type(e).__name__}: {str(e)[:100]}")
    
    return len(errors) == 0, '. '.join(errors), errors


def solve_task(tid, task):
    """Phase A→B→C→D の統合パイプライン"""
    
    # ═══ Phase A: verantyx全力分析 ═══
    t_analysis = time.time()
    analysis = full_analysis(task, timeout_sec=30)
    analysis_time = time.time() - t_analysis
    
    # ═══ Phase B: 戦略決定 ═══
    strategy_name, strategy_data = select_strategy(analysis)
    print(f"  [{tid}] strategy={strategy_name} analysis={analysis_time:.1f}s", flush=True)
    
    # ═══ Phase C: プロンプト生成 ═══
    system_prompt = build_system_prompt(strategy_name)
    user_prompt = build_user_prompt(task, analysis, strategy_name, strategy_data)
    
    # ═══ Phase D: V3生成 + 検証ループ ═══
    best_result = None
    last_feedback = None
    
    for attempt in range(MAX_ATTEMPTS):
        temp = TEMPS[attempt] if attempt < len(TEMPS) else 1.0
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if last_feedback:
            messages.append({"role": "user", "content": last_feedback})
        
        t0 = time.time()
        response = call_v3(messages, temp=temp)
        elapsed = time.time() - t0
        
        fn, err = extract_fn(response)
        if fn is None:
            print(f"  {tid} #{attempt+1}: parse fail: {err[:50]} ({elapsed:.0f}s)", flush=True)
            last_feedback = f"Previous code error: {err[:150]}. Write valid Python with def transform(grid). Output ```python block."
            continue
        
        all_pass, error_summary, error_list = verify_train(fn, task)
        
        if not all_pass:
            short = error_summary[:80]
            print(f"  {tid} #{attempt+1}: {short} ({elapsed:.0f}s)", flush=True)
            last_feedback = build_feedback(error_summary, analysis)
            continue
        
        # Train passed! → test実行
        test_preds = []
        for tex in task['test']:
            try:
                test_preds.append(fn(tex['input']))
            except Exception as e:
                test_preds = None
                break
        
        if test_preds is None:
            print(f"  {tid} #{attempt+1}: test exec error ({elapsed:.0f}s)", flush=True)
            last_feedback = "Code crashed on test input. Make it more robust. Output ```python block."
            continue
        
        test_ok = all(grid_eq(test_preds[j], task['test'][j]['output']) for j in range(len(task['test'])))
        status = "✅" if test_ok else "⚠️"
        print(f"  {status} {tid} #{attempt+1} strategy={strategy_name} t={temp} ({elapsed:.0f}s) [analysis:{analysis_time:.1f}s]", flush=True)
        
        best_result = {
            "task_id": tid,
            "status": "pass" if test_ok else "train_only",
            "strategy": strategy_name,
            "attempt": attempt + 1,
            "analysis_time": round(analysis_time, 2),
            "test": [{"output": p} for p in test_preds],
        }
        break
    
    return best_result


def main():
    tasks = sorted(f.stem for f in EVAL_DIR.glob("*.json"))
    done = set(f.stem for f in RESULTS_DIR.glob("*.json"))
    todo = [t for t in tasks if t not in done]
    
    print(f"=== Verantyx Brain (Full) ===", flush=True)
    print(f"Total: {len(tasks)}, Done: {len(done)}, Todo: {len(todo)}", flush=True)
    
    stats = {'pass': 0, 'train_only': 0, 'fail': 0}
    
    for tid in todo:
        with open(EVAL_DIR / f"{tid}.json") as f:
            task = json.load(f)
        
        result = solve_task(tid, task)
        if result is None:
            result = {"task_id": tid, "status": "fail"}
            print(f"  ❌ {tid}: all failed", flush=True)
        
        stats[result.get('status', 'fail')] = stats.get(result.get('status', 'fail'), 0) + 1
        
        with open(RESULTS_DIR / f"{tid}.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # 途中経過
        total_done = len(done) + stats['pass'] + stats['train_only'] + stats['fail']
        print(f"  Progress: {total_done}/{len(tasks)} | ✅{stats['pass']} ⚠️{stats['train_only']} ❌{stats['fail']}", flush=True)
    
    # 最終結果
    all_r = list(RESULTS_DIR.glob("*.json"))
    tp = sum(1 for f in all_r if json.load(open(f)).get("status") == "pass")
    to = sum(1 for f in all_r if json.load(open(f)).get("status") == "train_only")
    
    print(f"\n{'='*50}", flush=True)
    print(f"FINAL: ✅ Test correct: {tp}/{len(tasks)} ({tp/len(tasks)*100:.1f}%)", flush=True)
    print(f"       ⚠️ Train only: {to}", flush=True)
    print(f"       ❌ Failed: {len(tasks) - tp - to}", flush=True)


if __name__ == "__main__":
    main()
