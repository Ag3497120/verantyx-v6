#!/usr/bin/env python3
"""1問だけ解く（サブプロセス用）。signal不使用、time.time()ベース。"""
import json, sys, os, time
sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

task_path = sys.argv[1]
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 40

with open(task_path) as f:
    task = json.load(f)

train_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
test_inputs = [t["input"] for t in task["test"]]

from arc.grid import grid_eq
from collections import Counter

solutions = []
t0 = time.time()
deadline = t0 + timeout

def expired(): return time.time() > deadline

def verify_all(fn, pairs):
    for inp, exp in pairs:
        try:
            r = fn(inp)
            if r is None or not grid_eq(r, exp): return False
        except: return False
    return True

def predict(fn, inputs):
    preds = []
    for ti in inputs:
        try:
            p = fn(ti)
            if p is None: return None
            preds.append(p)
        except: return None
    return preds

# Phase A: Cross Engine
if not expired():
    try:
        from arc.cross_engine import solve_cross_engine
        _, verified = solve_cross_engine(train_pairs, test_inputs)
        for tag, piece in verified:
            name = getattr(piece, 'name', type(piece).__name__)
            try:
                preds = [piece.apply(ti) for ti in test_inputs]
                if all(p is not None for p in preds):
                    solutions.append({"name": f"cross:{name}", "preds": preds, "src": "cross", "conf": 0.95})
            except: pass
    except: pass

# Phase B: World Commands
if not expired():
    try:
        from arc.world_commands import build_all_commands
        world_cmds = build_all_commands(train_pairs)
        inp0, out0 = train_pairs[0]
        
        # D1: train[0]フィルタ → 全train検証
        d1_candidates = []
        for name, fn in world_cmds:
            if expired(): break
            try:
                r = fn(inp0)
                if r is not None and grid_eq(r, out0):
                    d1_candidates.append((name, fn))
            except: pass
        
        for name, fn in d1_candidates:
            if expired(): break
            if verify_all(fn, train_pairs):
                preds = predict(fn, test_inputs)
                if preds:
                    solutions.append({"name": name, "preds": preds, "src": "world_d1", "conf": 1.0})
        
        # Converge: train[0]フィルタ (サイズ爆発ガード付き)
        size_changing = {"repeat_2x2","repeat_3x3","up_2x","up_3x","up_h2x","up_v2x",
                         "tile_2x2","tile_3x3","tile_2x1","tile_1x2","down_2x","down_3x",
                         "stack_v","stack_h","sort_objs_size"}
        for name, fn in world_cmds:
            if expired(): break
            if name in size_changing: continue
            cur = inp0
            for _ in range(20):
                try: nxt = fn(cur)
                except: cur = None; break
                if nxt is None: cur = None; break
                if isinstance(nxt, list) and len(nxt) > 100: cur = None; break
                if grid_eq(cur, nxt): break
                cur = nxt
            if cur is None or not grid_eq(cur, out0): continue
            def mk_conv(f):
                def c(g):
                    x=g
                    for _ in range(20):
                        try: n=f(x)
                        except: return None
                        if n is None: return None
                        if grid_eq(x,n): return x
                        x=n
                    return x
                return c
            cfn = mk_conv(fn)
            if verify_all(cfn, train_pairs):
                preds = predict(cfn, test_inputs)
                if preds:
                    solutions.append({"name": f"conv({name})", "preds": preds, "src": "conv", "conf": 0.9})
        
        # D2: train[0]事前フィルタ
        if not expired():
            active = []
            mid_cache = {}
            for i, (name, fn) in enumerate(world_cmds):
                try:
                    mid = fn(inp0)
                    if mid is not None and isinstance(mid, list) and len(mid)>0 and not grid_eq(mid, inp0):
                        active.append((i, name, fn))
                        mid_cache[i] = mid
                except: pass
            active = active[:20]
            
            for i, n1, f1 in active:
                if expired(): break
                mid = mid_cache[i]
                for n2, f2 in world_cmds:
                    if expired(): break
                    try:
                        r2 = f2(mid)
                        if r2 is None or not grid_eq(r2, out0): continue
                    except: continue
                    def mk_pipe(a, b):
                        def fn(g):
                            m=a(g); return b(m) if m else None
                        return fn
                    pipe = mk_pipe(f1, f2)
                    if verify_all(pipe, train_pairs):
                        preds = predict(pipe, test_inputs)
                        if preds:
                            solutions.append({"name": f"{n1}→{n2}", "preds": preds, "src": "world_d2", "conf": 0.8})
    except: pass

# Phase C: Puzzle Language
if not expired():
    try:
        from arc.puzzle_lang import synthesize_programs
        programs = synthesize_programs(train_pairs)
        for prog in programs:
            if expired(): break
            if verify_all(prog.apply_fn, train_pairs):
                preds = predict(prog.apply_fn, test_inputs)
                if preds:
                    solutions.append({"name": f"puzzle:{prog.name}", "preds": preds, "src": "puzzle", "conf": 0.9})
    except: pass

elapsed = time.time() - t0

if not solutions:
    print(json.dumps({"status": "fail", "elapsed": round(elapsed,1)}))
else:
    vote = Counter()
    vote_map = {}
    for sol in solutions:
        key = json.dumps(sol["preds"][0])
        vote[key] += sol["conf"]
        if key not in vote_map or sol["conf"] > vote_map[key]["conf"]:
            vote_map[key] = sol
    best_key, _ = vote.most_common(1)[0]
    winner = vote_map[best_key]
    print(json.dumps({
        "status": "solved",
        "piece": winner["name"],
        "method": winner["src"],
        "n_solutions": len(solutions),
        "elapsed": round(elapsed,1),
        "test": [{"output": p} for p in winner["preds"]]
    }))
