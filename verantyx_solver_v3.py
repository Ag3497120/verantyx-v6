#!/usr/bin/env python3
"""
Verantyx ARC Solver v3 — Abstraction + Program Search
=====================================================
v2からの改善:
1. 色の抽象化 (color role mapping)
2. プリミティブの体系的組み合わせ探索
3. 入出力の構造分析によるルール絞り込み
4. オブジェクトレベル変換の強化
"""

import json, os, sys, time, gc
from collections import Counter
from itertools import combinations
import argparse

sys.path.insert(0, os.path.expanduser("~/verantyx_v6"))

EVAL_DIR = "/private/tmp/arc-agi-2/data/evaluation"
TRAIN_DIR = "/private/tmp/arc-agi-2/data/training"
RESULT_DIR = os.path.expanduser("~/verantyx_v6/verantyx_v3_results")
os.makedirs(RESULT_DIR, exist_ok=True)

from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors

# ═══════════════════════════════════════
# 構造分析: 入出力の関係を分析してルール絞り込み
# ═══════════════════════════════════════

def analyze_task(train_pairs):
    """Analyze structural relationships between inputs and outputs."""
    props = {}
    
    sizes = [(grid_shape(i), grid_shape(o)) for i, o in train_pairs]
    
    # Size relationship
    props["same_size"] = all(si == so for si, so in sizes)
    props["output_smaller"] = all(so[0] <= si[0] and so[1] <= si[1] for si, so in sizes)
    props["output_larger"] = all(so[0] >= si[0] and so[1] >= si[1] for si, so in sizes)
    
    # Fixed output size
    out_sizes = set(so for _, so in sizes)
    props["fixed_output_size"] = len(out_sizes) == 1
    
    # Size ratios
    if not props["same_size"]:
        ratios = set()
        for (ih, iw), (oh, ow) in sizes:
            if ih > 0 and iw > 0:
                ratios.add((round(oh/ih, 2), round(ow/iw, 2)))
        props["fixed_ratio"] = len(ratios) == 1
        if props["fixed_ratio"]:
            props["ratio"] = ratios.pop()
    
    # Color analysis
    in_colors_list = []
    out_colors_list = []
    for inp, out in train_pairs:
        ic = set()
        for row in inp: ic.update(row)
        oc = set()
        for row in out: oc.update(row)
        in_colors_list.append(ic)
        out_colors_list.append(oc)
    
    props["colors_preserved"] = all(oc == ic for oc, ic in zip(out_colors_list, in_colors_list))
    props["colors_reduced"] = all(oc < ic for oc, ic in zip(out_colors_list, in_colors_list))
    props["bg_color"] = most_common_color(train_pairs[0][0])
    
    return props

# ═══════════════════════════════════════
# 色の抽象化
# ═══════════════════════════════════════

def abstract_color_map(grid, bg):
    """Map concrete colors to abstract roles: bg=0, colors by frequency."""
    from collections import Counter
    counts = Counter()
    for row in grid:
        counts.update(row)
    
    # Remove bg
    if bg in counts:
        del counts[bg]
    
    # Sort by frequency (most common = role 1)
    sorted_colors = sorted(counts.keys(), key=lambda c: (-counts[c], c))
    cmap = {bg: 0}
    for i, c in enumerate(sorted_colors):
        cmap[c] = i + 1
    return cmap

def apply_color_map(grid, cmap):
    """Apply color mapping to grid."""
    return [[cmap.get(c, c) for c in row] for row in grid]

def invert_color_map(grid, cmap):
    """Invert color mapping."""
    inv = {v: k for k, v in cmap.items()}
    return [[inv.get(c, c) for c in row] for row in grid]

# ═══════════════════════════════════════
# プリミティブ定義（汎用変換関数）
# ═══════════════════════════════════════

def make_primitives(props, train_pairs):
    """Generate transformation primitives based on task properties."""
    primitives = []
    bg = props.get("bg_color", 0)
    
    # --- Identity & basic transforms ---
    primitives.append(("identity", lambda g: [row[:] for row in g]))
    primitives.append(("rot90", lambda g: [list(r) for r in zip(*g[::-1])]))
    primitives.append(("rot180", lambda g: [row[::-1] for row in g[::-1]]))
    primitives.append(("rot270", lambda g: [list(r) for r in zip(*[row[::-1] for row in g])]))
    primitives.append(("flip_h", lambda g: [row[::-1] for row in g]))
    primitives.append(("flip_v", lambda g: g[::-1]))
    primitives.append(("transpose", lambda g: [list(r) for r in zip(*g)]))
    
    # --- Crop to bounding box of non-bg ---
    def crop_to_content(g, background=bg):
        h, w = len(g), len(g[0])
        rows = [r for r in range(h) if any(g[r][c] != background for c in range(w))]
        cols = [c for c in range(w) if any(g[r][c] != background for r in range(h))]
        if not rows or not cols:
            return g
        return [g[r][min(cols):max(cols)+1] for r in range(min(rows), max(rows)+1)]
    primitives.append(("crop", lambda g: crop_to_content(g)))
    
    # --- Remove bg rows/cols ---
    def remove_bg_rows(g, background=bg):
        return [row for row in g if any(c != background for c in row)]
    def remove_bg_cols(g, background=bg):
        t = [list(r) for r in zip(*g)]
        t = [col for col in t if any(c != background for c in col)]
        if not t: return g
        return [list(r) for r in zip(*t)]
    primitives.append(("remove_bg_rows", lambda g: remove_bg_rows(g)))
    primitives.append(("remove_bg_cols", lambda g: remove_bg_cols(g)))
    
    # --- Color swaps (learn from first example) ---
    inp0, out0 = train_pairs[0]
    if grid_shape(inp0) == grid_shape(out0):
        h, w = grid_shape(inp0)
        # Detect color mapping
        color_map = {}
        consistent = True
        for r in range(h):
            for c in range(w):
                ic, oc = inp0[r][c], out0[r][c]
                if ic in color_map:
                    if color_map[ic] != oc:
                        consistent = False
                        break
                else:
                    color_map[ic] = oc
            if not consistent:
                break
        
        if consistent and color_map:
            cm = dict(color_map)
            primitives.append(("color_map", lambda g, m=cm: [[m.get(c, c) for c in row] for row in g]))
    
    # --- Gravity (drop non-bg cells down) ---
    def gravity_down(g, background=bg):
        h, w = len(g), len(g[0])
        result = [[background]*w for _ in range(h)]
        for c in range(w):
            non_bg = [g[r][c] for r in range(h) if g[r][c] != background]
            for i, v in enumerate(non_bg):
                result[h - len(non_bg) + i][c] = v
        return result
    primitives.append(("gravity_down", lambda g: gravity_down(g)))
    
    def gravity_left(g, background=bg):
        h, w = len(g), len(g[0])
        result = [[background]*w for _ in range(h)]
        for r in range(h):
            non_bg = [g[r][c] for c in range(w) if g[r][c] != background]
            for i, v in enumerate(non_bg):
                result[r][i] = v
        return result
    primitives.append(("gravity_left", lambda g: gravity_left(g)))
    
    # --- Fill enclosed regions ---
    def fill_enclosed(g, background=bg):
        h, w = len(g), len(g[0])
        result = [row[:] for row in g]
        # BFS from edges to find exterior bg cells
        visited = [[False]*w for _ in range(h)]
        queue = []
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h-1 or c == 0 or c == w-1) and g[r][c] == background:
                    queue.append((r, c))
                    visited[r][c] = True
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == background:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        # Fill interior bg cells with most common non-bg neighbor
        for r in range(h):
            for c in range(w):
                if g[r][c] == background and not visited[r][c]:
                    neighbors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != background:
                            neighbors.append(g[nr][nc])
                    if neighbors:
                        result[r][c] = Counter(neighbors).most_common(1)[0][0]
                    else:
                        result[r][c] = background  # keep as is
        return result
    primitives.append(("fill_enclosed", lambda g: fill_enclosed(g)))
    
    # --- Most common color replacement ---
    def replace_minority(g, background=bg):
        """Replace least common non-bg color with most common non-bg color."""
        counts = Counter()
        for row in g:
            for c in row:
                if c != background:
                    counts[c] += 1
        if len(counts) < 2:
            return g
        most = counts.most_common(1)[0][0]
        least = counts.most_common()[-1][0]
        return [[most if c == least else c for c in row] for row in g]
    primitives.append(("replace_minority", lambda g: replace_minority(g)))
    
    # --- Upscale 2x, 3x ---
    for scale in [2, 3]:
        def make_upscale(s):
            def fn(g):
                return [[g[r//s][c//s] for c in range(len(g[0])*s)] for r in range(len(g)*s)]
            return fn
        primitives.append((f"upscale_{scale}x", make_upscale(scale)))
    
    # --- Downscale (majority vote) ---
    for scale in [2, 3]:
        def make_downscale(s):
            def fn(g):
                h, w = len(g), len(g[0])
                nh, nw = h // s, w // s
                if nh == 0 or nw == 0:
                    return None
                result = []
                for r in range(nh):
                    row = []
                    for c in range(nw):
                        block = [g[r*s+dr][c*s+dc] for dr in range(s) for dc in range(s) if r*s+dr < h and c*s+dc < w]
                        row.append(Counter(block).most_common(1)[0][0])
                    result.append(row)
                return result
            return fn
        primitives.append((f"downscale_{scale}x", make_downscale(scale)))
    
    return primitives

# ═══════════════════════════════════════
# 2-Step Composition (controlled)
# ═══════════════════════════════════════

def try_compositions(primitives, train_pairs, max_first=10):
    """Try 2-step compositions. Only compose primitives that change the grid."""
    # Find primitives that produce valid (non-None) outputs
    active = []
    inp0 = train_pairs[0][0]
    for name, fn in primitives:
        try:
            r = fn(inp0)
            if r is not None and r != inp0:  # must change something
                active.append((name, fn))
        except:
            pass
    
    active = active[:max_first]  # limit
    compositions = []
    
    for n1, f1 in active:
        for n2, f2 in primitives:
            if n1 == n2:
                continue
            def make_comp(a, b):
                def fn(g):
                    mid = a(g)
                    if mid is None: return None
                    return b(mid)
                return fn
            compositions.append((f"{n1}→{n2}", make_comp(f1, f2)))
    
    return compositions

# ═══════════════════════════════════════
# Validation
# ═══════════════════════════════════════

def validate(name, fn, train_pairs):
    """Check if fn passes all training examples."""
    for inp, out in train_pairs:
        try:
            r = fn(inp)
            if r is None or not grid_eq(r, out):
                return False
        except:
            return False
    return True

# ═══════════════════════════════════════
# Main Solver
# ═══════════════════════════════════════

def solve_task(task_id, data_dir):
    result_path = os.path.join(RESULT_DIR, f"{task_id}.json")
    if os.path.exists(result_path):
        return "skip"
    
    task_path = os.path.join(data_dir, f"{task_id}.json")
    if not os.path.exists(task_path):
        return "missing"
    
    with open(task_path) as f:
        task = json.load(f)
    
    train_pairs = [(ex["input"], ex["output"]) for ex in task["train"]]
    test_inputs = [t["input"] for t in task["test"]]
    
    t0 = time.time()
    props = analyze_task(train_pairs)
    
    # ── Phase 1: Cross Engine (existing) ──
    cross_valid = []
    try:
        from arc.cross_engine import solve_cross_engine
        _, verified = solve_cross_engine(train_pairs, test_inputs)
        for tag, piece in verified:
            if validate("cross:" + piece.name, piece.apply, train_pairs):
                cross_valid.append(("cross:" + piece.name, piece.apply))
    except:
        pass
    
    # ── Phase 2: Puzzle Language ──
    puzzle_valid = []
    try:
        from arc.puzzle_lang import synthesize_programs
        programs = synthesize_programs(train_pairs)
        for prog in programs:
            if validate("puzzle:" + prog.name, prog.apply_fn, train_pairs):
                puzzle_valid.append(("puzzle:" + prog.name, prog.apply_fn))
    except:
        pass
    
    # ── Phase 3: Primitives ──
    prim_valid = []
    primitives = make_primitives(props, train_pairs)
    for name, fn in primitives:
        if validate(name, fn, train_pairs):
            prim_valid.append((name, fn))
    
    # ── Phase 4: 2-Step Compositions ──
    comp_valid = []
    try:
        compositions = try_compositions(primitives, train_pairs, max_first=8)
        for name, fn in compositions:
            if validate(name, fn, train_pairs):
                comp_valid.append((name, fn))
    except:
        pass
    
    all_valid = cross_valid + puzzle_valid + prim_valid + comp_valid
    elapsed = time.time() - t0
    
    if not all_valid:
        print(f"[{task_id}] ❌ | {elapsed:.1f}s")
        gc.collect()
        return "fail"
    
    # ── Majority Vote ──
    vote_counter = Counter()
    vote_map = {}
    for name, fn in all_valid:
        try:
            preds = [fn(ti) for ti in test_inputs]
            if any(p is None for p in preds):
                continue
            key = json.dumps(preds[0])
            vote_counter[key] += 1
            if key not in vote_map:
                vote_map[key] = (name, preds)
        except:
            pass
    
    if not vote_counter:
        print(f"[{task_id}] ❌ no predictions | {elapsed:.1f}s")
        gc.collect()
        return "fail"
    
    best_key, best_votes = vote_counter.most_common(1)[0]
    total_voters = sum(vote_counter.values())
    n_unique = len(vote_counter)
    winner_name, winner_preds = vote_map[best_key]
    
    # Confidence: high if many agree, low if split
    confidence = best_votes / total_voters if total_voters > 0 else 0
    
    result = {
        "method": "v3_solver",
        "piece": winner_name,
        "votes": best_votes,
        "total_voters": total_voters,
        "unique_outputs": n_unique,
        "confidence": round(confidence, 2),
        "sources": {
            "cross": len(cross_valid),
            "puzzle": len(puzzle_valid),
            "prim": len(prim_valid),
            "comp": len(comp_valid),
        },
        "elapsed_s": round(elapsed, 1),
        "test": [{"output": winner_preds[i]} for i in range(len(winner_preds))]
    }
    with open(result_path, 'w') as f:
        json.dump(result, f)
    
    src = f"cr={len(cross_valid)} pz={len(puzzle_valid)} pr={len(prim_valid)} co={len(comp_valid)}"
    print(f"[{task_id}] ✅ {winner_name} | {best_votes}/{total_voters} votes ({n_unique} unique) | {src} | {elapsed:.1f}s")
    gc.collect()
    return "solved"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()
    
    data_dir = EVAL_DIR if args.split == "evaluation" else TRAIN_DIR
    
    if args.task:
        task_ids = [args.task]
    else:
        task_ids = sorted([f.replace('.json','') for f in os.listdir(data_dir) if f.endswith('.json')])
        end = args.end or len(task_ids)
        task_ids = task_ids[args.start:end]
    
    print(f"{'='*65}")
    print(f"  Verantyx ARC Solver v3")
    print(f"  Cross Engine + Puzzle Lang + Primitives + Composition")
    print(f"  Structural Analysis + Majority Voting")
    print(f"{'='*65}")
    print(f"Split: {args.split}, Tasks: {len(task_ids)}\n")
    
    stats = {"solved": 0, "fail": 0, "skip": 0, "error": 0, "missing": 0}
    t_start = time.time()
    
    for tid in task_ids:
        status = solve_task(tid, data_dir)
        stats[status] = stats.get(status, 0) + 1
    
    elapsed_total = time.time() - t_start
    
    print(f"\n{'='*65}")
    print(f"Solved: {stats['solved']}, Failed: {stats['fail']}")
    print(f"Time: {elapsed_total:.0f}s ({elapsed_total/max(len(task_ids),1):.1f}s/task)\n")
    
    passed = []
    attempted = 0
    for f in sorted(os.listdir(RESULT_DIR)):
        if not f.endswith('.json'): continue
        tid = f.replace('.json','')
        tp = os.path.join(data_dir, f"{tid}.json")
        if not os.path.exists(tp): continue
        with open(tp) as tf: task = json.load(tf)
        with open(os.path.join(RESULT_DIR, f)) as rf: result = json.load(rf)
        attempted += 1
        ok = all(
            i < len(result.get('test',[])) and result['test'][i].get('output') == t['output']
            for i, t in enumerate(task['test'])
        )
        if ok:
            passed.append(tid)
            r = result
            print(f"  ✅ {tid} | {r.get('piece','?')} | conf={r.get('confidence',0)} | {r.get('votes',0)}/{r.get('total_voters',0)} votes")
    
    print(f"\n  Score: {len(passed)}/{len(task_ids)} ({100*len(passed)/len(task_ids):.1f}%)")
    print(f"  Attempted: {attempted}/{len(task_ids)}")

if __name__ == "__main__":
    main()
