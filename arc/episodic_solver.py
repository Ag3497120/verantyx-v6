"""
arc/episodic_solver.py — エピソード記憶ベースのソルバー

1. training 1000問からエピソード記憶を構築
2. eval問題の断片を抽出
3. 類似エピソードを想起
4. 想起された解法（Cross Engineの操作）を試行
5. train検証 → test予測
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

from arc.episodic_memory import (
    EpisodicMemory, build_episodic_memory,
    extract_fragments, extract_delta_fragments,
    Episode
)
from arc.grid import grid_eq


# ──── 解法操作の登録 ────

OPERATION_REGISTRY = {}


def register_op(name):
    def decorator(fn):
        OPERATION_REGISTRY[name] = fn
        return fn
    return decorator


@register_op('gravity_down')
def op_gravity_down(train_pairs, test_input):
    g = np.array(test_input)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    result = np.full_like(g, bg)
    for c in range(w):
        colors = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
        for i, color in enumerate(reversed(colors)):
            result[h - 1 - i, c] = color
    return result.tolist()


@register_op('gravity_up')
def op_gravity_up(train_pairs, test_input):
    g = np.array(test_input)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    result = np.full_like(g, bg)
    for c in range(w):
        colors = [int(g[r, c]) for r in range(h) if g[r, c] != bg]
        for i, color in enumerate(colors):
            result[i, c] = color
    return result.tolist()


@register_op('symmetry_fill_hv')
def op_symmetry_fill_hv(train_pairs, test_input):
    g = np.array(test_input)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    result = g.copy()
    # 水平
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if result[r, c] == bg and result[r, mc] != bg:
                result[r, c] = result[r, mc]
    # 垂直
    for r in range(h):
        mr = h - 1 - r
        for c in range(w):
            if result[r, c] == bg and result[mr, c] != bg:
                result[r, c] = result[mr, c]
    return result.tolist()


@register_op('symmetry_fill_h')
def op_symmetry_fill_h(train_pairs, test_input):
    g = np.array(test_input)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    result = g.copy()
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if result[r, c] == bg and result[r, mc] != bg:
                result[r, c] = result[r, mc]
    return result.tolist()


@register_op('symmetry_fill_v')
def op_symmetry_fill_v(train_pairs, test_input):
    g = np.array(test_input)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    result = g.copy()
    for r in range(h):
        mr = h - 1 - r
        for c in range(w):
            if result[r, c] == bg and result[mr, c] != bg:
                result[r, c] = result[mr, c]
    return result.tolist()


@register_op('fill_enclosed')
def op_fill_enclosed(train_pairs, test_input):
    g = np.array(test_input)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    # train例からfill色を学習
    fill_color = None
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        if gi.shape != go.shape:
            return None
        bg_t = int(Counter(gi.flatten()).most_common(1)[0][0])
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                if gi[r, c] == bg_t and go[r, c] != bg_t:
                    fc = int(go[r, c])
                    if fill_color is None:
                        fill_color = fc
                    elif fill_color != fc:
                        fill_color = fc  # 最後に見つけた色
    
    if fill_color is None:
        return None
    
    # flood fill外部を検出
    visited = np.zeros((h, w), dtype=bool)
    queue = []
    for r in range(h):
        for c in [0, w-1]:
            if g[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    for c in range(w):
        for r in [0, h-1]:
            if g[r, c] == bg and not visited[r, c]:
                queue.append((r, c)); visited[r, c] = True
    
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == bg:
                visited[nr, nc] = True
                queue.append((nr, nc))
    
    result = [row[:] for row in test_input]
    for r in range(h):
        for c in range(w):
            if g[r, c] == bg and not visited[r, c]:
                result[r][c] = fill_color
    return result


@register_op('color_map')
def op_color_map(train_pairs, test_input):
    """色マッピング"""
    color_map = {}
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        if gi.shape != go.shape:
            return None
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                ic, oc = int(gi[r, c]), int(go[r, c])
                if ic in color_map:
                    if color_map[ic] != oc:
                        return None
                else:
                    color_map[ic] = oc
    
    g = np.array(test_input)
    result = g.copy()
    for ic, oc in color_map.items():
        result[g == ic] = oc
    return result.tolist()


@register_op('flip_h')
def op_flip_h(train_pairs, test_input):
    return np.fliplr(np.array(test_input)).tolist()


@register_op('flip_v')
def op_flip_v(train_pairs, test_input):
    return np.flipud(np.array(test_input)).tolist()


@register_op('rot90')
def op_rot90(train_pairs, test_input):
    return np.rot90(np.array(test_input), 1).tolist()


@register_op('rot180')
def op_rot180(train_pairs, test_input):
    return np.rot90(np.array(test_input), 2).tolist()


@register_op('transpose')
def op_transpose(train_pairs, test_input):
    return np.array(test_input).T.tolist()


@register_op('abstract_nb_recolor')
def op_abstract_nb_recolor(train_pairs, test_input):
    """抽象近傍リカラー: S/B/O パターン"""
    gi = np.array(test_input)
    h, w = gi.shape
    bg = int(Counter(gi.flatten()).most_common(1)[0][0])
    
    rules = {}
    for inp, out in train_pairs:
        gin = np.array(inp)
        gout = np.array(out)
        if gin.shape != gout.shape:
            return None
        bg_t = int(Counter(gin.flatten()).most_common(1)[0][0])
        hi, wi = gin.shape
        
        for r in range(hi):
            for c in range(wi):
                center = int(gin[r, c])
                nb = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < hi and 0 <= nc < wi:
                        v = int(gin[nr, nc])
                        if v == bg_t: nb.append('B')
                        elif v == center: nb.append('S')
                        else: nb.append('O')
                    else:
                        nb.append('X')
                
                key = ('B' if center == bg_t else 'F', tuple(nb))
                out_c = int(gout[r, c])
                
                if out_c == center: out_role = 'SAME'
                elif out_c == bg_t: out_role = 'BG'
                else: out_role = out_c
                
                if key in rules:
                    if rules[key] != out_role:
                        return None
                else:
                    rules[key] = out_role
    
    result = [row[:] for row in test_input]
    for r in range(h):
        for c in range(w):
            center = int(gi[r, c])
            nb = []
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(gi[nr, nc])
                    if v == bg: nb.append('B')
                    elif v == center: nb.append('S')
                    else: nb.append('O')
                else:
                    nb.append('X')
            
            key = ('B' if center == bg else 'F', tuple(nb))
            if key in rules:
                r2 = rules[key]
                if r2 == 'SAME': result[r][c] = center
                elif r2 == 'BG': result[r][c] = bg
                else: result[r][c] = r2
    
    return result


@register_op('panel_xor')
def op_panel_xor(train_pairs, test_input):
    """separator → panel XOR"""
    g = np.array(test_input)
    h, w = g.shape
    bg = int(Counter(g.flatten()).most_common(1)[0][0])
    
    # 水平separator検出
    sep_rows = []
    for r in range(h):
        vals = set(int(v) for v in g[r])
        if len(vals) == 1 and vals.pop() != bg:
            sep_rows.append(r)
    
    # 垂直separator検出
    sep_cols = []
    for c in range(w):
        vals = set(int(v) for v in g[:, c])
        if len(vals) == 1 and vals.pop() != bg:
            sep_cols.append(c)
    
    if not sep_rows and not sep_cols:
        return None
    
    # パネル抽出
    panels = []
    row_bounds = [0] + sep_rows + [h]
    col_bounds = [0] + sep_cols + [w]
    
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r1, r2 = row_bounds[i], row_bounds[i+1]
            c1, c2 = col_bounds[j], col_bounds[j+1]
            if r1 in sep_rows: r1 += 1
            if c1 in sep_cols: c1 += 1
            if r1 < r2 and c1 < c2:
                panels.append(g[r1:r2, c1:c2])
    
    if len(panels) < 2:
        return None
    
    # XOR: panels[0] XOR panels[1] (bg=keep, fg=flip)
    p0, p1 = panels[0], panels[1]
    if p0.shape != p1.shape:
        return None
    
    result = np.full_like(p0, bg)
    for r in range(p0.shape[0]):
        for c in range(p0.shape[1]):
            v0 = p0[r, c] != bg
            v1 = p1[r, c] != bg
            if v0 != v1:
                result[r, c] = p0[r, c] if v0 else p1[r, c]
    
    return result.tolist()


# ──── メイン: 解法名→操作のマッピング ────

METHOD_TO_OPS = {
    'neighborhood_rule': ['abstract_nb_recolor'],
    'fill_enclosed': ['fill_enclosed'],
    'colormap': ['color_map'],
    'symmetry_fill': ['symmetry_fill_hv', 'symmetry_fill_h', 'symmetry_fill_v'],
    'flip_h': ['flip_h'],
    'flip_v': ['flip_v'],
    'rot90': ['rot90'],
    'rot180': ['rot180'],
    'corners_mirror': ['symmetry_fill_hv'],
    'self_tile': [],  # 複雑すぎ
    'gravity': ['gravity_down', 'gravity_up'],
}


def episodic_solve(memory: EpisodicMemory, train_pairs, test_input):
    """エピソード記憶から想起→解法実行"""
    
    # 断片抽出
    query_frags = set()
    for inp, out in train_pairs:
        query_frags |= extract_fragments(inp)
        query_frags |= extract_delta_fragments(inp, out)
    query_frags |= extract_fragments(test_input)
    
    # 想起
    recalled = memory.recall(query_frags, top_k=10)
    
    # 想起された解法を優先度順に試行
    tried_ops = set()
    
    for ep, score in recalled:
        if not ep.solution_method:
            continue
        
        # 解法名から操作候補を取得
        method = ep.solution_method
        
        # 部分マッチ
        ops = []
        for key, op_list in METHOD_TO_OPS.items():
            if key in method:
                ops.extend(op_list)
        
        # 全登録操作もフォールバック
        if not ops:
            ops = list(OPERATION_REGISTRY.keys())
        
        for op_name in ops:
            if op_name in tried_ops:
                continue
            tried_ops.add(op_name)
            
            op_fn = OPERATION_REGISTRY.get(op_name)
            if not op_fn:
                continue
            
            try:
                # train検証
                ok = True
                for inp, out in train_pairs:
                    pred = op_fn(train_pairs, inp)
                    if pred is None or not grid_eq(pred, out):
                        ok = False
                        break
                
                if ok:
                    result = op_fn(train_pairs, test_input)
                    if result is not None:
                        return result
            except Exception:
                continue
    
    # 想起なしフォールバック: 全操作を試行
    for op_name, op_fn in OPERATION_REGISTRY.items():
        if op_name in tried_ops:
            continue
        try:
            ok = True
            for inp, out in train_pairs:
                pred = op_fn(train_pairs, inp)
                if pred is None or not grid_eq(pred, out):
                    ok = False
                    break
            if ok:
                result = op_fn(train_pairs, test_input)
                if result is not None:
                    return result
        except:
            continue
    
    return None


# ──── CLI ────

if __name__ == "__main__":
    import sys, time, re
    
    if sys.argv[1] in ('--eval', '--train'):
        split = 'evaluation' if sys.argv[1] == '--eval' else 'training'
        data_dir = Path(f'/tmp/arc-agi-2/data/{split}')
        
        print("Building episodic memory...")
        memory = build_episodic_memory(
            '/tmp/arc-agi-2/data/training',
            solution_log='arc_cross_engine_v9.log'
        )
        print(f"  {len(memory.episodes)} episodes, {len(memory.fragment_index)} fragments")
        
        # 既存CE
        existing = set()
        try:
            with open('arc_cross_engine_v9.log') as f:
                for line in f:
                    m = re.search(r'✓.*?([0-9a-f]{8})', line)
                    if m: existing.add(m.group(1))
        except: pass
        
        solved = []
        for tf in sorted(data_dir.glob('*.json')):
            tid = tf.stem
            with open(tf) as f: task = json.load(f)
            tp = [(e['input'], e['output']) for e in task['train']]
            ti, to = task['test'][0]['input'], task['test'][0].get('output')
            
            r = episodic_solve(memory, tp, ti)
            if r and to and grid_eq(r, to):
                solved.append(tid)
                tag = 'NEW' if tid not in existing else ''
                print(f'  ✓ {tid} {tag}')
        
        total = len(list(data_dir.glob('*.json')))
        new = [t for t in solved if t not in existing]
        print(f'\n{split}: {len(solved)}/{total} (NEW: {len(new)})')
