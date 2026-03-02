"""
arc/cross6_fill_v2.py — 過学習しないfillソルバー

v1の問題: full key = 位置依存 → 汎化しない
v2: 抽象的な特徴のみ使う

キー候補（汎化しやすい順）:
1. nb_role: 8近傍を(同色bg, 異色fg, 境界外)の3値に抽象化
2. nb_color_set: 近傍に存在する色の集合
3. fg_adj_count: 隣接する前景セル数
4. ray_first: 4方向の最初の非背景色
5. between: 2つのオブジェクトの間にあるか
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import grid_eq, grid_shape, most_common_color


def _nb_features(g: np.ndarray, r: int, c: int, bg: int) -> Dict:
    """抽象的な近傍特徴"""
    h, w = g.shape
    
    # 8近傍
    nb_colors = []
    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            nb_colors.append(int(g[nr, nc]))
        else:
            nb_colors.append(-1)  # boundary
    
    # 4方向レイ
    ray_first = []
    ray_dist = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        r2, c2 = r + dr, c + dc
        d = 1
        found = -1
        while 0 <= r2 < h and 0 <= c2 < w:
            if g[r2, c2] != bg:
                found = int(g[r2, c2])
                break
            r2 += dr; c2 += dc; d += 1
        ray_first.append(found)
        ray_dist.append(d if found >= 0 else 0)
    
    # 抽象化
    fg_adj = sum(1 for c in nb_colors if c >= 0 and c != bg)
    bg_adj = sum(1 for c in nb_colors if c == bg)
    border_adj = sum(1 for c in nb_colors if c == -1)
    
    nb_role = tuple(('F' if c >= 0 and c != bg else 'B' if c == bg else 'X') for c in nb_colors)
    nb_fg_colors = tuple(sorted(set(c for c in nb_colors if c >= 0 and c != bg)))
    
    return {
        'nb_role': nb_role,
        'nb_fg_colors': nb_fg_colors,
        'fg_adj': fg_adj,
        'ray_first': tuple(ray_first),
        'ray_color_set': tuple(sorted(set(c for c in ray_first if c >= 0))),
        'nb8': tuple(nb_colors),
        'between_h': (ray_first[2] >= 0 and ray_first[3] >= 0),  # left and right both hit fg
        'between_v': (ray_first[0] >= 0 and ray_first[1] >= 0),  # up and down both hit fg
    }


# Key generators with decreasing specificity
KEY_GENS = [
    # (name, key_fn) — key_fn takes features dict, returns tuple key
    ('nb8_role', lambda f: f['nb_role']),
    ('nb_fg_colors+role', lambda f: (f['nb_fg_colors'], f['nb_role'])),
    ('ray_first', lambda f: f['ray_first']),
    ('nb_fg_colors+fg_adj', lambda f: (f['nb_fg_colors'], f['fg_adj'])),
    ('ray_color_set+fg_adj', lambda f: (f['ray_color_set'], f['fg_adj'])),
    ('fg_adj+between', lambda f: (f['fg_adj'], f['between_h'], f['between_v'])),
    ('fg_adj', lambda f: (f['fg_adj'],)),
]


def cross6_fill_v2_learn(train_pairs):
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    for kg_name, kg_fn in KEY_GENS:
        # Two modes: bg_only (only remap bg cells), all_cells (remap any cell)
        for mode in ['bg_only', 'all_cells']:
            rule = {}
            consistent = True
            
            for inp, out in train_pairs:
                g_in = np.array(inp)
                g_out = np.array(out)
                h, w = g_in.shape
                
                for r in range(h):
                    for c in range(w):
                        in_c = int(g_in[r, c])
                        out_c = int(g_out[r, c])
                        
                        if mode == 'bg_only' and in_c != bg:
                            if out_c != in_c:
                                consistent = False
                                break
                            continue
                        
                        feats = _nb_features(g_in, r, c, bg)
                        
                        if mode == 'all_cells':
                            key = (in_c,) + kg_fn(feats)
                        else:
                            key = kg_fn(feats)
                        
                        if key in rule:
                            if rule[key] != out_c:
                                consistent = False
                                break
                        else:
                            rule[key] = out_c
                    
                    if not consistent:
                        break
                if not consistent:
                    break
            
            if not consistent or not rule:
                continue
            
            # Verify
            all_pass = True
            for inp, out in train_pairs:
                g_in = np.array(inp)
                g_out = np.array(out)
                h, w = g_in.shape
                
                for r in range(h):
                    for c in range(w):
                        in_c = int(g_in[r, c])
                        out_c = int(g_out[r, c])
                        feats = _nb_features(g_in, r, c, bg)
                        
                        if mode == 'all_cells':
                            key = (in_c,) + kg_fn(feats)
                        else:
                            if in_c != bg:
                                if in_c != out_c:
                                    all_pass = False
                                    break
                                continue
                            key = kg_fn(feats)
                        
                        pred = rule.get(key, in_c)
                        if pred != out_c:
                            all_pass = False
                            break
                    if not all_pass:
                        break
                if not all_pass:
                    break
            
            if all_pass:
                return {'kg_name': kg_name, 'kg_fn': kg_fn, 'mode': mode, 
                        'rule': rule, 'bg': bg}
    
    return None


def cross6_fill_v2_apply(inp, learned):
    rule = learned['rule']
    bg = learned['bg']
    kg_fn = learned['kg_fn']
    mode = learned['mode']
    
    g_in = np.array(inp)
    h, w = g_in.shape
    result = [row[:] for row in inp]
    
    for r in range(h):
        for c in range(w):
            in_c = int(g_in[r, c])
            if mode == 'bg_only' and in_c != bg:
                continue
            
            feats = _nb_features(g_in, r, c, bg)
            if mode == 'all_cells':
                key = (in_c,) + kg_fn(feats)
            else:
                key = kg_fn(feats)
            
            if key in rule:
                result[r][c] = rule[key]
    
    return result


def cross6_fill_v2_solve(train_pairs, test_input):
    learned = cross6_fill_v2_learn(train_pairs)
    if learned is None:
        return None
    
    # Train verify
    for inp, out in train_pairs:
        pred = cross6_fill_v2_apply(inp, learned)
        if not grid_eq(pred, out):
            return None
    
    # Leave-one-out cross-validation to reduce false positives
    if len(train_pairs) >= 3:
        for i in range(len(train_pairs)):
            loo_train = train_pairs[:i] + train_pairs[i+1:]
            loo_test_in, loo_test_out = train_pairs[i]
            
            loo_learned = cross6_fill_v2_learn(loo_train)
            if loo_learned is None:
                return None
            
            loo_pred = cross6_fill_v2_apply(loo_test_in, loo_learned)
            if not grid_eq(loo_pred, loo_test_out):
                return None
    
    return cross6_fill_v2_apply(test_input, learned)
