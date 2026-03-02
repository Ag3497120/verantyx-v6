"""
arc/cross6_fill.py — 6軸Cross 背景填充ソルバー

identity+addition型タスク（294問）を対象:
- 既存オブジェクトはそのまま
- 背景セルの一部が新色に変わる
- 変わるかどうかは6軸Cross記述子で決まる

ソルバー:
1. 入力→出力で変化したセルを特定
2. 変化セルのCross記述子 vs 不変セルのCross記述子を学習
3. テスト入力で変化すべきセルを予測

核心: 背景セルから見た8方向のrun length（各方向で何色が何セル見えるか）
= 背景の位置情報 = 3D位置認識
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import grid_eq, grid_shape, most_common_color


def _neighbor_colors(g: np.ndarray, r: int, c: int) -> tuple:
    """8近傍の色 (存在しない方向は-1)"""
    h, w = g.shape
    result = []
    for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append(int(g[nr, nc]))
        else:
            result.append(-1)
    return tuple(result)


def _ray_colors(g: np.ndarray, r: int, c: int, bg: int) -> tuple:
    """4方向のレイ: 各方向で最初に出会う非背景色と距離"""
    h, w = g.shape
    rays = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        r2, c2 = r + dr, c + dc
        dist = 1
        found_color = -1
        while 0 <= r2 < h and 0 <= c2 < w:
            if g[r2, c2] != bg:
                found_color = int(g[r2, c2])
                break
            r2 += dr; c2 += dc; dist += 1
        rays.append((found_color, dist))
    return tuple(rays)


def _cross_key_for_fill(g: np.ndarray, r: int, c: int, bg: int, level: str = 'full') -> tuple:
    """Fill判定用のCrossキー
    
    Levels:
    - 'full': 8近傍色 + 4方向レイ色距離
    - 'nb8': 8近傍色のみ
    - 'ray': 4方向レイのみ
    - 'ray_color': レイの色のみ（距離なし）
    - 'nb_count': 近傍の色カウント
    """
    if level == 'full':
        return _neighbor_colors(g, r, c) + _ray_colors(g, r, c, bg)
    elif level == 'nb8':
        return _neighbor_colors(g, r, c)
    elif level == 'ray':
        return _ray_colors(g, r, c, bg)
    elif level == 'ray_color':
        rays = _ray_colors(g, r, c, bg)
        return tuple(color for color, dist in rays)
    elif level == 'nb_count':
        nb = _neighbor_colors(g, r, c)
        counts = Counter(nb)
        # Sorted by color
        return tuple(sorted(counts.items()))
    else:
        return _neighbor_colors(g, r, c)


def cross6_fill_learn(train_pairs: List[Tuple[List[List[int]], List[List[int]]]]) -> Optional[Dict]:
    """背景填充ルールを学習"""
    
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    levels = ['full', 'nb8', 'ray', 'ray_color', 'nb_count']
    
    for level in levels:
        rule = {}
        consistent = True
        coverage = True
        
        for inp, out in train_pairs:
            g_in = np.array(inp)
            g_out = np.array(out)
            h, w = g_in.shape
            
            for r in range(h):
                for c in range(w):
                    in_c = int(g_in[r, c])
                    out_c = int(g_out[r, c])
                    
                    if in_c != bg:
                        # Non-bg: must stay same
                        if out_c != in_c:
                            consistent = False
                            break
                        continue
                    
                    # Background cell
                    key = _cross_key_for_fill(g_in, r, c, bg, level)
                    
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
        
        if not consistent:
            continue
        
        if not rule:
            continue
        
        # Verify: all non-bg cells unchanged + bg cells mapped correctly
        all_pass = True
        for inp, out in train_pairs:
            g_in = np.array(inp)
            g_out = np.array(out)
            h, w = g_in.shape
            
            for r in range(h):
                for c in range(w):
                    in_c = int(g_in[r, c])
                    out_c = int(g_out[r, c])
                    
                    if in_c != bg:
                        if out_c != in_c:
                            all_pass = False
                            break
                    else:
                        key = _cross_key_for_fill(g_in, r, c, bg, level)
                        pred = rule.get(key, bg)
                        if pred != out_c:
                            all_pass = False
                            break
                if not all_pass:
                    break
            if not all_pass:
                break
        
        if all_pass:
            return {'level': level, 'rule': rule, 'bg': bg}
    
    # Fallback: also try treating ALL cells (not just bg)
    for level in levels:
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
                    
                    key = (in_c,) + _cross_key_for_fill(g_in, r, c, bg, level)
                    
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
                    key = (int(g_in[r, c]),) + _cross_key_for_fill(g_in, r, c, bg, level)
                    pred = rule.get(key, int(g_in[r, c]))
                    if pred != int(g_out[r, c]):
                        all_pass = False
                        break
                if not all_pass:
                    break
            if not all_pass:
                break
        
        if all_pass:
            return {'level': 'all_' + level, 'rule': rule, 'bg': bg}
    
    return None


def cross6_fill_apply(inp: List[List[int]], learned: Dict) -> List[List[int]]:
    """学習したfillルールを適用"""
    rule = learned['rule']
    bg = learned['bg']
    level = learned['level']
    
    g_in = np.array(inp)
    h, w = g_in.shape
    result = [row[:] for row in inp]
    
    if level.startswith('all_'):
        actual_level = level[4:]
        for r in range(h):
            for c in range(w):
                key = (int(g_in[r, c]),) + _cross_key_for_fill(g_in, r, c, bg, actual_level)
                if key in rule:
                    result[r][c] = rule[key]
    else:
        for r in range(h):
            for c in range(w):
                if int(g_in[r, c]) != bg:
                    continue
                key = _cross_key_for_fill(g_in, r, c, bg, level)
                if key in rule:
                    result[r][c] = rule[key]
    
    return result


def cross6_fill_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """Fill型ソルバー"""
    learned = cross6_fill_learn(train_pairs)
    if learned is None:
        return None
    
    # Verify on train
    for inp, out in train_pairs:
        pred = cross6_fill_apply(inp, learned)
        if not grid_eq(pred, out):
            return None
    
    return cross6_fill_apply(test_input, learned)


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.cross6_fill <task.json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        task = json.load(f)
    tp = [(e['input'], e['output']) for e in task['train']]
    ti = task['test'][0]['input']
    to = task['test'][0].get('output')
    r = cross6_fill_solve(tp, ti)
    if r:
        print('✅ SOLVED!' if to and grid_eq(r, to) else '⚠️ Solution')
    else:
        print('✗ No solution')
