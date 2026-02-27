"""
arc/abstract_nb_solver.py — Abstract neighborhood rule solver

Instead of mapping exact neighbor tuples to output colors,
learn abstract features:
1. Count of each color in neighborhood
2. Number of non-bg neighbors
3. Is cell on border of an object?
4. Relative color: same/different as center
5. Directional features: up/down/left/right neighbor colors

This allows generalization to unseen exact configurations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _g(g): return np.array(g, dtype=int)
def _l(a): return a.tolist()
def _bg(g): return int(Counter(g.flatten()).most_common(1)[0][0])


def _extract_features(grid, r, c, bg, radius=1):
    """Extract abstract features for cell (r,c)"""
    H, W = grid.shape
    center = int(grid[r, c])
    
    # Direct neighbors (4-connected)
    up = int(grid[r-1, c]) if r > 0 else -1
    down = int(grid[r+1, c]) if r < H-1 else -1
    left = int(grid[r, c-1]) if c > 0 else -1
    right = int(grid[r, c+1]) if c < W-1 else -1
    
    # Full neighborhood
    nb_colors = Counter()
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                nb_colors[int(grid[nr, nc])] += 1
    
    # Features
    n_bg = nb_colors.get(bg, 0)
    n_same = nb_colors.get(center, 0)
    n_nonbg = sum(v for k, v in nb_colors.items() if k != bg)
    n_distinct = len([k for k in nb_colors if k != bg])
    is_bg = (center == bg)
    
    # Is edge of object (non-bg cell with bg neighbor)?
    is_edge = (center != bg) and n_bg > 0
    
    # 4-direction abstract: same(1)/bg(0)/other(2)
    def _rel(v):
        if v == -1: return -1
        if v == center: return 1
        if v == bg: return 0
        return 2
    
    dir_pattern = (_rel(up), _rel(down), _rel(left), _rel(right))
    
    return (is_bg, n_nonbg, n_same, n_distinct, is_edge, dir_pattern, center)


def _extract_features_v2(grid, r, c, bg, color_map=None):
    """Simpler features: center_is_bg, count_nonbg_neighbors, center_color"""
    H, W = grid.shape
    center = int(grid[r, c])
    
    n_nonbg_4 = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != bg:
            n_nonbg_4 += 1
    
    n_same_4 = 0
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == center:
            n_same_4 += 1
    
    # Map center color to abstract ID
    if color_map:
        c_id = color_map.get(center, center)
    else:
        c_id = 0 if center == bg else 1
    
    return (c_id, n_nonbg_4, n_same_4)


def learn_abstract_nb(train_pairs):
    """Learn abstract neighborhood rule from train pairs"""
    inp0, out0 = _g(train_pairs[0][0]), _g(train_pairs[0][1])
    if inp0.shape != out0.shape:
        return None
    
    bg = _bg(inp0)
    
    # Try multiple feature levels
    for feat_fn_name in ['v1', 'v2']:
        mapping = {}
        consistent = True
        
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            if inp.shape != out.shape:
                return None
            
            H, W = inp.shape
            
            for r in range(H):
                for c in range(W):
                    if feat_fn_name == 'v1':
                        feat = _extract_features(inp, r, c, bg)
                    else:
                        feat = _extract_features_v2(inp, r, c, bg)
                    
                    out_color = int(out[r, c])
                    
                    if feat in mapping:
                        if mapping[feat] != out_color:
                            consistent = False
                            break
                    else:
                        mapping[feat] = out_color
                
                if not consistent:
                    break
            if not consistent:
                break
        
        if consistent and mapping:
            # Check that the mapping actually changes something
            has_change = False
            for feat, out_c in mapping.items():
                if feat_fn_name == 'v2':
                    in_c_id = feat[0]
                    # in_c_id 0=bg, 1=nonbg; if out_c != bg for bg input, or vice versa
                    if (in_c_id == 0 and out_c != bg) or (in_c_id == 1 and out_c == bg):
                        has_change = True
                        break
                else:
                    center = feat[-1]
                    if center != out_c:
                        has_change = True
                        break
            
            if has_change:
                return {'mapping': mapping, 'bg': bg, 'feat_fn': feat_fn_name}
    
    return None


def apply_abstract_nb(inp_g, rule):
    inp = _g(inp_g)
    bg = rule['bg']
    mapping = rule['mapping']
    feat_fn = rule['feat_fn']
    
    H, W = inp.shape
    out = inp.copy()
    
    for r in range(H):
        for c in range(W):
            if feat_fn == 'v1':
                feat = _extract_features(inp, r, c, bg)
            else:
                feat = _extract_features_v2(inp, r, c, bg)
            
            if feat in mapping:
                out[r, c] = mapping[feat]
            # else keep original
    
    return _l(out)


def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_g(pred), _g(out)):
            return False
    return True


def generate_abstract_nb_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces
    
    inp0, out0 = _g(train_pairs[0][0]), _g(train_pairs[0][1])
    if inp0.shape != out0.shape:
        return pieces
    
    try:
        rule = learn_abstract_nb(train_pairs)
        if rule is None:
            return pieces
        
        fn = lambda inp, r=rule: apply_abstract_nb(inp, r)
        if _verify(fn, train_pairs):
            pieces.append(CrossPiece(name=f"abs_nb:{rule['feat_fn']}", apply_fn=fn))
    except Exception:
        pass
    
    return pieces
