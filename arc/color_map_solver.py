"""
arc/color_map_solver.py — Global/local color mapping solver

Strategies:
1. Global color map: simple color → color replacement
2. Majority color per region
3. Color by object property (size, position, neighbor count)
4. Swap two colors
5. Replace color based on row/col position
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
from scipy.ndimage import label as connected_components
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _g(g): return np.array(g, dtype=int)
def _l(a): return a.tolist()
def _bg(g): return int(Counter(g.flatten()).most_common(1)[0][0])


# === Strategy 1: Simple global color map ===

def learn_global_color_map(train_pairs):
    """Each color in input maps to exactly one color in output"""
    color_map = {}
    
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape:
            return None
        
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                ic = int(inp[r, c])
                oc = int(out[r, c])
                if ic in color_map:
                    if color_map[ic] != oc:
                        return None
                else:
                    color_map[ic] = oc
    
    # Must actually change something
    if all(k == v for k, v in color_map.items()):
        return None
    
    return color_map


def apply_global_color_map(inp_g, cmap):
    inp = _g(inp_g)
    out = inp.copy()
    for old_c, new_c in cmap.items():
        out[inp == old_c] = new_c
    return _l(out)


# === Strategy 2: Swap two colors ===

def learn_color_swap(train_pairs):
    """Two specific colors are swapped"""
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape:
            return None
        
        changes = set()
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                if inp[r,c] != out[r,c]:
                    changes.add((int(inp[r,c]), int(out[r,c])))
        
        if len(changes) != 2:
            return None
        
        (a1, b1), (a2, b2) = changes
        if a1 == b2 and a2 == b1:
            return {'swap': (a1, a2)}
        return None
    
    return None


def apply_color_swap(inp_g, rule):
    inp = _g(inp_g)
    out = inp.copy()
    a, b = rule['swap']
    mask_a = (inp == a)
    mask_b = (inp == b)
    out[mask_a] = b
    out[mask_b] = a
    return _l(out)


# === Strategy 3: Recolor objects by size rank ===

def learn_recolor_by_size(train_pairs):
    """Each object gets a color based on its size rank"""
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape:
            return None
        
        bg = _bg(inp)
        mask = (inp != bg).astype(int)
        labeled, n = connected_components(mask)
        
        if n < 2:
            return None
        
        # Get sizes and output colors per object
        obj_data = []
        for i in range(1, n+1):
            cells = np.where(labeled == i)
            area = len(cells[0])
            # What color does this object become in output?
            out_colors = Counter(int(out[r,c]) for r,c in zip(*cells))
            main_out_color = out_colors.most_common(1)[0][0]
            obj_data.append((area, main_out_color, i))
        
        # Sort by size, check if output color follows rank
        sorted_objs = sorted(obj_data, key=lambda x: x[0])
        rank_colors = [o[1] for o in sorted_objs]
        
        # All same color? not useful
        if len(set(rank_colors)) <= 1:
            return None
    
    # Verify consistency across all train pairs
    # Learn the rank→color mapping from first pair
    inp0, out0 = _g(train_pairs[0][0]), _g(train_pairs[0][1])
    bg = _bg(inp0)
    mask = (inp0 != bg).astype(int)
    labeled, n = connected_components(mask)
    
    obj_data = []
    for i in range(1, n+1):
        cells = np.where(labeled == i)
        area = len(cells[0])
        out_colors = Counter(int(out0[r,c]) for r,c in zip(*cells))
        main_out_color = out_colors.most_common(1)[0][0]
        obj_data.append((area, main_out_color))
    
    sorted_objs = sorted(obj_data, key=lambda x: x[0])
    rank_map = {i: o[1] for i, o in enumerate(sorted_objs)}
    
    return {'rank_map': rank_map, 'bg': bg}


def apply_recolor_by_size(inp_g, rule):
    inp = _g(inp_g)
    bg = rule['bg']
    mask = (inp != bg).astype(int)
    labeled, n = connected_components(mask)
    
    out = inp.copy()
    
    obj_sizes = []
    for i in range(1, n+1):
        cells = np.where(labeled == i)
        area = len(cells[0])
        obj_sizes.append((area, i))
    
    sorted_objs = sorted(obj_sizes, key=lambda x: x[0])
    
    for rank, (area, obj_id) in enumerate(sorted_objs):
        if rank in rule['rank_map']:
            cells = np.where(labeled == obj_id)
            out[cells] = rule['rank_map'][rank]
    
    return _l(out)


# === Strategy 4: Replace minority color with majority neighbor color ===

def learn_minority_to_majority(train_pairs):
    """Cells with rare colors get replaced by most common neighbor color"""
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape: return None
        
        H, W = inp.shape
        color_counts = Counter(int(inp[r,c]) for r in range(H) for c in range(W))
        
        # Find which color gets replaced
        changed = {}
        for r in range(H):
            for c in range(W):
                if inp[r,c] != out[r,c]:
                    old_c = int(inp[r,c])
                    new_c = int(out[r,c])
                    # new_c should be the most common neighbor
                    nbs = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W:
                            nbs.append(int(inp[nr, nc]))
                    if nbs:
                        most_common_nb = Counter(nbs).most_common(1)[0][0]
                        if most_common_nb != new_c:
                            return None
                    changed[old_c] = True
        
        if not changed:
            return None
    
    # Learn which colors to replace
    replace_colors = set(changed.keys())
    return {'replace_colors': list(replace_colors)}


def apply_minority_to_majority(inp_g, rule):
    inp = _g(inp_g)
    out = inp.copy()
    H, W = inp.shape
    
    for r in range(H):
        for c in range(W):
            if int(inp[r,c]) in rule['replace_colors']:
                nbs = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        nbs.append(int(inp[nr, nc]))
                if nbs:
                    out[r, c] = Counter(nbs).most_common(1)[0][0]
    
    return _l(out)


# === Strategy 5: Fill enclosed bg regions with surrounding color ===

def learn_fill_enclosed(train_pairs):
    """Fill enclosed background regions with the color that surrounds them"""
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if inp.shape != out.shape: return None
        
        bg = _bg(inp)
        H, W = inp.shape
        
        # Find bg regions
        bg_mask = (inp == bg).astype(int)
        labeled, n = connected_components(bg_mask)
        
        for i in range(1, n+1):
            cells = np.where(labeled == i)
            rows, cols = cells
            
            # Is this region touching the border?
            touches_border = (rows.min() == 0 or rows.max() == H-1 or 
                            cols.min() == 0 or cols.max() == W-1)
            
            if touches_border:
                # Should stay bg
                for r, c in zip(rows, cols):
                    if out[r, c] != bg:
                        return None  # border region changed -> not this pattern
            else:
                # Should be filled — find surrounding color
                surround_colors = Counter()
                for r, c in zip(rows, cols):
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] != bg:
                            surround_colors[int(inp[nr, nc])] += 1
                
                if not surround_colors:
                    continue
                
                fill_color = surround_colors.most_common(1)[0][0]
                
                for r, c in zip(rows, cols):
                    if int(out[r, c]) != fill_color:
                        return None
    
    return {'bg': bg}


def apply_fill_enclosed(inp_g, rule):
    inp = _g(inp_g)
    bg = rule['bg']
    out = inp.copy()
    H, W = inp.shape
    
    bg_mask = (inp == bg).astype(int)
    labeled, n = connected_components(bg_mask)
    
    for i in range(1, n+1):
        cells = np.where(labeled == i)
        rows, cols = cells
        
        touches_border = (rows.min() == 0 or rows.max() == H-1 or
                        cols.min() == 0 or cols.max() == W-1)
        
        if not touches_border:
            surround_colors = Counter()
            for r, c in zip(rows, cols):
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] != bg:
                        surround_colors[int(inp[nr, nc])] += 1
            
            if surround_colors:
                fill_color = surround_colors.most_common(1)[0][0]
                for r, c in zip(rows, cols):
                    out[r, c] = fill_color
    
    return _l(out)


# === Main ===

def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_g(pred), _g(out)):
            return False
    return True


def generate_color_map_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces
    
    # Only same-size tasks
    inp0 = _g(train_pairs[0][0])
    out0 = _g(train_pairs[0][1])
    if inp0.shape != out0.shape:
        return pieces
    
    strategies = [
        ('global_color_map', learn_global_color_map, apply_global_color_map),
        ('color_swap', learn_color_swap, apply_color_swap),
        ('fill_enclosed', learn_fill_enclosed, apply_fill_enclosed),
        ('minority_to_majority', learn_minority_to_majority, apply_minority_to_majority),
        ('recolor_by_size', learn_recolor_by_size, apply_recolor_by_size),
    ]
    
    for name, learn_fn, apply_fn in strategies:
        try:
            rule = learn_fn(train_pairs)
            if rule is None: continue
            fn = lambda inp, r=rule, af=apply_fn: af(inp, r)
            if _verify(fn, train_pairs):
                pieces.append(CrossPiece(name=f"color_map:{name}", apply_fn=fn))
                return pieces
        except Exception:
            continue
    
    return pieces
