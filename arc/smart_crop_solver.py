"""
arc/smart_crop_solver.py — Smart crop/extract solver

Handles tasks where output = some region extracted from input.
Strategies:
1. Exact subgrid match → learn selection rule
2. Crop to unique color's bbox
3. Crop to region with most color diversity
4. Crop to specific repeating unit
5. Extract non-bg connected region with specific property
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


def _find_subgrid_pos(inp, out):
    """Find where out appears in inp as exact subgrid"""
    oh, ow = out.shape
    ih, iw = inp.shape
    positions = []
    for r0 in range(ih - oh + 1):
        for c0 in range(iw - ow + 1):
            if np.array_equal(inp[r0:r0+oh, c0:c0+ow], out):
                positions.append((r0, c0))
    return positions


def _objects(grid, bg):
    mask = (grid != bg).astype(int)
    labeled, n = connected_components(mask)
    objs = []
    for i in range(1, n+1):
        rows, cols = np.where(labeled == i)
        if len(rows) == 0: continue
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        colors = set(int(grid[r,c]) for r,c in zip(rows, cols))
        objs.append({
            'bbox': (int(r1), int(c1), int(r2), int(c2)),
            'area': len(rows),
            'colors': colors,
            'n_colors': len(colors),
            'mask': (labeled == i),
        })
    return objs


# === Strategy: crop to object with most colors ===

def learn_crop_most_colorful(train_pairs):
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
            return None
        bg = _bg(inp)
        objs = _objects(inp, bg)
        if not objs: return None
        
        best = max(objs, key=lambda o: o['n_colors'])
        r1, c1, r2, c2 = best['bbox']
        crop = inp[r1:r2+1, c1:c2+1]
        if not np.array_equal(crop, out):
            return None
    return {'strategy': 'most_colorful'}


# === Strategy: crop to largest object ===

def learn_crop_largest(train_pairs):
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
            return None
        bg = _bg(inp)
        objs = _objects(inp, bg)
        if not objs: return None
        
        best = max(objs, key=lambda o: o['area'])
        r1, c1, r2, c2 = best['bbox']
        crop = inp[r1:r2+1, c1:c2+1]
        if not np.array_equal(crop, out):
            return None
    return {'strategy': 'largest'}


# === Strategy: crop to smallest object ===

def learn_crop_smallest(train_pairs):
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
            return None
        bg = _bg(inp)
        objs = _objects(inp, bg)
        if not objs: return None
        
        best = min(objs, key=lambda o: o['area'])
        r1, c1, r2, c2 = best['bbox']
        crop = inp[r1:r2+1, c1:c2+1]
        if not np.array_equal(crop, out):
            return None
    return {'strategy': 'smallest'}


# === Strategy: crop to object with unique color ===

def learn_crop_unique_color_obj(train_pairs):
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
            return None
        bg = _bg(inp)
        objs = _objects(inp, bg)
        if len(objs) < 2: return None
        
        # Find object with a color no other object has
        all_colors = Counter()
        for o in objs:
            for c in o['colors']:
                all_colors[c] += 1
        
        unique_obj = None
        for o in objs:
            has_unique = any(all_colors[c] == 1 for c in o['colors'])
            if has_unique:
                if unique_obj is not None: return None  # multiple unique
                unique_obj = o
        
        if unique_obj is None: return None
        r1, c1, r2, c2 = unique_obj['bbox']
        crop = inp[r1:r2+1, c1:c2+1]
        if not np.array_equal(crop, out):
            return None
    
    return {'strategy': 'unique_color_obj'}


# === Strategy: crop non-bg bbox ===

def learn_crop_nonbg_bbox(train_pairs):
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
            return None
        bg = _bg(inp)
        rows, cols = np.where(inp != bg)
        if len(rows) == 0: return None
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        crop = inp[r1:r2+1, c1:c2+1]
        if not np.array_equal(crop, out):
            return None
    return {'strategy': 'nonbg_bbox'}


# === Strategy: output size divides input size (tile unit) ===

def learn_crop_tile_unit(train_pairs):
    """If output size divides input, extract the repeating unit"""
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh >= ih or ow >= iw: return None
        if ih % oh != 0 or iw % ow != 0: return None
        
        # Check if input is tiled version of output
        for r in range(0, ih, oh):
            for c in range(0, iw, ow):
                tile = inp[r:r+oh, c:c+ow]
                if not np.array_equal(tile, out):
                    # Allow some tiles to differ (find the common one)
                    pass
    
    # Find most common tile
    for inp_g, out_g in train_pairs:
        inp, out = _g(inp_g), _g(out_g)
        ih, iw = inp.shape
        oh, ow = out.shape
        
        tiles = []
        for r in range(0, ih, oh):
            for c in range(0, iw, ow):
                tiles.append(inp[r:r+oh, c:c+ow].tobytes())
        
        tile_counts = Counter(tiles)
        most_common_tile = tile_counts.most_common(1)[0][0]
        most_common = np.frombuffer(most_common_tile, dtype=inp.dtype).reshape(oh, ow)
        
        if not np.array_equal(most_common, out):
            return None
    
    return {'strategy': 'tile_unit'}


# === Strategy: crop to object by relative position ===

def learn_crop_by_position(train_pairs):
    """Crop to object at specific relative position (top-left, center, etc.)"""
    for pos_name, sort_fn in [
        ('top_left', lambda o: (o['bbox'][0], o['bbox'][1])),
        ('top_right', lambda o: (o['bbox'][0], -o['bbox'][3])),
        ('bottom_left', lambda o: (-o['bbox'][2], o['bbox'][1])),
        ('bottom_right', lambda o: (-o['bbox'][2], -o['bbox'][3])),
        ('center', lambda o: abs(o['bbox'][0] + o['bbox'][2]) + abs(o['bbox'][1] + o['bbox'][3])),
    ]:
        all_match = True
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            if out.shape[0] >= inp.shape[0] and out.shape[1] >= inp.shape[1]:
                all_match = False; break
            bg = _bg(inp)
            objs = _objects(inp, bg)
            if not objs:
                all_match = False; break
            
            selected = sorted(objs, key=sort_fn)[0]
            r1, c1, r2, c2 = selected['bbox']
            crop = inp[r1:r2+1, c1:c2+1]
            if not np.array_equal(crop, out):
                all_match = False; break
        
        if all_match:
            return {'strategy': 'by_position', 'pos': pos_name}
    
    return None


def _apply(inp_g, rule):
    inp = _g(inp_g)
    bg = _bg(inp)
    strat = rule['strategy']
    
    if strat == 'nonbg_bbox':
        rows, cols = np.where(inp != bg)
        if len(rows) == 0: return None
        return _l(inp[rows.min():rows.max()+1, cols.min():cols.max()+1])
    
    objs = _objects(inp, bg)
    if not objs: return None
    
    if strat == 'most_colorful':
        obj = max(objs, key=lambda o: o['n_colors'])
    elif strat == 'largest':
        obj = max(objs, key=lambda o: o['area'])
    elif strat == 'smallest':
        obj = min(objs, key=lambda o: o['area'])
    elif strat == 'unique_color_obj':
        all_colors = Counter()
        for o in objs:
            for c in o['colors']: all_colors[c] += 1
        obj = None
        for o in objs:
            if any(all_colors[c] == 1 for c in o['colors']):
                obj = o; break
        if obj is None: return None
    elif strat == 'by_position':
        sort_fns = {
            'top_left': lambda o: (o['bbox'][0], o['bbox'][1]),
            'top_right': lambda o: (o['bbox'][0], -o['bbox'][3]),
            'bottom_left': lambda o: (-o['bbox'][2], o['bbox'][1]),
            'bottom_right': lambda o: (-o['bbox'][2], -o['bbox'][3]),
            'center': lambda o: abs(o['bbox'][0] + o['bbox'][2]) + abs(o['bbox'][1] + o['bbox'][3]),
        }
        obj = sorted(objs, key=sort_fns[rule['pos']])[0]
    elif strat == 'tile_unit':
        ih, iw = inp.shape
        # Need output shape — infer from train
        # This is tricky, just try common divisors
        for oh in range(1, ih+1):
            for ow in range(1, iw+1):
                if ih % oh == 0 and iw % ow == 0:
                    tiles = []
                    for r in range(0, ih, oh):
                        for c in range(0, iw, ow):
                            tiles.append(inp[r:r+oh, c:c+ow].tobytes())
                    tc = Counter(tiles)
                    most = tc.most_common(1)[0][0]
                    return _l(np.frombuffer(most, dtype=inp.dtype).reshape(oh, ow))
        return None
    else:
        return None
    
    r1, c1, r2, c2 = obj['bbox']
    return _l(inp[r1:r2+1, c1:c2+1])


def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_g(pred), _g(out)):
            return False
    return True


def generate_smart_crop_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces
    
    inp0 = _g(train_pairs[0][0])
    out0 = _g(train_pairs[0][1])
    if out0.shape[0] >= inp0.shape[0] and out0.shape[1] >= inp0.shape[1]:
        return pieces
    
    learners = [
        learn_crop_nonbg_bbox,
        learn_crop_largest,
        learn_crop_smallest,
        learn_crop_most_colorful,
        learn_crop_unique_color_obj,
        learn_crop_by_position,
        learn_crop_tile_unit,
    ]
    
    for learn_fn in learners:
        try:
            rule = learn_fn(train_pairs)
            if rule is None: continue
            fn = lambda inp, r=rule: _apply(inp, r)
            if _verify(fn, train_pairs):
                pieces.append(CrossPiece(name=f"smart_crop:{rule['strategy']}", apply_fn=fn))
                return pieces
        except Exception:
            continue
    
    return pieces
