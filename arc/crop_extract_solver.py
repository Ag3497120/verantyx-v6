"""
arc/crop_extract_solver.py — Solver for crop/extract tasks

Targets: output is smaller than input (extracted region).
Strategies:
1. Extract bounding box of specific color
2. Extract the unique/different object
3. Extract the smallest/largest connected component
4. Extract the non-background region
5. Extract a sub-grid pattern (repeated pattern → extract unit)
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
from scipy.ndimage import label as connected_components
from arc.cross_engine import CrossPiece

Grid = List[List[int]]


def _grid_np(g):
    return np.array(g, dtype=int)

def _np_grid(a):
    return a.tolist()


def _get_bg(grid):
    """Most common color = background"""
    return int(Counter(grid.flatten()).most_common(1)[0][0])


def _find_objects(grid, bg=None):
    """Find connected components (non-bg)"""
    if bg is None:
        bg = _get_bg(grid)
    mask = (grid != bg).astype(int)
    labeled, n = connected_components(mask)
    objects = []
    for i in range(1, n + 1):
        rows, cols = np.where(labeled == i)
        if len(rows) == 0:
            continue
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        crop = grid[r1:r2+1, c1:c2+1].copy()
        objects.append({
            'bbox': (r1, c1, r2, c2),
            'crop': crop,
            'area': len(rows),
            'colors': set(int(grid[r, c]) for r, c in zip(rows, cols)),
            'mask': (labeled == i),
        })
    return objects


def _extract_bbox_color(grid, color, bg=None):
    """Extract bounding box of all cells with given color"""
    if bg is None:
        bg = _get_bg(grid)
    rows, cols = np.where(grid == color)
    if len(rows) == 0:
        return None
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    return grid[r1:r2+1, c1:c2+1].copy()


# === Strategy 1: Extract bbox of non-bg region ===

def learn_extract_nonbg(train_pairs):
    """Output = bounding box of all non-bg cells"""
    bg = None
    for inp_g, out_g in train_pairs:
        inp = _grid_np(inp_g)
        out = _grid_np(out_g)
        if inp.shape == out.shape:
            return None
        
        if bg is None:
            bg = _get_bg(inp)
        
        rows, cols = np.where(inp != bg)
        if len(rows) == 0:
            return None
        r1, r2 = rows.min(), rows.max()
        c1, c2 = cols.min(), cols.max()
        
        crop = inp[r1:r2+1, c1:c2+1]
        if not np.array_equal(crop, out):
            return None
    
    return {'bg': bg}


# === Strategy 2: Extract specific color's bbox ===

def learn_extract_color_bbox(train_pairs):
    """Output = bbox of cells with a specific color"""
    for target_color in range(1, 10):
        all_match = True
        for inp_g, out_g in train_pairs:
            inp = _grid_np(inp_g)
            out = _grid_np(out_g)
            
            crop = _extract_bbox_color(inp, target_color)
            if crop is None or not np.array_equal(crop, out):
                all_match = False
                break
        
        if all_match:
            return {'color': target_color}
    
    return None


# === Strategy 3: Extract smallest/largest object ===

def learn_extract_by_size(train_pairs):
    """Output = the smallest or largest connected component"""
    for mode in ['smallest', 'largest', 'second_largest']:
        all_match = True
        for inp_g, out_g in train_pairs:
            inp = _grid_np(inp_g)
            out = _grid_np(out_g)
            bg = _get_bg(inp)
            
            objs = _find_objects(inp, bg)
            if not objs:
                all_match = False
                break
            
            if mode == 'smallest':
                obj = min(objs, key=lambda o: o['area'])
            elif mode == 'largest':
                obj = max(objs, key=lambda o: o['area'])
            elif mode == 'second_largest':
                if len(objs) < 2:
                    all_match = False
                    break
                sorted_objs = sorted(objs, key=lambda o: -o['area'])
                obj = sorted_objs[1]
            
            r1, c1, r2, c2 = obj['bbox']
            crop = inp[r1:r2+1, c1:c2+1].copy()
            
            # Try with bg replacement
            if not np.array_equal(crop, out):
                # Try masking non-object cells to bg
                crop2 = np.full_like(crop, bg)
                mask_crop = obj['mask'][r1:r2+1, c1:c2+1]
                crop2[mask_crop] = crop[mask_crop]
                if not np.array_equal(crop2, out):
                    all_match = False
                    break
        
        if all_match:
            return {'mode': mode}
    
    return None


# === Strategy 4: Extract the unique object (the one that differs from others) ===

def learn_extract_unique(train_pairs):
    """Find the object that is unique (different shape/color from others)"""
    for inp_g, out_g in train_pairs:
        inp = _grid_np(inp_g)
        out = _grid_np(out_g)
        bg = _get_bg(inp)
        
        objs = _find_objects(inp, bg)
        if len(objs) < 2:
            return None
        
        # Check if output matches any single object's crop
        found = False
        for obj in objs:
            r1, c1, r2, c2 = obj['bbox']
            crop = inp[r1:r2+1, c1:c2+1].copy()
            if np.array_equal(crop, out):
                found = True
                break
            # Try masked
            crop2 = np.full_like(crop, bg)
            mask_crop = obj['mask'][r1:r2+1, c1:c2+1]
            crop2[mask_crop] = crop[mask_crop]
            if np.array_equal(crop2, out):
                found = True
                break
        
        if not found:
            return None
    
    # Figure out the selection criteria
    # Try: unique by shape (only one with that shape)
    # Try: unique by color
    # Try: unique by area
    
    criteria = None
    for criterion in ['unique_shape', 'unique_color', 'unique_area', 'min_area', 'max_area']:
        all_match = True
        for inp_g, out_g in train_pairs:
            inp = _grid_np(inp_g)
            out = _grid_np(out_g)
            bg = _get_bg(inp)
            objs = _find_objects(inp, bg)
            
            selected = _select_object(objs, inp, bg, criterion)
            if selected is None:
                all_match = False
                break
            
            r1, c1, r2, c2 = selected['bbox']
            crop = inp[r1:r2+1, c1:c2+1].copy()
            crop2 = np.full_like(crop, bg)
            mask_crop = selected['mask'][r1:r2+1, c1:c2+1]
            crop2[mask_crop] = crop[mask_crop]
            
            if not (np.array_equal(crop, out) or np.array_equal(crop2, out)):
                all_match = False
                break
        
        if all_match:
            criteria = criterion
            break
    
    if criteria:
        return {'criterion': criteria}
    return None


def _select_object(objs, grid, bg, criterion):
    """Select object by criterion"""
    if criterion == 'unique_shape':
        shapes = [(o['crop'].shape, i) for i, o in enumerate(objs)]
        shape_counts = Counter(s for s, _ in shapes)
        uniques = [(s, i) for s, i in shapes if shape_counts[s] == 1]
        if len(uniques) == 1:
            return objs[uniques[0][1]]
    
    elif criterion == 'unique_color':
        color_sets = [(frozenset(o['colors']), i) for i, o in enumerate(objs)]
        color_counts = Counter(cs for cs, _ in color_sets)
        uniques = [(cs, i) for cs, i in color_sets if color_counts[cs] == 1]
        if len(uniques) == 1:
            return objs[uniques[0][1]]
    
    elif criterion == 'unique_area':
        areas = [(o['area'], i) for i, o in enumerate(objs)]
        area_counts = Counter(a for a, _ in areas)
        uniques = [(a, i) for a, i in areas if area_counts[a] == 1]
        if len(uniques) == 1:
            return objs[uniques[0][1]]
    
    elif criterion == 'min_area':
        return min(objs, key=lambda o: o['area'])
    
    elif criterion == 'max_area':
        return max(objs, key=lambda o: o['area'])
    
    return None


def apply_extract_unique(inp_g, rule):
    inp = _grid_np(inp_g)
    bg = _get_bg(inp)
    objs = _find_objects(inp, bg)
    
    selected = _select_object(objs, inp, bg, rule['criterion'])
    if selected is None:
        return None
    
    r1, c1, r2, c2 = selected['bbox']
    crop = inp[r1:r2+1, c1:c2+1].copy()
    
    # Try both masked and unmasked
    crop2 = np.full_like(crop, bg)
    mask_crop = selected['mask'][r1:r2+1, c1:c2+1]
    crop2[mask_crop] = crop[mask_crop]
    
    return _np_grid(crop2)


# === Main piece generator ===

def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None:
            return False
        if not np.array_equal(_grid_np(pred), _grid_np(out)):
            return False
    return True


def generate_crop_extract_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces
    
    # Quick check: is output smaller?
    inp0 = _grid_np(train_pairs[0][0])
    out0 = _grid_np(train_pairs[0][1])
    if out0.shape[0] >= inp0.shape[0] and out0.shape[1] >= inp0.shape[1]:
        return pieces  # not a crop task
    
    strategies = [
        ('extract_nonbg', learn_extract_nonbg,
         lambda inp, r: _np_grid(_grid_np(inp)[np.where(_grid_np(inp) != r['bg'])[0].min():np.where(_grid_np(inp) != r['bg'])[0].max()+1, 
                                                np.where(_grid_np(inp) != r['bg'])[1].min():np.where(_grid_np(inp) != r['bg'])[1].max()+1])),
        ('extract_color_bbox', learn_extract_color_bbox,
         lambda inp, r: _np_grid(_extract_bbox_color(_grid_np(inp), r['color']))),
        ('extract_unique', learn_extract_unique, apply_extract_unique),
    ]
    
    # Also try extract by size
    size_rule = learn_extract_by_size(train_pairs)
    if size_rule:
        def _apply_size(inp, rule=size_rule):
            inp_np = _grid_np(inp)
            bg = _get_bg(inp_np)
            objs = _find_objects(inp_np, bg)
            if not objs:
                return None
            if rule['mode'] == 'smallest':
                obj = min(objs, key=lambda o: o['area'])
            elif rule['mode'] == 'largest':
                obj = max(objs, key=lambda o: o['area'])
            elif rule['mode'] == 'second_largest':
                if len(objs) < 2: return None
                obj = sorted(objs, key=lambda o: -o['area'])[1]
            r1, c1, r2, c2 = obj['bbox']
            crop = inp_np[r1:r2+1, c1:c2+1].copy()
            crop2 = np.full_like(crop, bg)
            mask_crop = obj['mask'][r1:r2+1, c1:c2+1]
            crop2[mask_crop] = crop[mask_crop]
            return _np_grid(crop2)
        
        if _verify(_apply_size, train_pairs):
            pieces.append(CrossPiece(name=f"crop:{size_rule['mode']}", apply_fn=_apply_size))
            return pieces
    
    for name, learn_fn, apply_fn in strategies:
        try:
            rule = learn_fn(train_pairs)
            if rule is None:
                continue
            fn = lambda inp, r=rule, af=apply_fn: af(inp, r)
            if _verify(fn, train_pairs):
                pieces.append(CrossPiece(name=f"crop:{name}", apply_fn=fn))
                return pieces
        except Exception:
            continue
    
    return pieces
