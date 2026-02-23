"""
arc/per_object_stamp.py — Per-Object Stamp Pattern Learning

Multi-layer Cross decomposition:
  Layer 1: Detect objects
  Layer 2: For each object, learn a stamp pattern (relative offsets + colors)
  Cross: Object-color × Stamp-template → Per-object add

Handles tasks where each object independently adds cells around itself.
"""

from typing import List, Tuple, Optional, Dict, Set
from collections import Counter, defaultdict
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.objects import detect_objects, ArcObject


def learn_per_object_stamp(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn a per-object stamp rule.
    
    For each training pair:
      1. Detect objects in input
      2. For each object, compute diff (what cells changed near it)
      3. Express diff as relative offsets from object center/cells
      4. Group by object color → stamp template per color
      5. Verify consistency across pairs
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    # Strategy 1: color → stamp template (all objects of same color get same stamp)
    rule = _learn_color_to_stamp(train_pairs, bg)
    if rule is not None:
        return rule
    
    # Strategy 2: each single-cell object gets a stamp based on its color
    rule = _learn_pixel_stamp(train_pairs, bg)
    if rule is not None:
        return rule
    
    # Strategy 3: object shape → stamp (shape determines the pattern)
    rule = _learn_shape_to_stamp(train_pairs, bg)
    if rule is not None:
        return rule
    
    return None


def _compute_adds_per_object(inp, out, bg, objs):
    """For each object, find what cells were added (inp→out) that are
    closest to this object."""
    h, w = grid_shape(inp)
    
    # Find all add cells
    all_adds = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] == bg and out[r][c] != bg:
                all_adds.append((r, c, out[r][c]))
    
    # Assign each add cell to the nearest object
    per_obj = defaultdict(list)
    for r, c, color in all_adds:
        best_dist = float('inf')
        best_idx = -1
        for idx, obj in enumerate(objs):
            for or_, oc in obj.cells:
                d = abs(r - or_) + abs(c - oc)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
        if best_idx >= 0:
            per_obj[best_idx].append((r, c, color))
    
    return per_obj


def _cells_to_relative_stamp(adds, obj):
    """Convert absolute add positions to relative offsets from object center."""
    cr, cc = obj.center
    cr, cc = int(round(cr)), int(round(cc))
    stamp = []
    for r, c, color in adds:
        stamp.append((r - cr, c - cc, color))
    return tuple(sorted(stamp))


def _cells_to_relative_stamp_from_cells(adds, obj, bg):
    """Convert adds to relative offsets, using 'relative to nearest obj cell'."""
    stamp = []
    for r, c, color in adds:
        # Find nearest object cell
        best_dr, best_dc = 0, 0
        best_d = float('inf')
        for or_, oc in obj.cells:
            d = abs(r - or_) + abs(c - oc)
            if d < best_d:
                best_d = d
                best_dr = r - or_
                best_dc = c - oc
        stamp.append((best_dr, best_dc, color))
    return tuple(sorted(stamp))


def _normalize_stamp_colors(stamp, obj_color):
    """Replace absolute colors with symbolic: 'self' for obj_color, else keep."""
    return tuple((dr, dc, 'self' if c == obj_color else c) for dr, dc, c in stamp)


def _learn_pixel_stamp(train_pairs, bg):
    """Each single-pixel object gets a stamp pattern based on its color."""
    
    # Check: are all objects single-pixel?
    for inp, _ in train_pairs:
        objs = detect_objects(inp, bg)
        if not objs:
            return None
        if not all(obj.size == 1 for obj in objs):
            return None
    
    # Learn stamp per color
    color_stamps = {}
    
    for inp, out in train_pairs:
        objs = detect_objects(inp, bg)
        per_obj = _compute_adds_per_object(inp, out, bg, objs)
        
        for idx, obj in enumerate(objs):
            adds = per_obj.get(idx, [])
            stamp = _cells_to_relative_stamp(adds, obj)
            
            if obj.color in color_stamps:
                if color_stamps[obj.color] != stamp:
                    # Try: stamp may vary by position (boundary clipping)
                    # Use "unclipped" stamp from non-boundary objects
                    return None
            else:
                color_stamps[obj.color] = stamp
    
    if not color_stamps:
        return None
    
    return {
        'type': 'pixel_stamp',
        'color_stamps': color_stamps,
        'name': 'pixel_stamp_by_color',
    }


def _learn_color_to_stamp(train_pairs, bg):
    """Objects of the same color get the same relative stamp pattern."""
    
    color_stamps = {}
    
    for inp, out in train_pairs:
        objs = detect_objects(inp, bg)
        if not objs:
            return None
        per_obj = _compute_adds_per_object(inp, out, bg, objs)
        
        for idx, obj in enumerate(objs):
            adds = per_obj.get(idx, [])
            if not adds:
                continue
            stamp = _cells_to_relative_stamp(adds, obj)
            
            # Normalize: use relative color (self vs other)
            norm_stamp = _normalize_stamp_colors(stamp, obj.color)
            
            if obj.color in color_stamps:
                if color_stamps[obj.color] != norm_stamp:
                    return None
            else:
                color_stamps[obj.color] = norm_stamp
    
    if not color_stamps:
        return None
    
    # Full verification
    for inp, out in train_pairs:
        result = _apply_stamp_rule({'type': 'color_stamp', 'color_stamps': color_stamps}, inp, bg)
        if result is None or not grid_eq(result, out):
            return None
    
    return {
        'type': 'color_stamp',
        'color_stamps': color_stamps,
        'name': 'stamp_by_obj_color',
    }


def _learn_shape_to_stamp(train_pairs, bg):
    """Objects with the same shape get the same stamp pattern."""
    
    shape_stamps = {}
    
    for inp, out in train_pairs:
        objs = detect_objects(inp, bg)
        if not objs:
            return None
        per_obj = _compute_adds_per_object(inp, out, bg, objs)
        
        for idx, obj in enumerate(objs):
            adds = per_obj.get(idx, [])
            if not adds:
                continue
            stamp = _cells_to_relative_stamp(adds, obj)
            norm_stamp = _normalize_stamp_colors(stamp, obj.color)
            
            if obj.shape in shape_stamps:
                if shape_stamps[obj.shape] != norm_stamp:
                    return None
            else:
                shape_stamps[obj.shape] = norm_stamp
    
    if not shape_stamps:
        return None
    
    for inp, out in train_pairs:
        result = _apply_stamp_rule({'type': 'shape_stamp', 'shape_stamps': shape_stamps}, inp, bg)
        if result is None or not grid_eq(result, out):
            return None
    
    return {
        'type': 'shape_stamp',
        'shape_stamps': shape_stamps,
        'name': 'stamp_by_obj_shape',
    }


def apply_per_object_stamp(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply a per-object stamp rule to an input grid."""
    bg = most_common_color(inp)
    return _apply_stamp_rule(rule, inp, bg)


def _apply_stamp_rule(rule, inp, bg):
    h, w = grid_shape(inp)
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    
    rtype = rule['type']
    
    if rtype == 'pixel_stamp':
        stamps = rule['color_stamps']
        for obj in objs:
            if obj.color not in stamps:
                continue
            stamp = stamps[obj.color]
            cr, cc = int(round(obj.center[0])), int(round(obj.center[1]))
            for dr, dc, color in stamp:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                    result[nr][nc] = color
    
    elif rtype == 'color_stamp':
        stamps = rule['color_stamps']
        for obj in objs:
            if obj.color not in stamps:
                continue
            stamp = stamps[obj.color]
            cr, cc = int(round(obj.center[0])), int(round(obj.center[1]))
            for dr, dc, color in stamp:
                actual_color = obj.color if color == 'self' else color
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                    result[nr][nc] = actual_color
    
    elif rtype == 'shape_stamp':
        stamps = rule['shape_stamps']
        for obj in objs:
            if obj.shape not in stamps:
                continue
            stamp = stamps[obj.shape]
            cr, cc = int(round(obj.center[0])), int(round(obj.center[1]))
            for dr, dc, color in stamp:
                actual_color = obj.color if color == 'self' else color
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                    result[nr][nc] = actual_color
    
    return result
