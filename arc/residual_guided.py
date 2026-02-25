"""
Residual-Guided Piece Generator (Module 20)

Analyzes the residual (diff between input and target output) to generate
TARGETED pieces. This is the "reverse direction" of cross-structure search.

Instead of trying all pieces and checking if they work, we:
1. Analyze WHAT needs to change (residual structure)
2. Generate pieces specifically designed for that change type
3. Feed them into the existing verify/compose pipeline
"""

from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.cross_engine import CrossPiece
from arc.grid import Grid, grid_shape, grid_eq, most_common_color

try:
    import numpy as np
    from scipy import ndimage
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _to_np(grid):
    return np.array(grid)


def _from_np(arr):
    return [[int(v) for v in row] for row in arr]


# ============================================================
# Residual Analysis
# ============================================================

def analyze_residual(inp, out, bg):
    """Analyze the structural difference between input and output."""
    inp_arr = _to_np(inp)
    out_arr = _to_np(out)
    h, w = inp_arr.shape
    
    diff_mask = inp_arr != out_arr
    n_diff = int(diff_mask.sum())
    if n_diff == 0:
        return {'type': 'identity'}
    
    added_mask = diff_mask & (inp_arr == bg)
    removed_mask = diff_mask & (out_arr == bg)
    recolored_mask = diff_mask & (inp_arr != bg) & (out_arr != bg)
    
    n_added = int(added_mask.sum())
    n_removed = int(removed_mask.sum())
    n_recolored = int(recolored_mask.sum())
    
    return {
        'type': _classify_residual(n_added, n_removed, n_recolored, 
                                    inp_arr, out_arr, bg, diff_mask,
                                    added_mask, removed_mask),
        'n_added': n_added,
        'n_removed': n_removed,
        'n_recolored': n_recolored,
        'n_diff': n_diff,
        'diff_pct': n_diff / (h * w) * 100,
    }


def _classify_residual(n_added, n_removed, n_recolored,
                        inp_arr, out_arr, bg, diff_mask,
                        added_mask, removed_mask):
    h, w = inp_arr.shape
    
    # Object movement: added ≈ removed, same shapes
    if n_added > 0 and n_removed > 0 and abs(n_added - n_removed) <= 2:
        add_labeled, na = ndimage.label(added_mask)
        rem_labeled, nr = ndimage.label(removed_mask)
        if na == nr and na <= 5:
            add_shapes = set()
            rem_shapes = set()
            for aid in range(1, na + 1):
                cells = set(tuple(x) for x in np.argwhere(add_labeled == aid))
                mn_r = min(r for r, _ in cells)
                mn_c = min(c for _, c in cells)
                add_shapes.add(frozenset((r - mn_r, c - mn_c) for r, c in cells))
            for rid in range(1, nr + 1):
                cells = set(tuple(x) for x in np.argwhere(rem_labeled == rid))
                mn_r = min(r for r, _ in cells)
                mn_c = min(c for _, c in cells)
                rem_shapes.add(frozenset((r - mn_r, c - mn_c) for r, c in cells))
            if add_shapes == rem_shapes:
                return 'object_movement'
    
    if n_added > 0 and n_removed == 0 and n_recolored == 0:
        obj_mask = inp_arr != bg
        adj = 0
        total = 0
        for r, c in zip(*np.where(added_mask)):
            total += 1
            if any(0 <= r + dr < h and 0 <= c + dc < w and obj_mask[r + dr, c + dc]
                   for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                adj += 1
        if total > 0 and adj >= total * 0.6:
            return 'object_expansion'
        return 'add_pattern'
    
    if n_recolored > 0 and n_added == 0 and n_removed == 0:
        return 'pure_recolor'
    
    if n_removed > 0 and n_added == 0 and n_recolored == 0:
        return 'pure_removal'
    
    return 'mixed'


# ============================================================
# Targeted Piece Generators
# ============================================================

def generate_residual_guided_pieces(
    train_pairs: List[Tuple[Grid, Grid]]
) -> List[CrossPiece]:
    """Generate pieces based on residual analysis of train pairs."""
    if not HAS_NUMPY:
        return []
    
    pieces = []
    
    try:
        inp0, out0 = train_pairs[0]
        if grid_shape(inp0) != grid_shape(out0):
            # Size change tasks: handled by other modules
            return pieces
        
        bg = most_common_color(inp0)
        residual = analyze_residual(inp0, out0, bg)
        rtype = residual['type']
        
        # Route to appropriate generator
        if rtype == 'pure_recolor':
            pieces.extend(_gen_recolor_pieces(train_pairs, bg))
        elif rtype == 'object_expansion':
            pieces.extend(_gen_expansion_pieces(train_pairs, bg))
        elif rtype == 'object_movement':
            pieces.extend(_gen_movement_pieces(train_pairs, bg))
        elif rtype == 'add_pattern':
            pieces.extend(_gen_add_pattern_pieces(train_pairs, bg))
        elif rtype == 'pure_removal':
            pieces.extend(_gen_removal_pieces(train_pairs, bg))
        elif rtype == 'mixed':
            # Try all generators — mixed tasks may match specialized rules
            pieces.extend(_gen_recolor_pieces(train_pairs, bg))
            pieces.extend(_gen_expansion_pieces(train_pairs, bg))
            pieces.extend(_gen_movement_pieces(train_pairs, bg))
            pieces.extend(_gen_removal_pieces(train_pairs, bg))
            pieces.extend(_gen_conditional_transform_pieces(train_pairs, bg))
    except Exception:
        pass
    
    return pieces


# ============================================================
# 1. Conditional Per-Object Recolor (pure_recolor: 92 tasks)
# ============================================================

def _gen_recolor_pieces(train_pairs, bg):
    """Generate recolor pieces based on object properties."""
    pieces = []
    
    # Strategy: learn color mapping based on various object properties
    # Properties: position, neighbor_color, enclosed_by, touching_border,
    #             relative_size, bbox_aspect_ratio, distance_to_center
    
    try:
        from arc.objects import detect_objects
    except ImportError:
        return pieces
    
    # Collect object -> output_color mappings
    all_mappings = []
    for inp, out in train_pairs:
        objs = detect_objects(inp, bg)
        if not objs:
            return pieces
        mappings = []
        for obj in objs:
            out_colors = Counter(out[r][c] for r, c in obj.cells if out[r][c] != bg)
            if out_colors:
                new_c = out_colors.most_common(1)[0][0]
                mappings.append((obj, new_c))
        all_mappings.append(mappings)
    
    if not all_mappings:
        return pieces
    
    # --- Strategy: recolor by neighbor object color ---
    rule = _learn_recolor_by_neighbor_obj_color(all_mappings, train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            'rg:recolor_by_neighbor_obj',
            lambda inp, r=_r: _apply_recolor_by_neighbor_obj(inp, r)
        ))
    
    # --- Strategy: recolor by position (top/bottom/left/right) ---
    rule = _learn_recolor_by_position(all_mappings, train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'rg:recolor_by_position:{_r["axis"]}',
            lambda inp, r=_r: _apply_recolor_by_position(inp, r)
        ))
    
    # --- Strategy: recolor by enclosed status ---
    rule = _learn_recolor_by_enclosed(all_mappings, train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            'rg:recolor_by_enclosed',
            lambda inp, r=_r: _apply_recolor_by_enclosed(inp, r)
        ))
    
    # --- Strategy: recolor by touching border ---
    rule = _learn_recolor_by_border_touch(all_mappings, train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            'rg:recolor_by_border',
            lambda inp, r=_r: _apply_recolor_by_border_touch(inp, r)
        ))
    
    # --- Strategy: recolor by object count per color group ---
    rule = _learn_recolor_by_group_count(all_mappings, train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            'rg:recolor_by_group_count',
            lambda inp, r=_r: _apply_recolor_by_group_count(inp, r)
        ))
    
    return pieces


def _learn_recolor_by_neighbor_obj_color(all_mappings, train_pairs, bg):
    """Recolor each object to the color of its nearest neighbor object."""
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    
    for mode in ['nearest', 'largest_neighbor', 'most_common_neighbor']:
        consistent = True
        for mappings in all_mappings:
            objs = [obj for obj, _ in mappings]
            for obj, new_c in mappings:
                # Find nearest other object
                min_dist = float('inf')
                nearest_color = None
                largest_size = 0
                color_counter = Counter()
                
                for other, _ in mappings:
                    if other is obj:
                        continue
                    # Distance between bounding boxes
                    r1, c1, r2, c2 = obj.bbox
                    or1, oc1, or2, oc2 = other.bbox
                    dr = max(0, max(r1, or1) - min(r2, or2))
                    dc = max(0, max(c1, oc1) - min(c2, oc2))
                    dist = dr + dc
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_color = other.color
                    if other.size > largest_size:
                        largest_size = other.size
                    color_counter[other.color] += 1
                
                if mode == 'nearest' and nearest_color != new_c:
                    consistent = False; break
                elif mode == 'largest_neighbor':
                    pass  # more complex
                elif mode == 'most_common_neighbor':
                    if color_counter and color_counter.most_common(1)[0][0] != new_c:
                        consistent = False; break
            if not consistent:
                break
        
        if consistent and mode == 'nearest':
            return {'mode': 'nearest', 'bg': bg}
    
    return None


def _apply_recolor_by_neighbor_obj(inp, params):
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    bg = params['bg']
    mode = params['mode']
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    
    h, w = grid_shape(inp)
    result = [row[:] for row in inp]
    
    for obj in objs:
        # Find nearest object
        min_dist = float('inf')
        nearest_color = obj.color
        r1, c1, r2, c2 = obj.bbox
        
        for other in objs:
            if other is obj:
                continue
            or1, oc1, or2, oc2 = other.bbox
            dr = max(0, max(r1, or1) - min(r2, or2))
            dc = max(0, max(c1, oc1) - min(c2, oc2))
            dist = dr + dc
            if dist < min_dist:
                min_dist = dist
                nearest_color = other.color
        
        for r, c in obj.cells:
            result[r][c] = nearest_color
    
    return result


def _learn_recolor_by_position(all_mappings, train_pairs, bg):
    """Recolor by position rank (top-to-bottom or left-to-right)."""
    for axis in ['row', 'col']:
        consistent = True
        pos_color_map = {}
        
        for mappings in all_mappings:
            if axis == 'row':
                sorted_m = sorted(mappings, key=lambda x: min(r for r, c in x[0].cells))
            else:
                sorted_m = sorted(mappings, key=lambda x: min(c for r, c in x[0].cells))
            
            for rank, (obj, new_c) in enumerate(sorted_m):
                if rank in pos_color_map:
                    if pos_color_map[rank] != new_c:
                        consistent = False; break
                else:
                    pos_color_map[rank] = new_c
            if not consistent:
                break
        
        if consistent and pos_color_map and len(set(pos_color_map.values())) >= 2:
            return {'axis': axis, 'pos_map': pos_color_map, 'bg': bg}
    
    return None


def _apply_recolor_by_position(inp, params):
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    bg = params['bg']
    axis = params['axis']
    pos_map = params['pos_map']
    
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    
    if axis == 'row':
        sorted_objs = sorted(objs, key=lambda o: min(r for r, c in o.cells))
    else:
        sorted_objs = sorted(objs, key=lambda o: min(c for r, c in o.cells))
    
    result = [row[:] for row in inp]
    for rank, obj in enumerate(sorted_objs):
        if rank in pos_map:
            for r, c in obj.cells:
                result[r][c] = pos_map[rank]
    
    return result


def _learn_recolor_by_enclosed(all_mappings, train_pairs, bg):
    """Recolor based on whether object is enclosed by another."""
    # Simplified: objects fully inside another object's bbox get one color,
    # objects not enclosed get another
    consistent = True
    enclosed_color = None
    free_color = None
    
    for mappings in all_mappings:
        for obj, new_c in mappings:
            r1, c1, r2, c2 = obj.bbox
            is_enclosed = False
            for other, _ in mappings:
                if other is obj:
                    continue
                or1, oc1, or2, oc2 = other.bbox
                if or1 <= r1 and oc1 <= c1 and or2 >= r2 and oc2 >= c2:
                    is_enclosed = True
                    break
            
            if is_enclosed:
                if enclosed_color is None:
                    enclosed_color = new_c
                elif enclosed_color != new_c:
                    consistent = False; break
            else:
                if free_color is None:
                    free_color = new_c
                elif free_color != new_c:
                    consistent = False; break
        if not consistent:
            break
    
    if consistent and enclosed_color is not None and free_color is not None and enclosed_color != free_color:
        return {'enclosed_color': enclosed_color, 'free_color': free_color, 'bg': bg}
    return None


def _apply_recolor_by_enclosed(inp, params):
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    bg = params['bg']
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    
    result = [row[:] for row in inp]
    for obj in objs:
        r1, c1, r2, c2 = obj.bbox
        is_enclosed = any(
            o is not obj and o.bbox[0] <= r1 and o.bbox[1] <= c1 
            and o.bbox[2] >= r2 and o.bbox[3] >= c2
            for o in objs
        )
        color = params['enclosed_color'] if is_enclosed else params['free_color']
        for r, c in obj.cells:
            result[r][c] = color
    
    return result


def _learn_recolor_by_border_touch(all_mappings, train_pairs, bg):
    """Recolor based on whether object touches grid border."""
    consistent = True
    touch_color = None
    nouch_color = None
    
    for i, mappings in enumerate(all_mappings):
        inp = train_pairs[i][0]
        h, w = grid_shape(inp)
        
        for obj, new_c in mappings:
            touches = any(r == 0 or r == h - 1 or c == 0 or c == w - 1 
                         for r, c in obj.cells)
            if touches:
                if touch_color is None:
                    touch_color = new_c
                elif touch_color != new_c:
                    consistent = False; break
            else:
                if nouch_color is None:
                    nouch_color = new_c
                elif nouch_color != new_c:
                    consistent = False; break
        if not consistent:
            break
    
    if consistent and touch_color is not None and nouch_color is not None and touch_color != nouch_color:
        return {'touch_color': touch_color, 'nouch_color': nouch_color, 'bg': bg}
    return None


def _apply_recolor_by_border_touch(inp, params):
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    bg = params['bg']
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    
    h, w = grid_shape(inp)
    result = [row[:] for row in inp]
    for obj in objs:
        touches = any(r == 0 or r == h - 1 or c == 0 or c == w - 1 
                     for r, c in obj.cells)
        color = params['touch_color'] if touches else params['nouch_color']
        for r, c in obj.cells:
            result[r][c] = color
    
    return result


def _learn_recolor_by_group_count(all_mappings, train_pairs, bg):
    """Recolor: objects of the majority color → one output, minority → another."""
    consistent = True
    majority_out = None
    minority_out = None
    
    for mappings in all_mappings:
        color_groups = Counter(obj.color for obj, _ in mappings)
        if len(color_groups) < 2:
            return None
        
        majority_color = color_groups.most_common(1)[0][0]
        
        for obj, new_c in mappings:
            if obj.color == majority_color:
                if majority_out is None:
                    majority_out = new_c
                elif majority_out != new_c:
                    consistent = False; break
            else:
                if minority_out is None:
                    minority_out = new_c
                elif minority_out != new_c:
                    consistent = False; break
        if not consistent:
            break
    
    if consistent and majority_out is not None and minority_out is not None:
        return {'majority_out': majority_out, 'minority_out': minority_out, 'bg': bg}
    return None


def _apply_recolor_by_group_count(inp, params):
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    bg = params['bg']
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    
    color_groups = Counter(obj.color for obj in objs)
    if not color_groups:
        return None
    majority_color = color_groups.most_common(1)[0][0]
    
    result = [row[:] for row in inp]
    for obj in objs:
        color = params['majority_out'] if obj.color == majority_color else params['minority_out']
        for r, c in obj.cells:
            result[r][c] = color
    
    return result


# ============================================================
# 2. Object Expansion (48 tasks)
# ============================================================

def _gen_expansion_pieces(train_pairs, bg):
    """Generate pieces for object expansion patterns."""
    pieces = []
    
    # Strategy: expand each object by N cells in all directions
    rule = _learn_expand_by_n(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'rg:expand_by_{_r["n"]}',
            lambda inp, r=_r: _apply_expand_by_n(inp, r)
        ))
    
    # Strategy: fill object bounding box
    rule = _learn_fill_bbox(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            'rg:fill_bbox',
            lambda inp, r=_r: _apply_fill_bbox(inp, r)
        ))
    
    # Strategy: draw border/outline around objects
    rule = _learn_draw_border(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'rg:draw_border_c{_r["border_color"]}',
            lambda inp, r=_r: _apply_draw_border(inp, r)
        ))
    
    return pieces


def _learn_expand_by_n(train_pairs, bg):
    """Learn: expand all non-bg cells by N in all cardinal directions."""
    for n in [1, 2, 3]:
        ok = True
        for inp, out in train_pairs:
            inp_arr = _to_np(inp)
            out_arr = _to_np(out)
            if inp_arr.shape != out_arr.shape:
                ok = False; break
            h, w = inp_arr.shape
            
            result = inp_arr.copy()
            for r in range(h):
                for c in range(w):
                    if inp_arr[r, c] != bg:
                        color = int(inp_arr[r, c])
                        for dr in range(-n, n + 1):
                            for dc in range(-n, n + 1):
                                if abs(dr) + abs(dc) <= n:
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == bg:
                                        result[nr, nc] = color
            
            if not np.array_equal(result, out_arr):
                ok = False; break
        
        if ok:
            return {'n': n, 'bg': bg}
    return None


def _apply_expand_by_n(inp, params):
    n = params['n']
    bg = params['bg']
    inp_arr = _to_np(inp)
    h, w = inp_arr.shape
    result = inp_arr.copy()
    
    for r in range(h):
        for c in range(w):
            if inp_arr[r, c] != bg:
                color = int(inp_arr[r, c])
                for dr in range(-n, n + 1):
                    for dc in range(-n, n + 1):
                        if abs(dr) + abs(dc) <= n:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and result[nr, nc] == bg:
                                result[nr, nc] = color
    
    return _from_np(result)


def _learn_fill_bbox(train_pairs, bg):
    """Learn: fill each object's bounding box with its color."""
    for inp, out in train_pairs:
        inp_arr = _to_np(inp)
        out_arr = _to_np(out)
        if inp_arr.shape != out_arr.shape:
            return None
        
        h, w = inp_arr.shape
        mask = inp_arr != bg
        labeled, n_objs = ndimage.label(mask)
        
        result = inp_arr.copy()
        for oid in range(1, n_objs + 1):
            cells = list(zip(*np.where(labeled == oid)))
            color = int(inp_arr[cells[0][0], cells[0][1]])
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if result[r, c] == bg:
                        result[r, c] = color
        
        if not np.array_equal(result, out_arr):
            return None
    
    return {'bg': bg}


def _apply_fill_bbox(inp, params):
    bg = params['bg']
    inp_arr = _to_np(inp)
    h, w = inp_arr.shape
    mask = inp_arr != bg
    labeled, n_objs = ndimage.label(mask)
    
    result = inp_arr.copy()
    for oid in range(1, n_objs + 1):
        cells = list(zip(*np.where(labeled == oid)))
        color = int(inp_arr[cells[0][0], cells[0][1]])
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if result[r, c] == bg:
                    result[r, c] = color
    
    return _from_np(result)


def _learn_draw_border(train_pairs, bg):
    """Learn: draw 1-cell border around each object with a specific color."""
    for border_color in range(10):
        if border_color == bg:
            continue
        ok = True
        for inp, out in train_pairs:
            inp_arr = _to_np(inp)
            out_arr = _to_np(out)
            if inp_arr.shape != out_arr.shape:
                ok = False; break
            
            h, w = inp_arr.shape
            mask = inp_arr != bg
            result = inp_arr.copy()
            
            for r in range(h):
                for c in range(w):
                    if inp_arr[r, c] == bg:
                        if any(0 <= r + dr < h and 0 <= c + dc < w 
                               and inp_arr[r + dr, c + dc] != bg
                               and inp_arr[r + dr, c + dc] != border_color
                               for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                            result[r, c] = border_color
            
            if not np.array_equal(result, out_arr):
                ok = False; break
        
        if ok:
            return {'border_color': border_color, 'bg': bg}
    return None


def _apply_draw_border(inp, params):
    bg = params['bg']
    border_color = params['border_color']
    inp_arr = _to_np(inp)
    h, w = inp_arr.shape
    result = inp_arr.copy()
    
    for r in range(h):
        for c in range(w):
            if inp_arr[r, c] == bg:
                if any(0 <= r + dr < h and 0 <= c + dc < w
                       and inp_arr[r + dr, c + dc] != bg
                       and inp_arr[r + dr, c + dc] != border_color
                       for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    result[r, c] = border_color
    
    return _from_np(result)


# ============================================================
# 3. Object Movement (24 tasks)
# ============================================================

def _gen_movement_pieces(train_pairs, bg):
    """Generate pieces for object movement patterns."""
    pieces = []
    
    # Strategy: learn fixed displacement vector for all objects
    rule = _learn_fixed_displacement(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'rg:move_({_r["dr"]},{_r["dc"]})',
            lambda inp, r=_r: _apply_fixed_displacement(inp, r)
        ))
    
    # Strategy: gravity toward a specific object or toward center
    rule = _learn_gravity_toward(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'rg:gravity_toward_{_r["target"]}',
            lambda inp, r=_r: _apply_gravity_toward(inp, r)
        ))
    
    return pieces


def _learn_fixed_displacement(train_pairs, bg):
    """All objects move by the same (dr, dc)."""
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    
    dr_dc = None
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        if len(in_objs) != len(out_objs):
            return None
        
        # Match objects by shape
        for io in in_objs:
            matched = False
            for oo in out_objs:
                if io.shape == oo.shape and io.color == oo.color:
                    # Compute displacement
                    ir1, ic1, _, _ = io.bbox
                    or1, oc1, _, _ = oo.bbox
                    d = (or1 - ir1, oc1 - ic1)
                    if dr_dc is None:
                        dr_dc = d
                    elif dr_dc != d:
                        return None
                    matched = True
                    break
            if not matched:
                return None
    
    if dr_dc and dr_dc != (0, 0):
        return {'dr': dr_dc[0], 'dc': dr_dc[1], 'bg': bg}
    return None


def _apply_fixed_displacement(inp, params):
    dr, dc = params['dr'], params['dc']
    bg = params['bg']
    inp_arr = _to_np(inp)
    h, w = inp_arr.shape
    result = np.full_like(inp_arr, bg)
    
    for r in range(h):
        for c in range(w):
            if inp_arr[r, c] != bg:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr, nc] = inp_arr[r, c]
    
    return _from_np(result)


def _learn_gravity_toward(train_pairs, bg):
    """Objects move toward each other or toward center."""
    # TODO: implement gravity toward center/largest object
    return None


def _apply_gravity_toward(inp, params):
    return None


# ============================================================
# 4. Add Pattern (96 tasks)
# ============================================================

def _gen_add_pattern_pieces(train_pairs, bg):
    """Generate pieces for pattern addition."""
    pieces = []
    
    # Strategy: project rays from objects in cardinal directions
    rule = _learn_ray_projection(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'rg:ray_project_{_r["direction"]}',
            lambda inp, r=_r: _apply_ray_projection(inp, r)
        ))
    
    return pieces


def _learn_ray_projection(train_pairs, bg):
    """Learn: project rays from each non-bg cell until hitting another non-bg cell or border."""
    for direction in ['all4', 'h', 'v', 'all8']:
        ok = True
        for inp, out in train_pairs:
            inp_arr = _to_np(inp)
            out_arr = _to_np(out)
            if inp_arr.shape != out_arr.shape:
                ok = False; break
            
            h, w = inp_arr.shape
            result = inp_arr.copy()
            
            if direction == 'all4':
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            elif direction == 'h':
                dirs = [(0, -1), (0, 1)]
            elif direction == 'v':
                dirs = [(-1, 0), (1, 0)]
            else:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
            
            for r in range(h):
                for c in range(w):
                    if inp_arr[r, c] != bg:
                        color = int(inp_arr[r, c])
                        for ddr, ddc in dirs:
                            nr, nc = r + ddr, c + ddc
                            while 0 <= nr < h and 0 <= nc < w:
                                if inp_arr[nr, nc] != bg:
                                    break
                                result[nr, nc] = color
                                nr += ddr
                                nc += ddc
            
            if not np.array_equal(result, out_arr):
                ok = False; break
        
        if ok:
            return {'direction': direction, 'bg': bg}
    return None


def _apply_ray_projection(inp, params):
    bg = params['bg']
    direction = params['direction']
    inp_arr = _to_np(inp)
    h, w = inp_arr.shape
    result = inp_arr.copy()
    
    if direction == 'all4':
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif direction == 'h':
        dirs = [(0, -1), (0, 1)]
    elif direction == 'v':
        dirs = [(-1, 0), (1, 0)]
    else:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for r in range(h):
        for c in range(w):
            if inp_arr[r, c] != bg:
                color = int(inp_arr[r, c])
                for ddr, ddc in dirs:
                    nr, nc = r + ddr, c + ddc
                    while 0 <= nr < h and 0 <= nc < w:
                        if inp_arr[nr, nc] != bg:
                            break
                        result[nr, nc] = color
                        nr += ddr
                        nc += ddc
    
    return _from_np(result)


# ============================================================
# 5. Pure Removal (13 tasks)
# ============================================================

def _gen_removal_pieces(train_pairs, bg):
    """Generate pieces for object removal."""
    pieces = []
    
    try:
        from arc.objects import detect_objects
    except ImportError:
        return pieces
    
    # Strategy: remove objects by color
    for remove_prop in ['smallest', 'largest', 'minority_color', 'unique_color']:
        rule = _learn_remove_by(train_pairs, bg, remove_prop)
        if rule:
            _r = rule
            _rp = remove_prop
            pieces.append(CrossPiece(
                f'rg:remove_{remove_prop}',
                lambda inp, r=_r, rp=_rp: _apply_remove_by(inp, r, rp)
            ))
    
    return pieces


def _learn_remove_by(train_pairs, bg, prop):
    for inp, out in train_pairs:
        from arc.objects import detect_objects
        inp_arr = _to_np(inp)
        out_arr = _to_np(out)
        if inp_arr.shape != out_arr.shape:
            return None
        
        objs = detect_objects(inp, bg)
        if not objs:
            return None
        
        result = inp_arr.copy()
        
        if prop == 'smallest':
            min_size = min(o.size for o in objs)
            for o in objs:
                if o.size == min_size:
                    for r, c in o.cells:
                        result[r, c] = bg
        elif prop == 'largest':
            max_size = max(o.size for o in objs)
            for o in objs:
                if o.size == max_size:
                    for r, c in o.cells:
                        result[r, c] = bg
        elif prop == 'minority_color':
            color_counts = Counter(o.color for o in objs)
            if len(color_counts) < 2:
                return None
            minority = color_counts.most_common()[-1][0]
            for o in objs:
                if o.color == minority:
                    for r, c in o.cells:
                        result[r, c] = bg
        elif prop == 'unique_color':
            color_counts = Counter(o.color for o in objs)
            unique_colors = {c for c, n in color_counts.items() if n == 1}
            if not unique_colors:
                return None
            for o in objs:
                if o.color in unique_colors:
                    for r, c in o.cells:
                        result[r, c] = bg
        
        if not np.array_equal(result, out_arr):
            return None
    
    return {'bg': bg}


def _apply_remove_by(inp, params, prop):
    from arc.objects import detect_objects
    bg = params['bg']
    inp_arr = _to_np(inp)
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    
    result = inp_arr.copy()
    
    if prop == 'smallest':
        min_size = min(o.size for o in objs)
        targets = [o for o in objs if o.size == min_size]
    elif prop == 'largest':
        max_size = max(o.size for o in objs)
        targets = [o for o in objs if o.size == max_size]
    elif prop == 'minority_color':
        color_counts = Counter(o.color for o in objs)
        minority = color_counts.most_common()[-1][0]
        targets = [o for o in objs if o.color == minority]
    elif prop == 'unique_color':
        color_counts = Counter(o.color for o in objs)
        unique_colors = {c for c, n in color_counts.items() if n == 1}
        targets = [o for o in objs if o.color in unique_colors]
    else:
        return None
    
    for o in targets:
        for r, c in o.cells:
            result[r, c] = bg
    
    return _from_np(result)


# ============================================================
# 6. Conditional Transform (mixed: select some objects, transform them)
# ============================================================

def _gen_conditional_transform_pieces(train_pairs, bg):
    """Generate pieces for conditional per-object transforms."""
    pieces = []
    
    try:
        from arc.objects import detect_objects
    except ImportError:
        return pieces
    
    # Strategy: select objects by property, apply a transform to selected ones
    # For now: select by color, transform = remove/recolor/fill_bbox
    
    rule = _learn_select_and_fill(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'rg:select_fill:{_r["selector"]}',
            lambda inp, r=_r: _apply_select_and_fill(inp, r)
        ))
    
    return pieces


def _learn_select_and_fill(train_pairs, bg):
    """Select objects by property, fill their bbox."""
    try:
        from arc.objects import detect_objects
    except ImportError:
        return None
    
    for selector in ['largest', 'smallest', 'most_common_color']:
        ok = True
        for inp, out in train_pairs:
            inp_arr = _to_np(inp)
            out_arr = _to_np(out)
            if inp_arr.shape != out_arr.shape:
                ok = False; break
            
            objs = detect_objects(inp, bg)
            if not objs:
                ok = False; break
            
            result = inp_arr.copy()
            
            if selector == 'largest':
                max_size = max(o.size for o in objs)
                selected = [o for o in objs if o.size == max_size]
            elif selector == 'smallest':
                min_size = min(o.size for o in objs)
                selected = [o for o in objs if o.size == min_size]
            else:
                cc = Counter(o.color for o in objs)
                mc = cc.most_common(1)[0][0]
                selected = [o for o in objs if o.color == mc]
            
            for obj in selected:
                r1, c1, r2, c2 = obj.bbox
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        if result[r, c] == bg:
                            result[r, c] = obj.color
            
            if not np.array_equal(result, out_arr):
                ok = False; break
        
        if ok:
            return {'selector': selector, 'bg': bg}
    
    return None


def _apply_select_and_fill(inp, params):
    from arc.objects import detect_objects
    bg = params['bg']
    selector = params['selector']
    
    inp_arr = _to_np(inp)
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    
    result = inp_arr.copy()
    
    if selector == 'largest':
        max_size = max(o.size for o in objs)
        selected = [o for o in objs if o.size == max_size]
    elif selector == 'smallest':
        min_size = min(o.size for o in objs)
        selected = [o for o in objs if o.size == min_size]
    else:
        cc = Counter(o.color for o in objs)
        mc = cc.most_common(1)[0][0]
        selected = [o for o in objs if o.color == mc]
    
    for obj in selected:
        r1, c1, r2, c2 = obj.bbox
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if result[r, c] == bg:
                    result[r, c] = obj.color
    
    return _from_np(result)
