"""
arc/per_object.py — Per-Object Conditional Transform for ARC-AGI-2

Detects objects, classifies them by property, applies transforms per-object.
Handles:
- Recolor by shape/size/position/neighbor-count
- Move objects based on property
- Scale/rotate/mirror individual objects
- Fill object bbox
- Remove objects by condition
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.objects import detect_objects, ArcObject


# ============================================================
# Object property extractors
# ============================================================

def _obj_shape_key(obj: ArcObject) -> tuple:
    """Normalized shape (translation-invariant)"""
    return obj.shape


def _obj_size(obj: ArcObject) -> int:
    return obj.size


def _obj_bbox_area(obj: ArcObject) -> int:
    return obj.height * obj.width


def _obj_color(obj: ArcObject) -> int:
    return obj.color


def _obj_height(obj: ArcObject) -> int:
    return obj.height


def _obj_width(obj: ArcObject) -> int:
    return obj.width


def _obj_is_square(obj: ArcObject) -> bool:
    return obj.height == obj.width


def _obj_fill_ratio(obj: ArcObject) -> float:
    return obj.size / (obj.height * obj.width) if obj.height * obj.width > 0 else 0


def _obj_position_quadrant(obj: ArcObject, h: int, w: int) -> str:
    cr, cc = obj.center
    return f"{'top' if cr < h/2 else 'bottom'}_{'left' if cc < w/2 else 'right'}"


def _obj_neighbor_count(obj: ArcObject, all_objs: List[ArcObject]) -> int:
    """Count adjacent objects (bbox touches or overlaps)"""
    r1, c1, r2, c2 = obj.bbox
    count = 0
    for other in all_objs:
        if other is obj:
            continue
        or1, oc1, or2, oc2 = other.bbox
        # Check if bboxes are adjacent (within 2 cells)
        if or1 > r2 + 2 or or2 < r1 - 2 or oc1 > c2 + 2 or oc2 < c1 - 2:
            continue
        count += 1
    return count


def _shapes_match_rotated(s1: tuple, s2: tuple) -> bool:
    """Check if two shapes match under 90° rotations"""
    if s1 == s2:
        return True
    # Compute rotations of s1
    cells = list(s1)
    for _ in range(3):
        cells = [(c, -r) for r, c in cells]
        # Normalize
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        cells_norm = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
        if cells_norm == s2:
            return True
    return False


def _shapes_match_with_mirror(s1: tuple, s2: tuple) -> bool:
    """Check if two shapes match under rotation + mirror"""
    if _shapes_match_rotated(s1, s2):
        return True
    # Mirror s1 horizontally
    cells = [(-r, c) for r, c in s1]
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    s1_mirror = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    return _shapes_match_rotated(s1_mirror, s2)


# ============================================================
# Per-object recolor learning
# ============================================================

def learn_per_object_recolor(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn per-object recolor rule: objects keep position/shape, change color based on property.
    
    Returns rule dict or None.
    """
    for bg in [0, most_common_color(train_pairs[0][0])]:
        # Detect objects in each pair
        all_matches = []
        consistent = True
        
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            oh, ow = grid_shape(out)
            if (h, w) != (oh, ow):
                consistent = False
                break
            
            in_objs = detect_objects(inp, bg)
            if not in_objs:
                consistent = False
                break
            
            # For each input object, find its output color
            pair_map = []
            for obj in in_objs:
                # Get output color at object's cells
                out_colors = Counter()
                for r, c in obj.cells:
                    if out[r][c] != bg:
                        out_colors[out[r][c]] += 1
                if not out_colors:
                    # Object was removed
                    pair_map.append((obj, None))
                else:
                    new_color = out_colors.most_common(1)[0][0]
                    pair_map.append((obj, new_color))
            
            all_matches.append((in_objs, pair_map))
        
        if not consistent:
            continue
        
        # Try different property -> color mappings
        strategies = [
            _try_shape_color_rule,
            _try_size_color_rule,
            _try_size_rank_color_rule,
            _try_neighbor_count_color_rule,
            _try_fill_ratio_color_rule,
        ]
        
        for strategy_fn in strategies:
            rule = strategy_fn(all_matches, bg)
            if rule:
                # VERIFY: apply rule to all training pairs
                ok = True
                for inp, out in train_pairs:
                    result = apply_per_object_recolor(inp, rule)
                    if result is None or not grid_eq(result, out):
                        ok = False
                        break
                if ok:
                    return rule
    
    return None


def _try_shape_color_rule(all_matches, bg) -> Optional[Dict]:
    """Map: shape -> output color"""
    shape_to_color = {}
    has_change = False
    
    for in_objs, pair_map in all_matches:
        for obj, new_color in pair_map:
            if new_color is None:
                continue
            if new_color != obj.color:
                has_change = True
            key = obj.shape
            if key in shape_to_color:
                if shape_to_color[key] != new_color:
                    return None
            else:
                shape_to_color[key] = new_color
    
    if not shape_to_color or not has_change:
        return None
    
    # Verify: at least 2 different output colors (otherwise trivial)
    if len(set(shape_to_color.values())) < 2:
        return None
    
    return {'type': 'shape_to_color', 'map': {str(k): v for k, v in shape_to_color.items()}, 
            'bg': bg, '_raw_map': shape_to_color}


def _try_size_color_rule(all_matches, bg) -> Optional[Dict]:
    """Map: size -> output color"""
    size_to_color = {}
    has_change = False
    for in_objs, pair_map in all_matches:
        for obj, new_color in pair_map:
            if new_color is None:
                continue
            if new_color != obj.color:
                has_change = True
            key = obj.size
            if key in size_to_color:
                if size_to_color[key] != new_color:
                    return None
            else:
                size_to_color[key] = new_color
    
    if not size_to_color or len(size_to_color) < 2 or not has_change:
        return None
    
    return {'type': 'size_to_color', 'map': size_to_color, 'bg': bg}


def _try_size_rank_color_rule(all_matches, bg) -> Optional[Dict]:
    """Map: size rank (0=smallest) -> output color"""
    rank_to_color = {}
    for in_objs, pair_map in all_matches:
        sorted_objs = sorted(pair_map, key=lambda x: x[0].size)
        for rank, (obj, new_color) in enumerate(sorted_objs):
            if new_color is None:
                continue
            if rank in rank_to_color:
                if rank_to_color[rank] != new_color:
                    return None
            else:
                rank_to_color[rank] = new_color
    
    if not rank_to_color:
        return None
    
    return {'type': 'size_rank_to_color', 'map': rank_to_color, 'bg': bg}


def _try_neighbor_count_color_rule(all_matches, bg) -> Optional[Dict]:
    """Map: neighbor count -> output color"""
    nc_to_color = {}
    for in_objs, pair_map in all_matches:
        for obj, new_color in pair_map:
            if new_color is None:
                continue
            nc = _obj_neighbor_count(obj, in_objs)
            if nc in nc_to_color:
                if nc_to_color[nc] != new_color:
                    return None
            else:
                nc_to_color[nc] = new_color
    
    if not nc_to_color or len(nc_to_color) < 2:
        return None
    
    return {'type': 'neighbor_count_to_color', 'map': nc_to_color, 'bg': bg}


def _try_fill_ratio_color_rule(all_matches, bg) -> Optional[Dict]:
    """Map: solid (ratio=1.0) vs hollow -> color"""
    fill_to_color = {}  # 'solid' or 'hollow' -> color
    for in_objs, pair_map in all_matches:
        for obj, new_color in pair_map:
            if new_color is None:
                continue
            fr = _obj_fill_ratio(obj)
            key = 'solid' if fr >= 0.99 else 'hollow'
            if key in fill_to_color:
                if fill_to_color[key] != new_color:
                    return None
            else:
                fill_to_color[key] = new_color
    
    if not fill_to_color or len(fill_to_color) < 2:
        return None
    
    return {'type': 'fill_ratio_to_color', 'map': fill_to_color, 'bg': bg}


def apply_per_object_recolor(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply learned per-object recolor rule"""
    bg = params['bg']
    rule_type = params['type']
    
    h, w = grid_shape(inp)
    in_objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    
    if rule_type == 'shape_to_color':
        raw_map = params.get('_raw_map', {})
        for obj in in_objs:
            key = obj.shape
            if key in raw_map:
                new_color = raw_map[key]
                for r, c in obj.cells:
                    result[r][c] = new_color
    
    elif rule_type == 'size_to_color':
        m = params['map']
        for obj in in_objs:
            if obj.size in m:
                for r, c in obj.cells:
                    result[r][c] = m[obj.size]
    
    elif rule_type == 'size_rank_to_color':
        m = params['map']
        sorted_objs = sorted(in_objs, key=lambda o: o.size)
        for rank, obj in enumerate(sorted_objs):
            if rank in m:
                for r, c in obj.cells:
                    result[r][c] = m[rank]
    
    elif rule_type == 'neighbor_count_to_color':
        m = params['map']
        for obj in in_objs:
            nc = _obj_neighbor_count(obj, in_objs)
            if nc in m:
                for r, c in obj.cells:
                    result[r][c] = m[nc]
    
    elif rule_type == 'fill_ratio_to_color':
        m = params['map']
        for obj in in_objs:
            fr = _obj_fill_ratio(obj)
            key = 'solid' if fr >= 0.99 else 'hollow'
            if key in m:
                for r, c in obj.cells:
                    result[r][c] = m[key]
    
    return result


# ============================================================
# Per-object fill bbox
# ============================================================

def learn_fill_object_bbox(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: fill each object's bounding box"""
    for bg in [0, most_common_color(train_pairs[0][0])]:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            oh, ow = grid_shape(out)
            if (h, w) != (oh, ow):
                ok = False; break
            
            objs = detect_objects(inp, bg)
            result = [row[:] for row in inp]
            for obj in objs:
                r1, c1, r2, c2 = obj.bbox
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        result[r][c] = obj.color
            
            if not grid_eq(result, out):
                ok = False; break
        
        if ok:
            return {'bg': bg, 'type': 'fill_bbox'}
    return None


def apply_fill_object_bbox(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    for obj in objs:
        r1, c1, r2, c2 = obj.bbox
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                result[r][c] = obj.color
    return result


# ============================================================
# Per-object: remove by property
# ============================================================

def learn_remove_objects(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn which objects get removed (replaced with bg)"""
    for bg in [0, most_common_color(train_pairs[0][0])]:
        # Find which property determines removal
        for prop in ['smallest', 'largest', 'by_color']:
            ok = True
            remove_colors = set()
            
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                if grid_shape(out) != (h, w):
                    ok = False; break
                
                objs = detect_objects(inp, bg)
                if not objs:
                    ok = False; break
                
                # Determine which objects were removed
                for obj in objs:
                    obj_still_exists = any(out[r][c] != bg for r, c in obj.cells)
                    
                    if prop == 'smallest':
                        min_size = min(o.size for o in objs)
                        should_remove = (obj.size == min_size)
                    elif prop == 'largest':
                        max_size = max(o.size for o in objs)
                        should_remove = (obj.size == max_size)
                    elif prop == 'by_color':
                        remove_colors.add(obj.color) if not obj_still_exists else None
                        should_remove = not obj_still_exists
                    else:
                        should_remove = False
                    
                    if should_remove and obj_still_exists:
                        ok = False; break
                    if not should_remove and not obj_still_exists:
                        ok = False; break
                
                if not ok:
                    break
            
            if ok:
                return {'bg': bg, 'remove_by': prop, 'remove_colors': list(remove_colors)}
    
    return None


def apply_remove_objects(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    prop = params['remove_by']
    
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    
    if prop == 'smallest':
        if not objs:
            return result
        min_size = min(o.size for o in objs)
        for obj in objs:
            if obj.size == min_size:
                for r, c in obj.cells:
                    result[r][c] = bg
    
    elif prop == 'largest':
        if not objs:
            return result
        max_size = max(o.size for o in objs)
        for obj in objs:
            if obj.size == max_size:
                for r, c in obj.cells:
                    result[r][c] = bg
    
    elif prop == 'by_color':
        remove_colors = set(params.get('remove_colors', []))
        for obj in objs:
            if obj.color in remove_colors:
                for r, c in obj.cells:
                    result[r][c] = bg
    
    return result


# ============================================================
# Cross-projection from dots  
# ============================================================

def learn_cross_projection(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: project lines from dots in 4 directions (cross pattern)"""
    for bg in [0]:
        for project_color_mode in ['same', 'fixed']:
            ok = True
            fixed_color = None
            
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                if grid_shape(out) != (h, w):
                    ok = False; break
                
                # Find dot cells (single non-bg cells)
                dots = []
                for r in range(h):
                    for c in range(w):
                        if inp[r][c] != bg:
                            dots.append((r, c, inp[r][c]))
                
                if not dots:
                    ok = False; break
                
                # Apply cross projection
                result = [row[:] for row in inp]
                for dr, dc, dot_color in dots:
                    proj_color = dot_color if project_color_mode == 'same' else fixed_color
                    if proj_color is None:
                        # Determine from output
                        # Check what color fills the cross lines
                        for r2 in range(h):
                            if r2 != dr and out[r2][dc] != bg and out[r2][dc] != inp[r2][dc]:
                                fixed_color = out[r2][dc]
                                proj_color = fixed_color
                                break
                    
                    if proj_color is None:
                        proj_color = dot_color
                    
                    # Project in 4 directions
                    for r2 in range(h):
                        if result[r2][dc] == bg:
                            result[r2][dc] = proj_color
                    for c2 in range(w):
                        if result[dr][c2] == bg:
                            result[dr][c2] = proj_color
                
                if not grid_eq(result, out):
                    ok = False; break
            
            if ok:
                return {'bg': bg, 'color_mode': project_color_mode, 'fixed_color': fixed_color}
    
    return None


def apply_cross_projection(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    color_mode = params['color_mode']
    fixed_color = params.get('fixed_color')
    
    h, w = grid_shape(inp)
    dots = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                dots.append((r, c, inp[r][c]))
    
    result = [row[:] for row in inp]
    for dr, dc, dot_color in dots:
        proj_color = dot_color if color_mode == 'same' else (fixed_color or dot_color)
        for r2 in range(h):
            if result[r2][dc] == bg:
                result[r2][dc] = proj_color
        for c2 in range(w):
            if result[dr][c2] == bg:
                result[dr][c2] = proj_color
    
    return result


# ============================================================
# Object-based extraction (output = specific object)
# ============================================================

def learn_extract_object(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: output is a specific object extracted from input.
    
    Selection criteria: unique shape, specific color, largest non-bg, etc.
    """
    for bg in [0, most_common_color(train_pairs[0][0])]:
        # Try: extract object with unique color
        for selector in ['unique_color', 'minority_color', 'largest', 'smallest',
                         'most_colors_mc', 'has_specific_color']:
            ok = True
            sel_params = {}
            
            for inp, out in train_pairs:
                objs = detect_objects(inp, bg, multicolor=True)
                if not objs:
                    ok = False; break
                
                oh, ow = grid_shape(out)
                
                selected = None
                
                if selector == 'largest':
                    selected = max(objs, key=lambda o: o.size)
                elif selector == 'smallest':
                    selected = min(objs, key=lambda o: o.size)
                elif selector == 'minority_color':
                    color_counts = Counter(o.color for o in objs)
                    if color_counts:
                        min_color = min(color_counts, key=color_counts.get)
                        candidates = [o for o in objs if o.color == min_color]
                        if len(candidates) == 1:
                            selected = candidates[0]
                
                if selected is None:
                    ok = False; break
                
                # Check if extracted object matches output
                r1, c1, r2, c2 = selected.bbox
                extracted = [inp[r][c1:c2+1] for r in range(r1, r2+1)]
                if not grid_eq(extracted, out):
                    ok = False; break
            
            if ok:
                return {'bg': bg, 'selector': selector, **sel_params}
    
    return None


def apply_extract_object(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    selector = params['selector']
    
    objs = detect_objects(inp, bg, multicolor=True)
    if not objs:
        return None
    
    selected = None
    if selector == 'largest':
        selected = max(objs, key=lambda o: o.size)
    elif selector == 'smallest':
        selected = min(objs, key=lambda o: o.size)
    elif selector == 'minority_color':
        color_counts = Counter(o.color for o in objs)
        if color_counts:
            min_color = min(color_counts, key=color_counts.get)
            candidates = [o for o in objs if o.color == min_color]
            if len(candidates) == 1:
                selected = candidates[0]
    
    if selected is None:
        return None
    
    r1, c1, r2, c2 = selected.bbox
    return [inp[r][c1:c2+1] for r in range(r1, r2+1)]


# ============================================================
# Mask application: use one region as mask for another
# ============================================================

def learn_mask_apply(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: split grid into regions, use one as mask for another.
    
    Common pattern: grid has 2-3 distinct colored regions.
    One acts as a template/mask, another provides content.
    """
    # This is a complex pattern - try basic version:
    # Grid split into 2 halves, one masks the other
    for bg in [0]:
        for split_dir in ['h', 'v']:
            ok = True
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                oh, ow = grid_shape(out)
                
                if split_dir == 'h' and h % 2 == 0:
                    half = h // 2
                    top = [inp[r][:] for r in range(half)]
                    bottom = [inp[r][:] for r in range(half, h)]
                    
                    if (half, w) != (oh, ow):
                        ok = False; break
                    
                    # Try: mask = where top != bg, content = bottom color
                    result = [[bg] * w for _ in range(half)]
                    for r in range(half):
                        for c in range(w):
                            if top[r][c] != bg:
                                result[r][c] = bottom[r][c] if bottom[r][c] != bg else top[r][c]
                            else:
                                result[r][c] = bg
                    
                    if not grid_eq(result, out):
                        # Try reverse
                        result = [[bg] * w for _ in range(half)]
                        for r in range(half):
                            for c in range(w):
                                if bottom[r][c] != bg:
                                    result[r][c] = top[r][c] if top[r][c] != bg else bottom[r][c]
                                else:
                                    result[r][c] = bg
                        
                        if not grid_eq(result, out):
                            ok = False; break
                
                elif split_dir == 'v' and w % 2 == 0:
                    half = w // 2
                    left = [inp[r][:half] for r in range(h)]
                    right = [inp[r][half:] for r in range(h)]
                    
                    if (h, half) != (oh, ow):
                        ok = False; break
                    
                    result = [[bg] * half for _ in range(h)]
                    for r in range(h):
                        for c in range(half):
                            if left[r][c] != bg:
                                result[r][c] = right[r][c] if right[r][c] != bg else left[r][c]
                            else:
                                result[r][c] = bg
                    
                    if not grid_eq(result, out):
                        ok = False; break
                else:
                    ok = False; break
            
            if ok:
                return {'bg': bg, 'split_dir': split_dir}
    
    return None


def apply_mask_apply(inp: Grid, params: Dict) -> Optional[Grid]:
    bg = params['bg']
    split_dir = params['split_dir']
    h, w = grid_shape(inp)
    
    if split_dir == 'h' and h % 2 == 0:
        half = h // 2
        top = [inp[r][:] for r in range(half)]
        bottom = [inp[r][:] for r in range(half, h)]
        result = [[bg] * w for _ in range(half)]
        for r in range(half):
            for c in range(w):
                if top[r][c] != bg:
                    result[r][c] = bottom[r][c] if bottom[r][c] != bg else top[r][c]
                else:
                    result[r][c] = bg
        return result
    
    elif split_dir == 'v' and w % 2 == 0:
        half = w // 2
        left = [inp[r][:half] for r in range(h)]
        right = [inp[r][half:] for r in range(h)]
        result = [[bg] * half for _ in range(h)]
        for r in range(h):
            for c in range(half):
                if left[r][c] != bg:
                    result[r][c] = right[r][c] if right[r][c] != bg else left[r][c]
                else:
                    result[r][c] = bg
        return result
    
    return None


# ============================================================
# Holes-to-color: recolor objects based on number of enclosed holes
# ============================================================

def _count_enclosed_holes(mask_2d):
    """Count number of enclosed (interior) holes in a binary mask."""
    import numpy as np
    from scipy import ndimage
    inv = ~mask_2d
    border_labeled, bn = ndimage.label(inv)
    if bn == 0:
        return 0
    border_labels = set()
    h, w = mask_2d.shape
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and inv[r, c]:
                border_labels.add(border_labeled[r, c])
    return bn - len(border_labels)


def learn_holes_to_color(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn holes→color mapping from training pairs."""
    import numpy as np
    from scipy import ndimage
    from arc.grid import most_common_color

    bg = most_common_color(train_pairs[0][0])
    mapping = {}

    for inp, out in train_pairs:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        labeled, n = ndimage.label(inp_arr != bg)
        for oid in range(1, n + 1):
            mask = labeled == oid
            rows, cols = np.where(mask)
            r0, c0 = rows.min(), cols.min()
            local = mask[r0:rows.max() + 1, c0:cols.max() + 1]
            nh = _count_enclosed_holes(local)

            out_colors = set(int(out_arr[r, c]) for r, c in zip(rows, cols))
            if len(out_colors) != 1:
                return None
            oc = list(out_colors)[0]

            if nh in mapping:
                if mapping[nh] != oc:
                    return None
            else:
                mapping[nh] = oc

    if not mapping:
        return None
    return {'type': 'holes_to_color', 'mapping': mapping, 'bg': bg}


def apply_holes_to_color(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply holes→color recoloring."""
    import numpy as np
    from scipy import ndimage

    mapping = params['mapping']
    bg = params['bg']
    inp_arr = np.array(inp)
    labeled, n = ndimage.label(inp_arr != bg)
    result = inp_arr.copy()

    for oid in range(1, n + 1):
        mask = labeled == oid
        rows, cols = np.where(mask)
        r0, c0 = rows.min(), cols.min()
        local = mask[r0:rows.max() + 1, c0:cols.max() + 1]
        nh = _count_enclosed_holes(local)

        if nh not in mapping:
            return None
        result[rows, cols] = mapping[nh]

    return result.tolist()


# ============================================================
# Cluster histogram: output = histogram of cluster counts per color
# ============================================================

def learn_cluster_histogram(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn cluster-count histogram rule: sort colors by n_clusters desc,
    output row i = color[i], right-aligned with n_clusters[i] cells."""
    import numpy as np
    from scipy import ndimage
    from arc.grid import most_common_color, grid_eq

    bg = most_common_color(train_pairs[0][0])

    for inp, out in train_pairs:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        oh, ow = out_arr.shape

        non_bg = sorted(set(int(v) for v in inp_arr.flatten() if v != bg))
        if not non_bg:
            return None
        if len(non_bg) != oh:
            return None

        cc = {}
        for color in non_bg:
            _, nc = ndimage.label(inp_arr == color)
            cc[color] = nc

        sorted_colors = sorted(non_bg, key=lambda c: -cc[c])
        max_nc = max(cc.values())

        if max_nc != ow:
            return None

        # Verify right-aligned histogram
        expected = np.full((oh, ow), bg, dtype=int)
        for ri, color in enumerate(sorted_colors):
            nc = cc[color]
            expected[ri, ow - nc:] = color

        if not np.array_equal(expected, out_arr):
            return None

    return {'type': 'cluster_histogram', 'bg': bg}


def apply_cluster_histogram(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply cluster histogram transformation."""
    import numpy as np
    from scipy import ndimage

    bg = params['bg']
    inp_arr = np.array(inp)
    non_bg = sorted(set(int(v) for v in inp_arr.flatten() if v != bg))
    if not non_bg:
        return None

    cc = {}
    for color in non_bg:
        _, nc = ndimage.label(inp_arr == color)
        cc[color] = nc

    sorted_colors = sorted(non_bg, key=lambda c: -cc[c])
    n = len(sorted_colors)
    mx = max(cc.values())

    out = np.full((n, mx), bg, dtype=int)
    for ri, color in enumerate(sorted_colors):
        nc = cc[color]
        out[ri, mx - nc:] = color

    return out.tolist()


# ============================================================
# Recolor each object to the color of its nearest neighboring object
# ============================================================

def learn_recolor_by_nearest_object(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: each object is recolored to the color of its nearest other object."""
    import numpy as np
    from scipy import ndimage
    from arc.grid import most_common_color

    bg = most_common_color(train_pairs[0][0])

    for inp, out in train_pairs:
        inp_arr = np.array(inp)
        out_arr = np.array(out)
        if inp_arr.shape != out_arr.shape:
            return None

        labeled, n = ndimage.label(inp_arr != bg)
        if n < 2:
            return None

        objs = []
        for oid in range(1, n + 1):
            mask = labeled == oid
            rows, cols = np.where(mask)
            cin_set = set(int(inp_arr[r, c]) for r, c in zip(rows, cols))
            cout_set = set(int(out_arr[r, c]) for r, c in zip(rows, cols))
            if len(cin_set) != 1 or len(cout_set) != 1:
                return None
            objs.append({
                'cin': list(cin_set)[0],
                'cout': list(cout_set)[0],
                'cells': list(zip(rows.tolist(), cols.tolist())),
            })

        # For each object, check cout == nearest other object's cin
        for i, obj in enumerate(objs):
            nearest_color = None
            nearest_dist = float('inf')
            for j, other in enumerate(objs):
                if i == j:
                    continue
                for r, c in other['cells']:
                    for r2, c2 in obj['cells']:
                        d = abs(r - r2) + abs(c - c2)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_color = other['cin']

            if nearest_color != obj['cout']:
                return None

    return {'type': 'recolor_by_nearest_object', 'bg': bg}


def apply_recolor_by_nearest_object(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply: recolor each object to its nearest other object's color."""
    import numpy as np
    from scipy import ndimage

    bg = params['bg']
    inp_arr = np.array(inp)
    labeled, n = ndimage.label(inp_arr != bg)
    if n < 2:
        return None

    objs = []
    for oid in range(1, n + 1):
        mask = labeled == oid
        rows, cols = np.where(mask)
        cin_set = set(int(inp_arr[r, c]) for r, c in zip(rows, cols))
        if len(cin_set) != 1:
            return None
        objs.append({
            'cin': list(cin_set)[0],
            'cells': list(zip(rows.tolist(), cols.tolist())),
        })

    result = inp_arr.copy()
    for i, obj in enumerate(objs):
        nearest_color = None
        nearest_dist = float('inf')
        for j, other in enumerate(objs):
            if i == j:
                continue
            for r, c in other['cells']:
                for r2, c2 in obj['cells']:
                    d = abs(r - r2) + abs(c - c2)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_color = other['cin']

        if nearest_color is None:
            return None
        for r, c in obj['cells']:
            result[r, c] = nearest_color

    return result.tolist()
