"""
arc/obj_correspondence.py — Object Correspondence + Conditional Transform

Cross-Structure approach:
  Piece 1: Object detection + correspondence (input↔output matching)
  Piece 2: Transform classification per object (identity/recolor/move/delete)
  Piece 3: Condition learning (WHY does this object get transformed?)
  Cross product: Condition × Transform → Program

Addresses 459 same-size multi-object tasks.
"""

from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors
from arc.objects import detect_objects, find_matching_objects, ArcObject


def learn_object_program(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn an object-level transformation program from training pairs.
    
    Returns a rule dict with:
      - selector: which objects to transform (condition)
      - transform: what to do (recolor, move, delete)
      - params: transform-specific parameters
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    # Try multiple strategies
    for strategy in [
        _learn_selective_recolor,
        _learn_conditional_recolor_by_size,
        _learn_conditional_recolor_by_neighbor,
        _learn_conditional_recolor_by_position,
        _learn_selective_delete,
        _learn_object_fill_between,
        _learn_recolor_by_containment,
    ]:
        rule = strategy(train_pairs, bg)
        if rule is not None:
            # Full verification
            ok = True
            for inp, out in train_pairs:
                result = apply_object_program(inp, rule)
                if result is None or not grid_eq(result, out):
                    ok = False
                    break
            if ok:
                return rule
    
    return None


def _learn_selective_recolor(train_pairs, bg):
    """Learn: objects matching a condition get recolored to a specific color."""
    
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        if len(matches) < 2:
            return None
    
    # Collect per-pair recolor info
    all_recolored = []
    all_unchanged = []
    target_color = None
    
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        
        recolored = []
        unchanged = []
        for oi, oo in matches:
            if oi.color != oo.color and oi.shape == oo.shape and oi.bbox == oo.bbox:
                recolored.append(oi)
                if target_color is None:
                    target_color = oo.color
                elif target_color != oo.color:
                    return None  # inconsistent target color
            elif oi.color == oo.color and oi.shape == oo.shape:
                unchanged.append(oi)
        
        if not recolored:
            return None
        all_recolored.append(recolored)
        all_unchanged.append(unchanged)
    
    # Find condition that separates recolored from unchanged
    condition = _find_separator(all_recolored, all_unchanged, train_pairs, bg)
    if condition is None:
        return None
    
    return {
        'type': 'selective_recolor',
        'condition': condition,
        'target_color': target_color,
        'name': f'recolor_{condition["type"]}_to_{target_color}',
    }


def _learn_conditional_recolor_by_size(train_pairs, bg):
    """Recolor objects based on size comparison (smallest/largest/specific size)."""
    
    for selector in ['smallest', 'largest']:
        consistent = True
        target_color = None
        
        for inp, out in train_pairs:
            in_objs = detect_objects(inp, bg)
            out_objs = detect_objects(out, bg)
            matches = find_matching_objects(in_objs, out_objs)
            
            if not matches:
                consistent = False; break
            
            if selector == 'smallest':
                target_size = min(oi.size for oi, _ in matches)
            else:
                target_size = max(oi.size for oi, _ in matches)
            
            for oi, oo in matches:
                if oi.size == target_size:
                    if oi.color != oo.color:
                        if target_color is None:
                            target_color = oo.color
                        elif target_color != oo.color:
                            consistent = False; break
                else:
                    if oi.color != oo.color:
                        consistent = False; break
            if not consistent:
                break
        
        if consistent and target_color is not None:
            return {
                'type': 'selective_recolor',
                'condition': {'type': f'{selector}_size'},
                'target_color': target_color,
                'name': f'recolor_{selector}_to_{target_color}',
            }
    
    return None


def _learn_conditional_recolor_by_neighbor(train_pairs, bg):
    """Recolor objects based on their neighbor count or adjacency."""
    
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        if len(matches) < 2:
            return None
    
    # Try: recolor based on whether object touches border
    consistent = True
    target_color = None
    selector = 'touches_border'
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        
        for oi, oo in matches:
            touches = any(r == 0 or r == h-1 or c == 0 or c == w-1 for r, c in oi.cells)
            if oi.color != oo.color:
                if target_color is None:
                    target_color = oo.color
                elif target_color != oo.color:
                    consistent = False; break
                if not touches:
                    consistent = False; break
            else:
                if touches:
                    consistent = False; break
        if not consistent:
            break
    
    if consistent and target_color is not None:
        return {
            'type': 'selective_recolor',
            'condition': {'type': 'touches_border'},
            'target_color': target_color,
            'name': f'recolor_border_to_{target_color}',
        }
    
    # Try: recolor based on NOT touching border
    consistent = True
    target_color = None
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        
        for oi, oo in matches:
            touches = any(r == 0 or r == h-1 or c == 0 or c == w-1 for r, c in oi.cells)
            if oi.color != oo.color:
                if target_color is None:
                    target_color = oo.color
                elif target_color != oo.color:
                    consistent = False; break
                if touches:
                    consistent = False; break
            else:
                if not touches:
                    consistent = False; break
        if not consistent:
            break
    
    if consistent and target_color is not None:
        return {
            'type': 'selective_recolor',
            'condition': {'type': 'not_touches_border'},
            'target_color': target_color,
            'name': f'recolor_interior_to_{target_color}',
        }
    
    return None


def _learn_conditional_recolor_by_position(train_pairs, bg):
    """Recolor objects based on relative position (leftmost, rightmost, top, bottom)."""
    
    for selector in ['leftmost', 'rightmost', 'topmost', 'bottommost']:
        consistent = True
        target_color = None
        
        for inp, out in train_pairs:
            in_objs = detect_objects(inp, bg)
            out_objs = detect_objects(out, bg)
            matches = find_matching_objects(in_objs, out_objs)
            if not matches:
                consistent = False; break
            
            if selector == 'leftmost':
                target_obj = min(matches, key=lambda m: m[0].bbox[1])
            elif selector == 'rightmost':
                target_obj = max(matches, key=lambda m: m[0].bbox[3])
            elif selector == 'topmost':
                target_obj = min(matches, key=lambda m: m[0].bbox[0])
            else:
                target_obj = max(matches, key=lambda m: m[0].bbox[2])
            
            oi, oo = target_obj
            if oi.color != oo.color:
                if target_color is None:
                    target_color = oo.color
                elif target_color != oo.color:
                    consistent = False; break
            else:
                consistent = False; break
            
            # All other objects should be unchanged
            for oi2, oo2 in matches:
                if oi2 is oi:
                    continue
                if oi2.color != oo2.color:
                    consistent = False; break
            if not consistent:
                break
        
        if consistent and target_color is not None:
            return {
                'type': 'selective_recolor',
                'condition': {'type': selector},
                'target_color': target_color,
                'name': f'recolor_{selector}_to_{target_color}',
            }
    
    return None


def _learn_selective_delete(train_pairs, bg):
    """Learn: delete objects matching a condition (they become bg in output)."""
    
    for selector in ['smallest_obj', 'largest_obj', 'unique_color', 'duplicate_shape']:
        consistent = True
        
        for inp, out in train_pairs:
            in_objs = detect_objects(inp, bg)
            out_objs = detect_objects(out, bg)
            
            if selector == 'smallest_obj':
                min_size = min(o.size for o in in_objs) if in_objs else 0
                to_delete = [o for o in in_objs if o.size == min_size]
            elif selector == 'largest_obj':
                max_size = max(o.size for o in in_objs) if in_objs else 0
                to_delete = [o for o in in_objs if o.size == max_size]
            elif selector == 'unique_color':
                color_counts = Counter(o.color for o in in_objs)
                to_delete = [o for o in in_objs if color_counts[o.color] == 1]
            elif selector == 'duplicate_shape':
                shape_counts = Counter(o.shape for o in in_objs)
                to_delete = [o for o in in_objs if shape_counts[o.shape] > 1]
            else:
                to_delete = []
            
            if not to_delete:
                consistent = False; break
            
            # Check: deleting these objects + keeping others = output
            h, w = grid_shape(inp)
            result = [row[:] for row in inp]
            for obj in to_delete:
                for r, c in obj.cells:
                    result[r][c] = bg
            
            if not grid_eq(result, out):
                consistent = False; break
        
        if consistent:
            return {
                'type': 'selective_delete',
                'condition': {'type': selector},
                'name': f'delete_{selector}',
            }
    
    return None


def _learn_object_fill_between(train_pairs, bg):
    """Fill bg cells between objects with the nearest object's color."""
    
    consistent = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        in_objs = detect_objects(inp, bg)
        if len(in_objs) < 2:
            consistent = False; break
        
        # Build result: each bg cell gets the nearest object's color
        result = [row[:] for row in inp]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    continue
                # Skip if output is still bg
                if out[r][c] == bg:
                    continue
                    
                best_dist = float('inf')
                best_color = bg
                for obj in in_objs:
                    for or_, oc in obj.cells:
                        d = abs(r - or_) + abs(c - oc)
                        if d < best_dist:
                            best_dist = d
                            best_color = obj.color
                
                if best_color != out[r][c]:
                    consistent = False; break
                result[r][c] = best_color
            if not consistent:
                break
        
        if not consistent:
            break
        if not grid_eq(result, out):
            consistent = False; break
    
    if consistent:
        return {
            'type': 'fill_nearest_object',
            'name': 'fill_nearest_object_color',
        }
    
    return None


def _learn_recolor_by_containment(train_pairs, bg):
    """Recolor small objects based on which larger object contains them."""
    
    consistent = True
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        
        for oi, oo in matches:
            if oi.color == oo.color:
                continue
            
            # Find enclosing object
            r1, c1, r2, c2 = oi.bbox
            enclosing = None
            for other in in_objs:
                if other is oi:
                    continue
                or1, oc1, or2, oc2 = other.bbox
                if or1 <= r1 and oc1 <= c1 and or2 >= r2 and oc2 >= c2:
                    enclosing = other
                    break
            
            if enclosing is None:
                consistent = False; break
            
            # Check: new color = enclosing object's color?
            if oo.color != enclosing.color:
                consistent = False; break
        
        if not consistent:
            break
    
    if consistent:
        return {
            'type': 'recolor_by_containment',
            'name': 'recolor_to_enclosing_color',
        }
    
    return None


def _find_separator(recolored_groups, unchanged_groups, train_pairs, bg):
    """Find a condition that separates recolored objects from unchanged ones."""
    
    # Try: unique color (color appears only once)
    consistent = True
    for recolored, unchanged, (inp, _) in zip(recolored_groups, unchanged_groups, train_pairs):
        all_objs = detect_objects(inp, bg)
        color_counts = Counter(o.color for o in all_objs)
        for obj in recolored:
            if color_counts[obj.color] != 1:
                consistent = False; break
        for obj in unchanged:
            if color_counts[obj.color] == 1:
                consistent = False; break
        if not consistent:
            break
    if consistent:
        return {'type': 'unique_color'}
    
    # Try: unique shape
    consistent = True
    for recolored, unchanged, (inp, _) in zip(recolored_groups, unchanged_groups, train_pairs):
        all_objs = detect_objects(inp, bg)
        shape_counts = Counter(o.shape for o in all_objs)
        for obj in recolored:
            if shape_counts[obj.shape] != 1:
                consistent = False; break
        for obj in unchanged:
            if shape_counts[obj.shape] == 1:
                consistent = False; break
        if not consistent:
            break
    if consistent:
        return {'type': 'unique_shape'}
    
    # Try: smallest size
    consistent = True
    for recolored, unchanged, (inp, _) in zip(recolored_groups, unchanged_groups, train_pairs):
        all_objs = detect_objects(inp, bg)
        min_size = min(o.size for o in all_objs)
        for obj in recolored:
            if obj.size != min_size:
                consistent = False; break
        if not consistent:
            break
    if consistent:
        return {'type': 'smallest_size'}
    
    # Try: largest size
    consistent = True
    for recolored, unchanged, (inp, _) in zip(recolored_groups, unchanged_groups, train_pairs):
        all_objs = detect_objects(inp, bg)
        max_size = max(o.size for o in all_objs)
        for obj in recolored:
            if obj.size != max_size:
                consistent = False; break
        if not consistent:
            break
    if consistent:
        return {'type': 'largest_size'}
    
    return None


def apply_object_program(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply an object-level transformation program."""
    bg = most_common_color(inp)
    h, w = grid_shape(inp)
    in_objs = detect_objects(inp, bg)
    
    if not in_objs:
        return inp
    
    rtype = rule['type']
    
    if rtype == 'selective_recolor':
        condition = rule['condition']
        target_color = rule['target_color']
        result = [row[:] for row in inp]
        
        selected = _select_objects(in_objs, condition, inp, bg)
        for obj in selected:
            for r, c in obj.cells:
                result[r][c] = target_color
        return result
    
    elif rtype == 'selective_delete':
        condition = rule['condition']
        result = [row[:] for row in inp]
        selected = _select_objects(in_objs, condition, inp, bg)
        for obj in selected:
            for r, c in obj.cells:
                result[r][c] = bg
        return result
    
    elif rtype == 'fill_nearest_object':
        result = [row[:] for row in inp]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    continue
                best_dist = float('inf')
                best_color = bg
                for obj in in_objs:
                    for or_, oc in obj.cells:
                        d = abs(r - or_) + abs(c - oc)
                        if d < best_dist:
                            best_dist = d
                            best_color = obj.color
                result[r][c] = best_color
        return result
    
    elif rtype == 'recolor_by_containment':
        result = [row[:] for row in inp]
        for obj in in_objs:
            r1, c1, r2, c2 = obj.bbox
            for other in in_objs:
                if other is obj:
                    continue
                or1, oc1, or2, oc2 = other.bbox
                if or1 <= r1 and oc1 <= c1 and or2 >= r2 and oc2 >= c2:
                    for r, c in obj.cells:
                        result[r][c] = other.color
                    break
        return result
    
    return None


def _select_objects(objs: List[ArcObject], condition: Dict, 
                    inp: Grid, bg: int) -> List[ArcObject]:
    """Select objects matching a condition."""
    h, w = grid_shape(inp)
    ctype = condition['type']
    
    if ctype == 'unique_color':
        color_counts = Counter(o.color for o in objs)
        return [o for o in objs if color_counts[o.color] == 1]
    
    elif ctype == 'unique_shape':
        shape_counts = Counter(o.shape for o in objs)
        return [o for o in objs if shape_counts[o.shape] == 1]
    
    elif ctype == 'smallest_size':
        min_size = min(o.size for o in objs)
        return [o for o in objs if o.size == min_size]
    
    elif ctype == 'largest_size':
        max_size = max(o.size for o in objs)
        return [o for o in objs if o.size == max_size]
    
    elif ctype == 'smallest_obj':
        min_size = min(o.size for o in objs)
        return [o for o in objs if o.size == min_size]
    
    elif ctype == 'largest_obj':
        max_size = max(o.size for o in objs)
        return [o for o in objs if o.size == max_size]
    
    elif ctype == 'touches_border':
        return [o for o in objs if any(r == 0 or r == h-1 or c == 0 or c == w-1 
                                        for r, c in o.cells)]
    
    elif ctype == 'not_touches_border':
        return [o for o in objs if not any(r == 0 or r == h-1 or c == 0 or c == w-1 
                                            for r, c in o.cells)]
    
    elif ctype == 'leftmost':
        min_col = min(o.bbox[1] for o in objs)
        return [o for o in objs if o.bbox[1] == min_col]
    
    elif ctype == 'rightmost':
        max_col = max(o.bbox[3] for o in objs)
        return [o for o in objs if o.bbox[3] == max_col]
    
    elif ctype == 'topmost':
        min_row = min(o.bbox[0] for o in objs)
        return [o for o in objs if o.bbox[0] == min_row]
    
    elif ctype == 'bottommost':
        max_row = max(o.bbox[2] for o in objs)
        return [o for o in objs if o.bbox[2] == max_row]
    
    elif ctype == 'duplicate_shape':
        shape_counts = Counter(o.shape for o in objs)
        return [o for o in objs if shape_counts[o.shape] > 1]
    
    return []
