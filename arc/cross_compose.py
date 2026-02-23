"""
arc/cross_compose.py — Cross-Structure Multi-Step Composition

Instead of enumerating step combinations, decompose the transform into
Cross pieces that constrain each other:

  Axis 1: WHAT changes (diff analysis → change_type)
  Axis 2: WHERE it changes (spatial selector → condition)  
  Axis 3: HOW it changes (value mapping → action)
  Axis 4: WHY it changes (object/region context → reason)

The Cross product of these axes defines the program space.
CEGIS verifies each candidate program against all training pairs.
"""

from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors
from arc.objects import detect_objects, ArcObject


# === Axis 1: WHAT — Diff Analysis ===

def analyze_diff(inp: Grid, out: Grid, bg: int) -> Dict:
    """Decompose input→output diff into structural description"""
    h, w = grid_shape(inp)
    
    removes = []    # was non-bg, becomes bg
    adds = []       # was bg, becomes non-bg
    recolors = []   # was non-bg, becomes different non-bg
    
    for r in range(h):
        for c in range(w):
            if inp[r][c] == out[r][c]:
                continue
            if inp[r][c] != bg and out[r][c] == bg:
                removes.append((r, c, inp[r][c]))
            elif inp[r][c] == bg and out[r][c] != bg:
                adds.append((r, c, out[r][c]))
            else:
                recolors.append((r, c, inp[r][c], out[r][c]))
    
    # Cluster adds by color
    add_colors = Counter(color for _, _, color in adds)
    
    # Cluster recolors by mapping
    recolor_map = Counter((fr, to) for _, _, fr, to in recolors)
    
    return {
        'removes': removes,
        'adds': adds,
        'recolors': recolors,
        'n_removes': len(removes),
        'n_adds': len(adds),
        'n_recolors': len(recolors),
        'add_colors': dict(add_colors),
        'recolor_map': dict(recolor_map),
        'is_pure_add': len(removes) == 0 and len(recolors) == 0,
        'is_pure_remove': len(adds) == 0 and len(recolors) == 0,
        'is_pure_recolor': len(adds) == 0 and len(removes) == 0,
    }


# === Axis 2: WHERE — Spatial Condition Learner ===

def learn_spatial_condition(positions: List[Tuple[int, int]], 
                           inp: Grid, bg: int) -> Optional[Dict]:
    """Learn what spatial condition selects exactly these positions"""
    h, w = grid_shape(inp)
    pos_set = set(positions)
    
    # Check: adjacent to specific color?
    for color in range(10):
        if color == bg:
            continue
        adj_to_color = set()
        for r in range(h):
            for c in range(w):
                if inp[r][c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == color:
                            adj_to_color.add((r, c))
                            break
        if adj_to_color == pos_set:
            return {'type': 'adjacent_to_color', 'color': color}
    
    # Check: within bounding box of specific color?
    for color in range(10):
        if color == bg:
            continue
        rows = [r for r in range(h) for c in range(w) if inp[r][c] == color]
        cols = [c for r in range(h) for c in range(w) if inp[r][c] == color]
        if not rows:
            continue
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        bbox_bg = set()
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                if inp[r][c] == bg:
                    bbox_bg.add((r, c))
        if bbox_bg == pos_set:
            return {'type': 'within_color_bbox', 'color': color}
    
    # Check: between objects of same color (on same row/col)?
    objs = detect_objects(inp, bg)
    by_color = {}
    for o in objs:
        by_color.setdefault(o.color, []).append(o)
    
    for color, group in by_color.items():
        if len(group) < 2:
            continue
        between_cells = set()
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                ci = group[i].center
                cj = group[j].center
                # Same row
                if abs(ci[0] - cj[0]) < 1:
                    row = int(ci[0])
                    c_min = int(min(ci[1], cj[1]))
                    c_max = int(max(ci[1], cj[1]))
                    for c in range(c_min, c_max+1):
                        if inp[row][c] == bg:
                            between_cells.add((row, c))
                # Same col
                if abs(ci[1] - cj[1]) < 1:
                    col = int(ci[1])
                    r_min = int(min(ci[0], cj[0]))
                    r_max = int(max(ci[0], cj[0]))
                    for r in range(r_min, r_max+1):
                        if inp[r][col] == bg:
                            between_cells.add((r, col))
        if between_cells == pos_set:
            return {'type': 'between_same_color', 'color': color}
    
    # Check: on row/col of specific object
    for obj in objs:
        row_cells = set()
        col_cells = set()
        cr, cc = int(obj.center[0]), int(obj.center[1])
        for c in range(w):
            if inp[cr][c] == bg:
                row_cells.add((cr, c))
        for r in range(h):
            if inp[r][cc] == bg:
                col_cells.add((r, cc))
        
        if row_cells == pos_set:
            return {'type': 'on_obj_row', 'obj_color': obj.color, 'obj_idx': objs.index(obj)}
        if col_cells == pos_set:
            return {'type': 'on_obj_col', 'obj_color': obj.color, 'obj_idx': objs.index(obj)}
        if (row_cells | col_cells) == pos_set:
            return {'type': 'on_obj_cross', 'obj_color': obj.color, 'obj_idx': objs.index(obj)}
    
    return None


def apply_spatial_condition(condition: Dict, inp: Grid, bg: int) -> Set[Tuple[int, int]]:
    """Return positions matching the spatial condition"""
    h, w = grid_shape(inp)
    
    ctype = condition['type']
    
    if ctype == 'adjacent_to_color':
        color = condition['color']
        result = set()
        for r in range(h):
            for c in range(w):
                if inp[r][c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == color:
                            result.add((r, c))
                            break
        return result
    
    elif ctype == 'within_color_bbox':
        color = condition['color']
        rows = [r for r in range(h) for c in range(w) if inp[r][c] == color]
        cols = [c for r in range(h) for c in range(w) if inp[r][c] == color]
        if not rows:
            return set()
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        return {(r, c) for r in range(r1, r2+1) for c in range(c1, c2+1) if inp[r][c] == bg}
    
    elif ctype == 'between_same_color':
        color = condition['color']
        objs = detect_objects(inp, bg)
        group = [o for o in objs if o.color == color]
        result = set()
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                ci = group[i].center
                cj = group[j].center
                if abs(ci[0] - cj[0]) < 1:
                    row = int(ci[0])
                    c_min = int(min(ci[1], cj[1]))
                    c_max = int(max(ci[1], cj[1]))
                    for c in range(c_min, c_max+1):
                        if inp[row][c] == bg:
                            result.add((row, c))
                if abs(ci[1] - cj[1]) < 1:
                    col = int(ci[1])
                    r_min = int(min(ci[0], cj[0]))
                    r_max = int(max(ci[0], cj[0]))
                    for r in range(r_min, r_max+1):
                        if inp[r][col] == bg:
                            result.add((r, col))
        return result
    
    elif ctype == 'on_obj_cross':
        color = condition['obj_color']
        objs = detect_objects(inp, bg)
        group = [o for o in objs if o.color == color]
        if not group:
            return set()
        obj = group[0]  # first matching object
        cr, cc = int(obj.center[0]), int(obj.center[1])
        result = set()
        for c in range(w):
            if inp[cr][c] == bg:
                result.add((cr, c))
        for r in range(h):
            if inp[r][cc] == bg:
                result.add((r, cc))
        return result
    
    return set()


# === Axis 3: HOW — Value Mapping ===

def learn_value_mapping(adds: List[Tuple[int, int, int]], 
                        inp: Grid, bg: int) -> Optional[Dict]:
    """Learn what determines the output color at added positions"""
    if not adds:
        return None
    
    h, w = grid_shape(inp)
    colors = set(c for _, _, c in adds)
    
    # Single color → constant
    if len(colors) == 1:
        return {'type': 'constant', 'color': colors.pop()}
    
    # Color = nearest non-bg neighbor's color?
    consistent = True
    for r, c, out_color in adds:
        best_dist = float('inf')
        best_color = None
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] != bg:
                    d = abs(dr) + abs(dc)
                    if d < best_dist:
                        best_dist = d
                        best_color = inp[nr][nc]
        if best_color != out_color:
            consistent = False
            break
    
    if consistent:
        return {'type': 'nearest_color'}
    
    # Color = specific object's color based on relationship?
    objs = detect_objects(inp, bg)
    for obj in objs:
        obj_color = obj.color
        all_match = all(c == obj_color for _, _, c in adds)
        if all_match:
            # Is this the closest object to all add positions?
            return {'type': 'constant', 'color': obj_color}
    
    return None


def apply_value_mapping(mapping: Dict, positions: Set[Tuple[int, int]],
                        inp: Grid, bg: int) -> Grid:
    """Apply value mapping to selected positions"""
    h, w = grid_shape(inp)
    result = [row[:] for row in inp]
    
    mtype = mapping['type']
    
    if mtype == 'constant':
        for r, c in positions:
            result[r][c] = mapping['color']
    
    elif mtype == 'nearest_color':
        for r, c in positions:
            best_dist = float('inf')
            best_color = bg
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] != bg:
                        d = abs(dr) + abs(dc)
                        if d < best_dist:
                            best_dist = d
                            best_color = inp[nr][nc]
            result[r][c] = best_color
    
    return result


# === Cross Program: Compose axes ===

class CrossProgram:
    """A multi-step program defined by Cross-Structure decomposition"""
    
    def __init__(self, steps: List[Dict]):
        self.steps = steps
        self.name = '+'.join(s.get('name', '?') for s in steps)
    
    def apply(self, inp: Grid) -> Optional[Grid]:
        result = [row[:] for row in inp]
        bg = most_common_color(inp)
        
        for step in self.steps:
            stype = step['step_type']
            
            if stype == 'add_at_condition':
                positions = apply_spatial_condition(step['condition'], result, bg)
                result = apply_value_mapping(step['mapping'], positions, result, bg)
            
            elif stype == 'remove_at_condition':
                positions = apply_spatial_condition(step['condition'], result, bg)
                for r, c in positions:
                    if (r, c) in set((r2, c2) for r2, c2 in positions):
                        result[r][c] = bg
            
            elif stype == 'recolor_map':
                h, w = grid_shape(result)
                color_map = step['color_map']
                for r in range(h):
                    for c in range(w):
                        if result[r][c] in color_map:
                            result[r][c] = color_map[result[r][c]]
        
        return result


def learn_cross_program(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[CrossProgram]:
    """Learn a Cross-Structure multi-step program from training pairs.
    
    Decomposes the transform into axes, finds consistent rules per axis,
    then composes into a CrossProgram.
    """
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    # Analyze diff for all training pairs
    diffs = [analyze_diff(inp, out, bg) for inp, out in train_pairs]
    
    # Check consistency of diff type
    all_pure_add = all(d['is_pure_add'] for d in diffs)
    all_pure_recolor = all(d['is_pure_recolor'] for d in diffs)
    
    steps = []
    
    # === Pure add: learn WHERE + HOW ===
    if all_pure_add:
        # Learn spatial condition from first pair
        add_positions = [(r, c) for r, c, _ in diffs[0]['adds']]
        condition = learn_spatial_condition(add_positions, train_pairs[0][0], bg)
        if condition is None:
            return None
        
        # Verify condition consistency across all pairs
        for i, (inp, out) in enumerate(train_pairs[1:], 1):
            predicted_pos = apply_spatial_condition(condition, inp, bg)
            actual_pos = set((r, c) for r, c, _ in diffs[i]['adds'])
            if predicted_pos != actual_pos:
                return None
        
        # Learn value mapping
        mapping = learn_value_mapping(diffs[0]['adds'], train_pairs[0][0], bg)
        if mapping is None:
            return None
        
        # Verify mapping
        for i, (inp, out) in enumerate(train_pairs):
            positions = apply_spatial_condition(condition, inp, bg)
            result = apply_value_mapping(mapping, positions, inp, bg)
            if not grid_eq(result, out):
                return None
        
        steps.append({
            'step_type': 'add_at_condition',
            'condition': condition,
            'mapping': mapping,
            'name': f"add_{condition['type']}_{mapping['type']}",
        })
    
    # === Pure recolor: learn color mapping ===
    elif all_pure_recolor:
        # Check: consistent global color swap?
        color_map = {}
        for r, c, fr, to in diffs[0]['recolors']:
            if fr in color_map:
                if color_map[fr] != to:
                    color_map = None
                    break
            else:
                color_map[fr] = to
        
        if color_map is None:
            return None
        
        # Verify across pairs
        for i, diff in enumerate(diffs[1:], 1):
            for r, c, fr, to in diff['recolors']:
                if fr in color_map:
                    if color_map[fr] != to:
                        return None
                else:
                    color_map[fr] = to
        
        # Full verification
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            result = [row[:] for row in inp]
            for r in range(h):
                for c in range(w):
                    if result[r][c] in color_map:
                        result[r][c] = color_map[result[r][c]]
            if not grid_eq(result, out):
                return None
        
        steps.append({
            'step_type': 'recolor_map',
            'color_map': color_map,
            'name': f"recolor_{'_'.join(f'{k}to{v}' for k,v in color_map.items())}",
        })
    
    # === Mixed: try decomposing into add + recolor steps ===
    else:
        # Step 1: Handle recolors (if any)
        if all(d['n_recolors'] > 0 for d in diffs):
            color_map = {}
            consistent = True
            for diff in diffs:
                for r, c, fr, to in diff['recolors']:
                    if fr in color_map:
                        if color_map[fr] != to:
                            consistent = False; break
                    else:
                        color_map[fr] = to
                if not consistent: break
            
            if consistent and color_map:
                steps.append({
                    'step_type': 'recolor_map',
                    'color_map': color_map,
                    'name': 'recolor',
                })
        
        # Step 2: Handle adds (if any)
        if all(d['n_adds'] > 0 for d in diffs):
            # Apply recolor first, then analyze remaining diff
            remaining_adds = []
            for i, (inp, out) in enumerate(train_pairs):
                intermediate = [row[:] for row in inp]
                for step in steps:
                    if step['step_type'] == 'recolor_map':
                        h, w = grid_shape(intermediate)
                        for r in range(h):
                            for c in range(w):
                                if intermediate[r][c] in step['color_map']:
                                    intermediate[r][c] = step['color_map'][intermediate[r][c]]
                
                # Find remaining adds
                h, w = grid_shape(out)
                pair_adds = []
                for r in range(h):
                    for c in range(w):
                        if intermediate[r][c] == bg and out[r][c] != bg:
                            pair_adds.append((r, c, out[r][c]))
                remaining_adds.append(pair_adds)
            
            if remaining_adds[0]:
                add_positions = [(r, c) for r, c, _ in remaining_adds[0]]
                
                # Apply recolor to get intermediate input
                intermediate0 = [row[:] for row in train_pairs[0][0]]
                for step in steps:
                    if step['step_type'] == 'recolor_map':
                        h, w = grid_shape(intermediate0)
                        for r in range(h):
                            for c in range(w):
                                if intermediate0[r][c] in step['color_map']:
                                    intermediate0[r][c] = step['color_map'][intermediate0[r][c]]
                
                condition = learn_spatial_condition(add_positions, intermediate0, bg)
                if condition is not None:
                    mapping = learn_value_mapping(remaining_adds[0], intermediate0, bg)
                    if mapping is not None:
                        steps.append({
                            'step_type': 'add_at_condition',
                            'condition': condition,
                            'mapping': mapping,
                            'name': f"add_{condition['type']}",
                        })
    
    if not steps:
        return None
    
    # Final verification
    prog = CrossProgram(steps)
    for inp, out in train_pairs:
        result = prog.apply(inp)
        if result is None or not grid_eq(result, out):
            return None
    
    return prog
