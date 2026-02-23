"""
arc/conditional.py — Conditional/parametric DSL for ARC-AGI-2

Addresses Wall 3: Conditional branching (240 unsolved tasks, 26%)

Rules that depend on context: neighbor counts, region properties,
position within grid, etc.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from collections import Counter
from arc.grid import Grid, grid_shape, most_common_color, grid_eq
from arc.objects import detect_objects, ArcObject


def learn_conditional_color_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: if cell has N neighbors of color X, change to color Y.
    
    Maps: (cell_is_bg, n_nonbg_neighbors) -> output_color_role
    """
    bg = most_common_color(train_pairs[0][0])
    mapping = {}
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        if grid_shape(out) != (h, w):
            return None
        
        for r in range(h):
            for c in range(w):
                # Count 8-neighbors (including diagonals)
                n_nonbg = 0
                n_same = 0
                neighbor_colors = set()
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if inp[nr][nc] != bg:
                                n_nonbg += 1
                                neighbor_colors.add(inp[nr][nc])
                            if inp[nr][nc] == inp[r][c]:
                                n_same += 1
                
                cell_is_bg = inp[r][c] == bg
                n_distinct = len(neighbor_colors)
                
                key = (cell_is_bg, n_nonbg, n_same, n_distinct)
                
                # Output role
                if out[r][c] == bg:
                    out_role = 0
                elif out[r][c] == inp[r][c]:
                    out_role = 1
                else:
                    out_role = 100 + out[r][c]
                
                if key in mapping:
                    if mapping[key] != out_role:
                        return None
                else:
                    mapping[key] = out_role
    
    if not mapping or grid_eq(train_pairs[0][0], train_pairs[0][1]):
        return None
    
    return {'mapping': mapping, 'bg': bg}


def apply_conditional_color_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply conditional color rule"""
    mapping = rule['mapping']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            n_nonbg = 0
            n_same = 0
            neighbor_colors = set()
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if inp[nr][nc] != bg:
                            n_nonbg += 1
                            neighbor_colors.add(inp[nr][nc])
                        if inp[nr][nc] == inp[r][c]:
                            n_same += 1
            
            cell_is_bg = inp[r][c] == bg
            n_distinct = len(neighbor_colors)
            key = (cell_is_bg, n_nonbg, n_same, n_distinct)
            
            if key in mapping:
                out_role = mapping[key]
                if out_role == 0:
                    row.append(bg)
                elif out_role == 1:
                    row.append(inp[r][c])
                elif out_role >= 100:
                    row.append(out_role - 100)
                else:
                    row.append(inp[r][c])
            else:
                row.append(inp[r][c])
        result.append(row)
    
    return result


def learn_region_property_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: recolor objects based on their properties.
    
    Properties: size, height, width, is_square, is_line, n_holes
    """
    bg = most_common_color(train_pairs[0][0])
    
    # Check all pairs are same size
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # size -> output_color mapping
    size_map = {}
    shape_map = {}
    
    for inp, out in train_pairs:
        objs_in = detect_objects(inp, bg)
        
        for obj in objs_in:
            # Check what color this object became in output
            out_colors = Counter()
            for r, c in obj.cells:
                out_colors[out[r][c]] += 1
            
            if not out_colors:
                continue
            
            dominant = out_colors.most_common(1)[0][0]
            
            # Size-based mapping
            sz = obj.size
            if sz in size_map:
                if size_map[sz] != dominant:
                    size_map = None
                    break
            else:
                size_map[sz] = dominant
        
        if size_map is None:
            break
    
    # Try shape-based mapping
    shape_ok = True
    shape_map = {}
    for inp, out in train_pairs:
        objs_in = detect_objects(inp, bg)
        for obj in objs_in:
            out_colors = Counter()
            for r, c in obj.cells:
                out_colors[out[r][c]] += 1
            if not out_colors:
                continue
            dominant = out_colors.most_common(1)[0][0]
            sh = obj.shape
            if sh in shape_map:
                if shape_map[sh] != dominant:
                    shape_ok = False
                    break
            else:
                shape_map[sh] = dominant
        if not shape_ok:
            break
    
    result = {}
    if size_map and any(v != k for k, v in size_map.items()):
        result['size_map'] = size_map
    if shape_ok and shape_map:
        result['shape_map'] = shape_map
    
    if not result:
        return None
    
    result['bg'] = bg
    return result


def apply_region_property_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply region property-based recoloring"""
    bg = rule['bg']
    size_map = rule.get('size_map', {})
    shape_map = rule.get('shape_map', {})
    
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    
    for obj in objs:
        new_color = None
        
        # Try shape map first (more specific)
        if shape_map and obj.shape in shape_map:
            new_color = shape_map[obj.shape]
        elif size_map and obj.size in size_map:
            new_color = size_map[obj.size]
        
        if new_color is not None:
            for r, c in obj.cells:
                result[r][c] = new_color
    
    return result


def learn_object_sort_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: output is objects sorted/arranged by some property"""
    bg = most_common_color(train_pairs[0][0])
    
    # Only handle same-size for now
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # Try: sort objects by size and recolor by rank
    for inp, out in train_pairs:
        objs_in = detect_objects(inp, bg)
        objs_out = detect_objects(out, bg)
        
        if len(objs_in) != len(objs_out):
            return None
        if len(objs_in) < 2:
            return None
    
    # Check if output recolors by size rank
    color_by_rank = {}
    consistent = True
    
    for inp, out in train_pairs:
        objs_in = detect_objects(inp, bg)
        sorted_by_size = sorted(objs_in, key=lambda o: o.size)
        
        for rank, obj in enumerate(sorted_by_size):
            out_colors = set(out[r][c] for r, c in obj.cells if out[r][c] != bg)
            if len(out_colors) == 1:
                oc = out_colors.pop()
                if rank in color_by_rank:
                    if color_by_rank[rank] != oc:
                        consistent = False
                        break
                else:
                    color_by_rank[rank] = oc
        if not consistent:
            break
    
    if consistent and color_by_rank:
        return {'type': 'sort_by_size', 'color_by_rank': color_by_rank, 'bg': bg}
    
    return None
