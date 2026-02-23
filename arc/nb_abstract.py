"""
arc/nb_abstract.py — Abstract Neighborhood Rules for ARC-AGI-2

Addresses Wall 2: NB generalization (222 unsolved tasks, 24%)

Instead of memorizing exact color patterns in neighborhoods,
abstract them to relative roles: bg, self, other, any_nonbg.
This allows generalization to unseen color combinations.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter
from arc.grid import Grid, grid_shape, most_common_color, grid_eq


def _abstract_nb(nb: Tuple[int, ...], center_color: int, bg: int) -> Tuple:
    """Convert absolute colors to relative roles.
    
    Roles:
      -1 = out of bounds
       0 = background
       1 = same as center
       2 = other non-bg color (first encountered)
       3 = other non-bg color (second encountered)
       ...
    """
    other_map = {}
    next_id = 2
    result = []
    for v in nb:
        if v == -1:
            result.append(-1)
        elif v == bg:
            result.append(0)
        elif v == center_color:
            result.append(1)
        else:
            if v not in other_map:
                other_map[v] = next_id
                next_id += 1
            result.append(other_map[v])
    return tuple(result)


def _abstract_output(out_color: int, center_color: int, bg: int, 
                     nb_colors: Dict[int, int]) -> int:
    """Convert output color to abstract role.
    
    Returns:
      -1 = out of bounds (shouldn't happen)
       0 = background
       1 = same as center (input)
       2+ = maps to specific role
       100+ = absolute color (when no relative mapping found)
    """
    if out_color == bg:
        return 0
    elif out_color == center_color:
        return 1
    elif out_color in nb_colors:
        return nb_colors[out_color]
    else:
        return 100 + out_color  # absolute color fallback


def _resolve_output(abstract_out: int, center_color: int, bg: int,
                    role_to_color: Dict[int, int]) -> int:
    """Convert abstract output back to concrete color"""
    if abstract_out == 0:
        return bg
    elif abstract_out == 1:
        return center_color
    elif abstract_out >= 100:
        return abstract_out - 100  # absolute color
    elif abstract_out in role_to_color:
        return role_to_color[abstract_out]
    else:
        return center_color  # fallback


def learn_abstract_nb_rule(train_pairs: List[Tuple[Grid, Grid]], 
                           radius: int = 1) -> Optional[Dict]:
    """Learn an abstract neighborhood rule from training pairs.
    
    Returns dict with 'mapping' and 'radius' if consistent, else None.
    """
    bg = most_common_color(train_pairs[0][0])
    mapping = {}
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        oh, ow = grid_shape(out)
        if (h, w) != (oh, ow):
            return None
        
        for r in range(h):
            for c in range(w):
                center = inp[r][c]
                
                # Collect raw neighborhood
                raw_nb = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        raw_nb.append(inp[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
                
                # Abstract the neighborhood
                abs_nb = _abstract_nb(tuple(raw_nb), center, bg)
                
                # Build role-to-color mapping for this cell
                other_map = {}
                next_id = 2
                for v in raw_nb:
                    if v != -1 and v != bg and v != center and v not in other_map:
                        other_map[v] = next_id
                        next_id += 1
                
                # Abstract the output
                abs_out = _abstract_output(out[r][c], center, bg, other_map)
                
                # Check consistency
                if abs_nb in mapping:
                    if mapping[abs_nb] != abs_out:
                        return None
                else:
                    mapping[abs_nb] = abs_out
    
    # Verify it actually changes something
    if not mapping:
        return None
    
    inp0, out0 = train_pairs[0]
    if grid_eq(inp0, out0):
        return None
    
    return {'mapping': mapping, 'radius': radius, 'bg': bg}


def apply_abstract_nb_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply an abstract neighborhood rule to a grid"""
    mapping = rule['mapping']
    radius = rule['radius']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            center = inp[r][c]
            
            # Collect raw neighborhood
            raw_nb = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    raw_nb.append(inp[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
            
            # Abstract
            abs_nb = _abstract_nb(tuple(raw_nb), center, bg)
            
            if abs_nb in mapping:
                abs_out = mapping[abs_nb]
                # Build role-to-color mapping for resolution
                other_map = {}
                role_to_color = {}
                next_id = 2
                for v in raw_nb:
                    if v != -1 and v != bg and v != center and v not in other_map:
                        other_map[v] = next_id
                        role_to_color[next_id] = v
                        next_id += 1
                
                row.append(_resolve_output(abs_out, center, bg, role_to_color))
            else:
                row.append(center)  # fallback: keep original
        result.append(row)
    
    return result


def learn_count_based_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn a rule based on neighbor counts rather than exact patterns.
    
    Rule: output[r][c] = f(input[r][c], count_of_each_color_in_neighborhood)
    
    This handles cases where the exact pattern doesn't matter,
    only the count of neighbors of each type.
    """
    bg = most_common_color(train_pairs[0][0])
    mapping = {}
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        if grid_shape(out) != (h, w):
            return None
        
        for r in range(h):
            for c in range(w):
                center = inp[r][c]
                
                # Count 4-neighbors
                counts = Counter()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = inp[nr][nc]
                        if v == bg:
                            counts['bg'] += 1
                        elif v == center:
                            counts['self'] += 1
                        else:
                            counts['other'] += 1
                    else:
                        counts['edge'] += 1
                
                key = (center == bg, counts['bg'], counts['self'], counts['other'], counts['edge'])
                
                # Abstract output
                if out[r][c] == bg:
                    out_abs = 'bg'
                elif out[r][c] == center:
                    out_abs = 'self'
                else:
                    out_abs = out[r][c]  # absolute for new colors
                
                if key in mapping:
                    if mapping[key] != out_abs:
                        return None
                else:
                    mapping[key] = out_abs
    
    if not mapping:
        return None
    
    if grid_eq(train_pairs[0][0], train_pairs[0][1]):
        return None
    
    return {'mapping': mapping, 'bg': bg}


def apply_count_based_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply a count-based neighborhood rule"""
    mapping = rule['mapping']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            center = inp[r][c]
            
            counts = Counter()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = inp[nr][nc]
                    if v == bg:
                        counts['bg'] += 1
                    elif v == center:
                        counts['self'] += 1
                    else:
                        counts['other'] += 1
                else:
                    counts['edge'] += 1
            
            key = (center == bg, counts['bg'], counts['self'], counts['other'], counts['edge'])
            
            if key in mapping:
                out_abs = mapping[key]
                if out_abs == 'bg':
                    row.append(bg)
                elif out_abs == 'self':
                    row.append(center)
                else:
                    row.append(out_abs)
            else:
                row.append(center)
        result.append(row)
    
    return result
