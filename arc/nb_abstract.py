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
from arc.grid import Grid, grid_shape, most_common_color, grid_eq, grid_colors


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


def learn_structural_nb_rule(train_pairs: List[Tuple[Grid, Grid]], 
                             radius: int = 1) -> Optional[Dict]:
    """Learn a neighborhood rule with structural abstraction for fallback.
    
    Two-level mapping:
    1. Exact neighborhood -> output (same as original)
    2. Structural pattern -> output (for unseen exact patterns)
    
    Structural pattern: each cell is classified as:
      -1 = out of bounds
       0 = background
       1 = same color as center
       2 = different non-bg color (any)
    """
    bg = most_common_color(train_pairs[0][0])
    
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    exact_mapping = {}
    struct_mapping = {}
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        for r in range(h):
            for c in range(w):
                center = inp[r][c]
                
                # Exact neighborhood
                raw_nb = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        raw_nb.append(inp[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
                exact_key = tuple(raw_nb)
                
                # Structural neighborhood
                struct_nb = []
                for v in raw_nb:
                    if v == -1:
                        struct_nb.append(-1)
                    elif v == bg:
                        struct_nb.append(0)
                    elif v == center:
                        struct_nb.append(1)
                    else:
                        struct_nb.append(2)
                struct_key = tuple(struct_nb)
                
                out_val = out[r][c]
                
                # Exact mapping
                if exact_key in exact_mapping:
                    if exact_mapping[exact_key] != out_val:
                        return None  # inconsistent
                else:
                    exact_mapping[exact_key] = out_val
                
                # Structural mapping — output as role
                if out_val == bg:
                    out_role = 0  # bg
                elif out_val == center:
                    out_role = 1  # same as center
                else:
                    # Check if output color exists in neighborhood
                    found_in_nb = False
                    for v in raw_nb:
                        if v == out_val and v != center and v != bg:
                            found_in_nb = True
                            break
                    if found_in_nb:
                        out_role = 3  # neighbor's color
                    else:
                        out_role = 100 + out_val  # absolute new color
                
                if struct_key in struct_mapping:
                    if struct_mapping[struct_key] != out_role:
                        # Structural mapping inconsistent — still ok, just can't use it
                        struct_mapping[struct_key] = None
                else:
                    struct_mapping[struct_key] = out_role
    
    if not exact_mapping:
        return None
    if grid_eq(train_pairs[0][0], train_pairs[0][1]):
        return None
    
    # Clean struct_mapping (remove inconsistent entries)
    struct_mapping = {k: v for k, v in struct_mapping.items() if v is not None}
    
    return {
        'exact': exact_mapping,
        'struct': struct_mapping,
        'radius': radius,
        'bg': bg,
    }


def apply_structural_nb_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply structural neighborhood rule with fallback"""
    exact = rule['exact']
    struct = rule['struct']
    radius = rule['radius']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            center = inp[r][c]
            
            raw_nb = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    raw_nb.append(inp[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
            exact_key = tuple(raw_nb)
            
            # Try exact match first
            if exact_key in exact:
                row.append(exact[exact_key])
                continue
            
            # Fallback to structural match
            struct_nb = []
            for v in raw_nb:
                if v == -1:
                    struct_nb.append(-1)
                elif v == bg:
                    struct_nb.append(0)
                elif v == center:
                    struct_nb.append(1)
                else:
                    struct_nb.append(2)
            struct_key = tuple(struct_nb)
            
            if struct_key in struct:
                out_role = struct[struct_key]
                if out_role == 0:
                    row.append(bg)
                elif out_role == 1:
                    row.append(center)
                elif out_role == 3:
                    # Find first non-bg non-center neighbor
                    found = center
                    for v in raw_nb:
                        if v != -1 and v != bg and v != center:
                            found = v
                            break
                    row.append(found)
                elif out_role >= 100:
                    row.append(out_role - 100)
                else:
                    row.append(center)
            else:
                row.append(center)  # ultimate fallback
        result.append(row)
    
    return result


def learn_cross_nb_rule(train_pairs: List[Tuple[Grid, Grid]],
                        radius: int = 1) -> Optional[Dict]:
    """Cross-structure neighborhood rule.
    
    Decomposition:
      Piece 1 (spatial): structural_nb_pattern → role_id
      Piece 2 (color):   role_id → concrete_color (task constant)
    
    The structural pattern uses:
      -1 = OOB, 0 = bg, 1 = same_as_center, 2 = other_nonbg
    
    The output role uses:
      'bg' = background
      'keep' = same as input center
      'nb' = copy from a non-bg neighbor
      'const_X' = task-specific constant color X
    """
    bg = most_common_color(train_pairs[0][0])
    
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # Phase 1: Collect all structural_pattern → output_role mappings
    struct_to_role = {}
    
    # First pass: identify constant colors (appear in output of ALL pairs)
    all_out_colors = None
    all_in_colors = None
    for inp, out in train_pairs:
        oc = grid_colors(out)
        ic = grid_colors(inp)
        if all_out_colors is None:
            all_out_colors = oc
            all_in_colors = ic
        else:
            all_out_colors &= oc
            all_in_colors &= ic
    
    # Constant new colors: in all outputs but never in any input
    const_colors = all_out_colors - all_in_colors - {bg}
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        
        # Build per-pair color ordering: assign IDs to non-bg colors by first appearance
        color_order = {}
        next_other = 2
        for r2 in range(h):
            for c2 in range(w):
                v = inp[r2][c2]
                if v != bg and v not in color_order:
                    color_order[v] = next_other
                    next_other += 1
        
        for r in range(h):
            for c in range(w):
                center = inp[r][c]
                
                # Build structural neighborhood with ordered other-colors
                raw_nb = []
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        raw_nb.append(inp[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
                
                struct_nb = []
                local_other_map = {}
                local_next = 2
                for v in raw_nb:
                    if v == -1:
                        struct_nb.append(-1)
                    elif v == bg:
                        struct_nb.append(0)
                    elif v == center:
                        struct_nb.append(1)
                    else:
                        # Different non-bg colors get different IDs (stable within cell)
                        if v not in local_other_map:
                            local_other_map[v] = local_next
                            local_next += 1
                        struct_nb.append(local_other_map[v])
                struct_key = tuple(struct_nb)
                
                # Determine output role
                out_val = out[r][c]
                if out_val == bg:
                    role = 'bg'
                elif out_val == center:
                    role = 'keep'
                elif out_val in const_colors:
                    role = f'const_{out_val}'
                else:
                    # Check if it matches a local other color → use its local ID
                    if out_val in local_other_map:
                        role = f'other_{local_other_map[out_val]}'
                    else:
                        role = f'abs_{out_val}'
                
                if struct_key in struct_to_role:
                    if struct_to_role[struct_key] != role:
                        return None  # inconsistent
                else:
                    struct_to_role[struct_key] = role
    
    if not struct_to_role:
        return None
    if grid_eq(train_pairs[0][0], train_pairs[0][1]):
        return None
    
    # Check that it actually has non-trivial roles
    roles_used = set(struct_to_role.values())
    if roles_used == {'keep'}:
        return None
    
    return {
        'struct_to_role': struct_to_role,
        'const_colors': const_colors,
        'radius': radius,
        'bg': bg,
    }


def apply_cross_nb_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply cross-structure neighborhood rule"""
    struct_to_role = rule['struct_to_role']
    radius = rule['radius']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            center = inp[r][c]
            
            raw_nb = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    raw_nb.append(inp[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1)
            
            # Build structural key with local other-color IDs
            local_other_map = {}
            local_id_to_color = {}
            local_next = 2
            struct_nb = []
            for v in raw_nb:
                if v == -1:
                    struct_nb.append(-1)
                elif v == bg:
                    struct_nb.append(0)
                elif v == center:
                    struct_nb.append(1)
                else:
                    if v not in local_other_map:
                        local_other_map[v] = local_next
                        local_id_to_color[local_next] = v
                        local_next += 1
                    struct_nb.append(local_other_map[v])
            struct_key = tuple(struct_nb)
            
            if struct_key in struct_to_role:
                role = struct_to_role[struct_key]
                if role == 'bg':
                    row.append(bg)
                elif role == 'keep':
                    row.append(center)
                elif role.startswith('other_'):
                    other_id = int(role.split('_')[1])
                    row.append(local_id_to_color.get(other_id, center))
                elif role.startswith('const_'):
                    row.append(int(role.split('_')[1]))
                elif role.startswith('abs_'):
                    row.append(int(role.split('_')[1]))
                else:
                    row.append(center)
            else:
                row.append(center)
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


# === Rotation-Invariant NB Rule ===

def _rotate_3x3(pat: tuple) -> tuple:
    """Rotate 3x3 pattern 90° clockwise."""
    m = [list(pat[0:3]), list(pat[3:6]), list(pat[6:9])]
    r = [[m[2][0], m[1][0], m[0][0]],
         [m[2][1], m[1][1], m[0][1]],
         [m[2][2], m[1][2], m[0][2]]]
    return tuple(r[0] + r[1] + r[2])

def _flip_h_3x3(pat: tuple) -> tuple:
    """Flip 3x3 pattern horizontally."""
    m = [list(pat[0:3]), list(pat[3:6]), list(pat[6:9])]
    return tuple(m[0][::-1] + m[1][::-1] + m[2][::-1])

def _canonical_3x3(pat: tuple) -> tuple:
    """Get canonical form (min of all 8 rotations/flips)."""
    variants = []
    p = pat
    for _ in range(4):
        variants.append(p)
        variants.append(_flip_h_3x3(p))
        p = _rotate_3x3(p)
    return min(variants)

def _get_struct_nb(inp, r, c, bg, radius=1):
    """Get structural 3x3 neighborhood pattern."""
    h = len(inp)
    w = len(inp[0]) if h > 0 else 0
    center = inp[r][c]
    nb = []
    local_other_map = {}
    local_next = 2
    raw_nb = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                v = inp[nr][nc]
                raw_nb.append(v)
                if v == bg:
                    nb.append(0)
                elif v == center:
                    nb.append(1)
                else:
                    if v not in local_other_map:
                        local_other_map[v] = local_next
                        local_next += 1
                    nb.append(local_other_map[v])
            else:
                raw_nb.append(-1)
                nb.append(-1)
    return tuple(nb), raw_nb, local_other_map

def _get_output_role(out_val, center, bg, local_other_map):
    """Get abstract output role."""
    if out_val == bg:
        return 'bg'
    elif out_val == center:
        return 'keep'
    elif out_val in local_other_map:
        return f'other_{local_other_map[out_val]}'
    else:
        return f'abs_{out_val}'

def _resolve_role(role, center, bg, local_id_to_color):
    """Resolve abstract role back to concrete color."""
    if role == 'bg':
        return bg
    elif role == 'keep':
        return center
    elif role.startswith('other_'):
        oid = int(role.split('_')[1])
        return local_id_to_color.get(oid, center)
    elif role.startswith('abs_'):
        return int(role.split('_')[1])
    return center


def learn_rotation_invariant_nb_rule(train_pairs: List[Tuple[Grid, Grid]],
                                      radius: int = 1) -> Optional[Dict]:
    """Learn NB rule with rotation/flip invariance.
    
    Patterns are canonicalized under the 8 symmetries of the square (D4 group).
    This dramatically improves generalization to unseen test patterns.
    """
    bg = most_common_color(train_pairs[0][0])
    
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    if radius != 1:
        return None  # Only support radius=1 for now (3x3)
    
    canonical_map = {}  # canonical_pattern -> role
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        for r in range(h):
            for c in range(w):
                struct_nb, raw_nb, local_other_map = _get_struct_nb(inp, r, c, bg, radius)
                cpat = _canonical_3x3(struct_nb)
                role = _get_output_role(out[r][c], inp[r][c], bg, local_other_map)
                
                if cpat in canonical_map:
                    if canonical_map[cpat] != role:
                        return None  # inconsistent
                else:
                    canonical_map[cpat] = role
    
    if not canonical_map:
        return None
    if grid_eq(train_pairs[0][0], train_pairs[0][1]):
        return None
    
    roles_used = set(canonical_map.values())
    if roles_used == {'keep'}:
        return None
    
    return {'canonical_map': canonical_map, 'radius': radius, 'bg': bg}


def apply_rotation_invariant_nb_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply rotation-invariant NB rule."""
    canonical_map = rule['canonical_map']
    radius = rule['radius']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            center = inp[r][c]
            struct_nb, raw_nb, local_other_map = _get_struct_nb(inp, r, c, bg, radius)
            
            # Build reverse map
            local_id_to_color = {v: k for k, v in local_other_map.items()}
            
            cpat = _canonical_3x3(struct_nb)
            
            if cpat in canonical_map:
                role = canonical_map[cpat]
                row.append(_resolve_role(role, center, bg, local_id_to_color))
            else:
                row.append(center)  # fallback
        result.append(row)
    
    return result


def learn_rotsym_count_nb_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Ultra-coarse NB rule: only uses counts of bg/self/other in 4-connected + 8-connected.
    
    Key = (is_bg, n4_bg, n4_self, n4_other, n8_bg, n8_self, n8_other)
    
    Even coarser than rotation-invariant, but with maximum generalization.
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
                
                n4 = {'bg': 0, 'self': 0, 'other': 0, 'edge': 0}
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = inp[nr][nc]
                        if v == bg: n4['bg'] += 1
                        elif v == center: n4['self'] += 1
                        else: n4['other'] += 1
                    else:
                        n4['edge'] += 1
                
                n8 = {'bg': 0, 'self': 0, 'other': 0, 'edge': 0}
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            v = inp[nr][nc]
                            if v == bg: n8['bg'] += 1
                            elif v == center: n8['self'] += 1
                            else: n8['other'] += 1
                        else:
                            n8['edge'] += 1
                
                key = (center == bg,
                       n4['bg'], n4['self'], n4['other'],
                       n8['bg'], n8['self'], n8['other'])
                
                ov = out[r][c]
                if ov == bg: role = 'bg'
                elif ov == center: role = 'keep'
                else: role = f'abs_{ov}'
                
                if key in mapping:
                    if mapping[key] != role:
                        return None
                else:
                    mapping[key] = role
    
    if not mapping:
        return None
    if grid_eq(train_pairs[0][0], train_pairs[0][1]):
        return None
    
    roles = set(mapping.values())
    if roles == {'keep'}:
        return None
    
    return {'mapping': mapping, 'bg': bg}


def apply_rotsym_count_nb_rule(inp: Grid, rule: Dict) -> Grid:
    """Apply ultra-coarse rotation-symmetric count NB rule."""
    mapping = rule['mapping']
    bg = rule['bg']
    h, w = grid_shape(inp)
    
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            center = inp[r][c]
            
            n4 = {'bg': 0, 'self': 0, 'other': 0, 'edge': 0}
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = inp[nr][nc]
                    if v == bg: n4['bg'] += 1
                    elif v == center: n4['self'] += 1
                    else: n4['other'] += 1
                else:
                    n4['edge'] += 1
            
            n8 = {'bg': 0, 'self': 0, 'other': 0, 'edge': 0}
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = inp[nr][nc]
                        if v == bg: n8['bg'] += 1
                        elif v == center: n8['self'] += 1
                        else: n8['other'] += 1
                    else:
                        n8['edge'] += 1
            
            key = (center == bg,
                   n4['bg'], n4['self'], n4['other'],
                   n8['bg'], n8['self'], n8['other'])
            
            if key in mapping:
                role = mapping[key]
                if role == 'bg': row.append(bg)
                elif role == 'keep': row.append(center)
                elif role.startswith('abs_'): row.append(int(role.split('_')[1]))
                else: row.append(center)
            else:
                row.append(center)
        result.append(row)
    
    return result
