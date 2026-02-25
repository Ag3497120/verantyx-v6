"""
Meta-Cross Engine: Hierarchical routing over concept-role-operation spaces.

Architecture:
  1. Concept Detection: What structural concepts exist in the input?
     (pointer, mask, panel, frame, symmetry, scatter, ...)
  2. Role Assignment: Which objects play which roles?
     (marker, template, target, border, separator, ...)
  3. Operation Tree: What program tree solves this?
     (extract→nearest, overlay→at_marker, split→per_panel→merge, ...)

The Meta-Cross understands each sub-space's capabilities and routes
tasks to the most promising concept-role-operation paths.
"""

from typing import List, Tuple, Optional, Dict, Set, Any
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.cross_engine import CrossPiece

try:
    import numpy as np
    from scipy import ndimage
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================
# Concept Detection Layer
# ============================================================

class ConceptSignature:
    """Describes what structural concepts are present in a task."""
    def __init__(self):
        self.has_pointer = False       # marker cell pointing somewhere
        self.has_mask = False          # one object usable as template/mask
        self.has_panels = False        # grid divided into panels by separators
        self.has_frame = False         # nested rectangles / frame structure
        self.has_scatter = False       # many small same-color disconnected cells
        self.has_symmetry_break = False # almost symmetric but not quite
        self.has_color_groups = False  # same color in disconnected clusters
        self.has_size_hierarchy = False # objects of different sizes
        self.has_unique_marker = False # exactly one cell of a unique color
        self.has_separator = False     # line of single color dividing grid
        self.n_concepts = 0
        self.scores = {}  # concept_name -> confidence score [0,1]


def detect_concepts(inp, bg) -> ConceptSignature:
    """Detect structural concepts in a grid."""
    sig = ConceptSignature()
    if not HAS_NUMPY:
        return sig
    
    arr = np.array(inp)
    h, w = arr.shape
    non_bg_colors = sorted(set(int(v) for v in arr.flatten() if v != bg))
    
    if not non_bg_colors:
        return sig
    
    # Color frequency
    color_counts = Counter(int(v) for v in arr.flatten() if v != bg)
    
    # --- Unique marker: exactly one cell of a rare color ---
    for color, count in color_counts.items():
        if count == 1:
            sig.has_unique_marker = True
            sig.has_pointer = True
            sig.scores['pointer'] = 0.8
            break
    
    # --- Color groups: same color in 3+ disconnected clusters ---
    for color in non_bg_colors:
        cmask = arr == color
        _, nc = ndimage.label(cmask)
        if nc >= 3:
            sig.has_color_groups = True
            sig.has_scatter = True
            sig.scores['color_groups'] = min(1.0, nc / 5.0)
            break
    
    # --- Separator: full row or column of single non-bg color ---
    for r in range(h):
        row_vals = set(int(arr[r, c]) for c in range(w))
        row_vals.discard(bg)
        if len(row_vals) == 1 and all(arr[r, c] != bg for c in range(w)):
            sig.has_separator = True
            sig.has_panels = True
            sig.scores['panels'] = 0.9
            break
    if not sig.has_separator:
        for c in range(w):
            col_vals = set(int(arr[r, c]) for r in range(h))
            col_vals.discard(bg)
            if len(col_vals) == 1 and all(arr[r, c] != bg for r in range(h)):
                sig.has_separator = True
                sig.has_panels = True
                sig.scores['panels'] = 0.9
                break
    
    # --- Frame: nested rectangles ---
    labeled, n_objs = ndimage.label(arr != bg)
    if n_objs >= 1:
        # Check for rectangular bounding boxes that contain other objects
        obj_bboxes = []
        for oid in range(1, n_objs + 1):
            cells = list(zip(*np.where(labeled == oid)))
            r1 = min(r for r, c in cells); r2 = max(r for r, c in cells)
            c1 = min(c for r, c in cells); c2 = max(c for r, c in cells)
            obj_bboxes.append((r1, c1, r2, c2, oid))
        
        for i, (r1, c1, r2, c2, _) in enumerate(obj_bboxes):
            for j, (or1, oc1, or2, oc2, _) in enumerate(obj_bboxes):
                if i != j and r1 < or1 and c1 < oc1 and r2 > or2 and c2 > oc2:
                    sig.has_frame = True
                    sig.scores['frame'] = 0.7
                    break
    
    # --- Mask: two distinct shapes where one could be a template ---
    if len(non_bg_colors) >= 2:
        color_shapes = {}
        for color in non_bg_colors:
            cells = list(zip(*np.where(arr == color)))
            if cells:
                r1 = min(r for r, c in cells); r2 = max(r for r, c in cells)
                c1 = min(c for r, c in cells); c2 = max(c for r, c in cells)
                sh = r2 - r1 + 1; sw = c2 - c1 + 1
                color_shapes[color] = (sh, sw, len(cells))
        
        sizes = sorted(color_shapes.values(), key=lambda x: x[2])
        if len(sizes) >= 2:
            smallest = sizes[0]
            largest = sizes[-1]
            if smallest[2] < largest[2] * 0.3:  # small shape is much smaller
                sig.has_mask = True
                sig.scores['mask'] = 0.6
    
    # --- Size hierarchy ---
    if n_objs >= 3:
        obj_sizes = []
        for oid in range(1, n_objs + 1):
            obj_sizes.append(int(np.sum(labeled == oid)))
        if max(obj_sizes) > 3 * min(obj_sizes):
            sig.has_size_hierarchy = True
            sig.scores['size_hierarchy'] = 0.5
    
    # --- Symmetry break ---
    # Check if grid is almost symmetric (>90% match)
    for flip_fn in [np.fliplr, np.flipud]:
        flipped = flip_fn(arr)
        match_pct = np.mean(arr == flipped)
        if 0.8 <= match_pct < 1.0:
            sig.has_symmetry_break = True
            sig.scores['symmetry_break'] = match_pct
            break
    
    sig.n_concepts = sum(1 for v in vars(sig).values() 
                         if isinstance(v, bool) and v)
    return sig


# ============================================================
# Role Assignment Layer
# ============================================================

class RoleAssignment:
    """Assigns roles to objects/colors in the input."""
    def __init__(self):
        self.marker_color = None      # unique/special color that acts as pointer
        self.marker_pos = None        # (r, c) of marker
        self.template_color = None    # color whose shape is used as mask
        self.target_color = None      # color that gets transformed
        self.separator_color = None   # color of grid divider
        self.frame_color = None       # color of outer frame
        self.fill_color = None        # color used for filling


def assign_roles(inp, bg, sig: ConceptSignature) -> RoleAssignment:
    """Assign roles to colors/objects based on detected concepts."""
    roles = RoleAssignment()
    arr = np.array(inp)
    h, w = arr.shape
    color_counts = Counter(int(v) for v in arr.flatten() if v != bg)
    
    if not color_counts:
        return roles
    
    # Marker: least common non-bg color with count=1
    if sig.has_unique_marker:
        for color, count in color_counts.most_common()[::-1]:
            if count == 1:
                roles.marker_color = color
                cells = list(zip(*np.where(arr == color)))
                if cells:
                    roles.marker_pos = cells[0]
                break
    
    # Template: smallest object (by cell count)
    if sig.has_mask:
        min_color = min(color_counts, key=color_counts.get)
        max_color = max(color_counts, key=color_counts.get)
        roles.template_color = min_color
        roles.target_color = max_color
    
    # Separator: a color that forms a complete row or column
    if sig.has_separator:
        for color in color_counts:
            for r in range(h):
                if all(arr[r, c] == color for c in range(w)):
                    roles.separator_color = color
                    break
            if roles.separator_color:
                break
            for c in range(w):
                if all(arr[r, c] == color for r in range(h)):
                    roles.separator_color = color
                    break
            if roles.separator_color:
                break
    
    return roles


# ============================================================
# Operation Tree Layer
# ============================================================

def generate_operation_trees(
    train_pairs: List[Tuple[Grid, Grid]],
    sig: ConceptSignature,
    roles: RoleAssignment
) -> List[CrossPiece]:
    """Generate operation trees based on concept + role analysis."""
    pieces = []
    
    if not HAS_NUMPY:
        return pieces
    
    bg = most_common_color(train_pairs[0][0])
    
    # Route to concept-specific generators
    if sig.has_pointer and roles.marker_color is not None:
        pieces.extend(_gen_pointer_ops(train_pairs, bg, roles))
    
    if sig.has_panels and roles.separator_color is not None:
        pieces.extend(_gen_panel_ops(train_pairs, bg, roles))
    
    if sig.has_frame:
        pieces.extend(_gen_frame_ops(train_pairs, bg, roles))
    
    if sig.has_mask and roles.template_color is not None:
        pieces.extend(_gen_mask_ops(train_pairs, bg, roles))
    
    if sig.has_symmetry_break:
        pieces.extend(_gen_symmetry_repair_ops(train_pairs, bg))
    
    if sig.has_color_groups:
        pieces.extend(_gen_color_group_ops(train_pairs, bg))
    
    return pieces


# ============================================================
# Concept-Specific Operation Generators
# ============================================================

def _gen_pointer_ops(train_pairs, bg, roles):
    """Operations guided by a marker/pointer cell."""
    pieces = []
    mc = roles.marker_color
    
    # Op 1: Extract object nearest to marker
    rule = _learn_extract_nearest_to_marker(train_pairs, bg, mc)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'mc:extract_nearest_marker_c{mc}',
            lambda inp, r=_r: _apply_extract_nearest_to_marker(inp, r)
        ))
    
    # Op 2: Shoot ray from marker in direction away from nearest object
    rule = _learn_marker_ray(train_pairs, bg, mc)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'mc:marker_ray_c{mc}',
            lambda inp, r=_r: _apply_marker_ray(inp, r)
        ))
    
    # Op 3: Recolor marker's containing object 
    rule = _learn_marker_recolor(train_pairs, bg, mc)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'mc:marker_recolor_c{mc}',
            lambda inp, r=_r: _apply_marker_recolor(inp, r)
        ))
    
    # Op 4: Fill from marker position (flood fill with learned color)
    rule = _learn_marker_fill(train_pairs, bg, mc)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'mc:marker_fill_c{mc}',
            lambda inp, r=_r: _apply_marker_fill(inp, r)
        ))
    
    return pieces


def _learn_extract_nearest_to_marker(train_pairs, bg, marker_color):
    """Learn: find marker → find nearest object → extract its bbox."""
    from arc.objects import detect_objects
    
    for inp, out in train_pairs:
        arr = np.array(inp)
        oh, ow = grid_shape(out)
        ih, iw = grid_shape(inp)
        if oh >= ih and ow >= iw:
            return None  # not an extract task
    
    # Learn from all train pairs
    for inp, out in train_pairs:
        arr = np.array(inp)
        h, w = arr.shape
        out_arr = np.array(out)
        oh, ow = out_arr.shape
        
        # Find marker
        markers = [(r, c) for r in range(h) for c in range(w) 
                    if int(arr[r, c]) == marker_color]
        if len(markers) != 1:
            return None
        mr, mc = markers[0]
        
        # Find all objects (exclude marker)
        objs = detect_objects(inp, bg)
        if not objs:
            return None
        
        # Find nearest object to marker
        best_dist = float('inf')
        best_obj = None
        for obj in objs:
            if obj.size == 1 and obj.color == marker_color:
                continue  # skip the marker itself
            # Distance from marker to object's nearest cell
            min_d = min(abs(r - mr) + abs(c - mc) for r, c in obj.cells)
            if min_d < best_dist:
                best_dist = min_d
                best_obj = obj
        
        if best_obj is None:
            return None
        
        # Check: does extracting best_obj's bbox match output?
        r1, c1, r2, c2 = best_obj.bbox
        extracted = [row[c1:c2+1] for row in inp[r1:r2+1]]
        
        if not grid_eq(extracted, out):
            # Try: extract multicolor bbox
            sub = arr[r1:r2+1, c1:c2+1]
            if sub.shape == out_arr.shape and np.array_equal(sub, out_arr):
                pass  # ok
            else:
                return None
    
    return {'marker_color': marker_color, 'bg': bg}


def _apply_extract_nearest_to_marker(inp, params):
    from arc.objects import detect_objects
    mc = params['marker_color']
    bg = params['bg']
    arr = np.array(inp)
    h, w = arr.shape
    
    markers = [(r, c) for r in range(h) for c in range(w) if int(arr[r, c]) == mc]
    if len(markers) != 1:
        return None
    mr, mcc = markers[0]
    
    objs = detect_objects(inp, bg)
    best_dist = float('inf')
    best_obj = None
    for obj in objs:
        if obj.size == 1 and obj.color == mc:
            continue
        min_d = min(abs(r - mr) + abs(c - mcc) for r, c in obj.cells)
        if min_d < best_dist:
            best_dist = min_d
            best_obj = obj
    
    if best_obj is None:
        return None
    
    r1, c1, r2, c2 = best_obj.bbox
    return [row[c1:c2+1] for row in inp[r1:r2+1]]


def _learn_marker_ray(train_pairs, bg, marker_color):
    """Learn: shoot ray from marker in a learned direction with a learned color."""
    for inp, out in train_pairs:
        arr_i = np.array(inp); arr_o = np.array(out)
        if arr_i.shape != arr_o.shape:
            return None
    
    # Find consistent direction and ray color
    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ray_color = None
        ok = True
        
        for inp, out in train_pairs:
            arr_i = np.array(inp); arr_o = np.array(out)
            h, w = arr_i.shape
            
            markers = [(r, c) for r in range(h) for c in range(w) 
                        if int(arr_i[r, c]) == marker_color]
            if len(markers) != 1:
                ok = False; break
            mr, mc = markers[0]
            
            # Check what the output adds along this direction
            dr, dc = direction
            r, c = mr + dr, mc + dc
            local_ray_color = None
            while 0 <= r < h and 0 <= c < w:
                if arr_i[r, c] != bg:
                    break  # hit obstacle
                if arr_o[r, c] != bg and arr_o[r, c] != arr_i[r, c]:
                    if local_ray_color is None:
                        local_ray_color = int(arr_o[r, c])
                    elif local_ray_color != int(arr_o[r, c]):
                        ok = False; break
                r += dr; c += dc
            
            if not ok:
                break
            if local_ray_color is None:
                ok = False; break
            if ray_color is None:
                ray_color = local_ray_color
            elif ray_color != local_ray_color:
                ok = False; break
            
            # Verify: output = input + ray
            result = arr_i.copy()
            r, c = mr + dr, mc + dc
            while 0 <= r < h and 0 <= c < w:
                if arr_i[r, c] != bg:
                    break
                result[r, c] = ray_color
                r += dr; c += dc
            
            if not np.array_equal(result, arr_o):
                ok = False; break
        
        if ok and ray_color is not None:
            return {'marker_color': marker_color, 'direction': direction,
                    'ray_color': ray_color, 'bg': bg}
    
    # Try: direction = away from nearest object
    ok = True
    ray_color = None
    for inp, out in train_pairs:
        arr_i = np.array(inp); arr_o = np.array(out)
        h, w = arr_i.shape
        
        markers = [(r, c) for r in range(h) for c in range(w)
                    if int(arr_i[r, c]) == marker_color]
        if len(markers) != 1:
            ok = False; break
        mr, mc = markers[0]
        
        # Find nearest non-bg, non-marker cell
        best_dist = float('inf')
        nearest = None
        for r in range(h):
            for c in range(w):
                if arr_i[r, c] != bg and arr_i[r, c] != marker_color:
                    d = abs(r - mr) + abs(c - mc)
                    if d < best_dist:
                        best_dist = d
                        nearest = (r, c)
        
        if nearest is None:
            ok = False; break
        
        # Direction = away from nearest
        dr = 0 if mr == nearest[0] else (-1 if mr > nearest[0] else 1)
        dc = 0 if mc == nearest[1] else (-1 if mc > nearest[1] else 1)
        # Normalize to cardinal
        if abs(mr - nearest[0]) >= abs(mc - nearest[1]):
            dc = 0
            dr = -1 if mr < nearest[0] else 1
        else:
            dr = 0
            dc = -1 if mc < nearest[1] else 1
        
        # Shoot ray
        r, c = mr + dr, mc + dc
        local_color = None
        while 0 <= r < h and 0 <= c < w:
            if arr_i[r, c] != bg:
                break
            if arr_o[r, c] != bg:
                if local_color is None:
                    local_color = int(arr_o[r, c])
                elif local_color != int(arr_o[r, c]):
                    ok = False; break
            r += dr; c += dc
        
        if not ok or local_color is None:
            ok = False; break
        if ray_color is None:
            ray_color = local_color
        
        # Verify
        result = arr_i.copy()
        r, c = mr + dr, mc + dc
        while 0 <= r < h and 0 <= c < w:
            if arr_i[r, c] != bg:
                break
            result[r, c] = ray_color
            r += dr; c += dc
        
        if not np.array_equal(result, arr_o):
            ok = False; break
    
    if ok and ray_color is not None:
        return {'marker_color': marker_color, 'direction': 'away_from_nearest',
                'ray_color': ray_color, 'bg': bg}
    
    return None


def _apply_marker_ray(inp, params):
    mc = params['marker_color']
    bg = params['bg']
    ray_color = params['ray_color']
    arr = np.array(inp)
    h, w = arr.shape
    
    markers = [(r, c) for r in range(h) for c in range(w) if int(arr[r, c]) == mc]
    if len(markers) != 1:
        return None
    mr, mcc = markers[0]
    
    if params['direction'] == 'away_from_nearest':
        best_dist = float('inf')
        nearest = None
        for r in range(h):
            for c in range(w):
                if arr[r, c] != bg and arr[r, c] != mc:
                    d = abs(r - mr) + abs(c - mcc)
                    if d < best_dist:
                        best_dist = d
                        nearest = (r, c)
        if nearest is None:
            return None
        if abs(mr - nearest[0]) >= abs(mcc - nearest[1]):
            dr = -1 if mr < nearest[0] else 1
            dc = 0
        else:
            dr = 0
            dc = -1 if mcc < nearest[1] else 1
    else:
        dr, dc = params['direction']
    
    result = arr.copy()
    r, c = mr + dr, mcc + dc
    while 0 <= r < h and 0 <= c < w:
        if arr[r, c] != bg:
            break
        result[r, c] = ray_color
        r += dr; c += dc
    
    return [[int(v) for v in row] for row in result]


def _learn_marker_recolor(train_pairs, bg, marker_color):
    """Learn: recolor the object containing the marker."""
    # TODO
    return None


def _apply_marker_recolor(inp, params):
    return None


def _learn_marker_fill(train_pairs, bg, marker_color):
    """Learn: flood fill from marker position."""
    # TODO
    return None


def _apply_marker_fill(inp, params):
    return None


def _gen_panel_ops(train_pairs, bg, roles):
    """Operations on panel-divided grids."""
    # Existing panel_ops handles most of this
    # Add: per-panel conditional operations
    return []


def _gen_frame_ops(train_pairs, bg, roles):
    """Operations on nested frame structures."""
    pieces = []
    
    # Op: fill between frame layers with inner color
    rule = _learn_frame_fill(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            'mc:frame_fill',
            lambda inp, r=_r: _apply_frame_fill(inp, r)
        ))
    
    return pieces


def _learn_frame_fill(train_pairs, bg):
    """Learn: fill space between nested rectangles."""
    # Detect nested rectangles in input, check if output fills between them
    for inp, out in train_pairs:
        arr_i = np.array(inp); arr_o = np.array(out)
        if arr_i.shape != arr_o.shape:
            return None
    
    # Find rectangular objects
    arr = np.array(train_pairs[0][0])
    h, w = arr.shape
    non_bg_colors = sorted(set(int(v) for v in arr.flatten() if v != bg))
    
    for outer_color in non_bg_colors:
        outer_cells = list(zip(*np.where(arr == outer_color)))
        if not outer_cells:
            continue
        or1 = min(r for r, c in outer_cells); or2 = max(r for r, c in outer_cells)
        oc1 = min(c for r, c in outer_cells); oc2 = max(c for r, c in outer_cells)
        
        # Is there an inner color?
        inner_colors = set()
        for r in range(or1, or2 + 1):
            for c in range(oc1, oc2 + 1):
                v = int(arr[r, c])
                if v != bg and v != outer_color:
                    inner_colors.add(v)
        
        if len(inner_colors) != 1:
            continue
        inner_color = inner_colors.pop()
        
        # Check output: is the area between outer and inner frames filled?
        arr_o = np.array(train_pairs[0][1])
        
        # Build expected: fill bg cells inside outer bbox with inner_color
        result = arr.copy()
        for r in range(or1, or2 + 1):
            for c in range(oc1, oc2 + 1):
                if arr[r, c] == bg:
                    result[r, c] = inner_color
        
        if np.array_equal(result, arr_o):
            # Verify all
            ok = True
            for inp2, out2 in train_pairs[1:]:
                a2 = np.array(inp2)
                oc = list(zip(*np.where(a2 == outer_color)))
                if not oc:
                    ok = False; break
                r1 = min(r for r, c in oc); r2 = max(r for r, c in oc)
                c1 = min(c for r, c in oc); c2 = max(c for r, c in oc)
                ic = set()
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        v = int(a2[r, c])
                        if v != bg and v != outer_color:
                            ic.add(v)
                if len(ic) != 1:
                    ok = False; break
                fill_c = ic.pop()
                res = a2.copy()
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        if a2[r, c] == bg:
                            res[r, c] = fill_c
                if not np.array_equal(res, np.array(out2)):
                    ok = False; break
            
            if ok:
                return {'outer_color': outer_color, 'bg': bg}
    
    return None


def _apply_frame_fill(inp, params):
    bg = params['bg']
    outer_color = params['outer_color']
    arr = np.array(inp)
    h, w = arr.shape
    
    oc = list(zip(*np.where(arr == outer_color)))
    if not oc:
        return None
    r1 = min(r for r, c in oc); r2 = max(r for r, c in oc)
    c1 = min(c for r, c in oc); c2 = max(c for r, c in oc)
    
    ic = set()
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            v = int(arr[r, c])
            if v != bg and v != outer_color:
                ic.add(v)
    if len(ic) != 1:
        return None
    fill_c = ic.pop()
    
    result = arr.copy()
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if arr[r, c] == bg:
                result[r, c] = fill_c
    
    return [[int(v) for v in row] for row in result]


def _gen_mask_ops(train_pairs, bg, roles):
    """Operations using one object as a mask/template for another."""
    pieces = []
    
    # Op: stamp template shape at each target cell
    rule = _learn_template_stamp(train_pairs, bg, roles)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'mc:template_stamp',
            lambda inp, r=_r: _apply_template_stamp(inp, r)
        ))
    
    return pieces


def _learn_template_stamp(train_pairs, bg, roles):
    """Learn: use small object's shape as stamp at large object's cells."""
    tc = roles.template_color
    tgt = roles.target_color
    if tc is None or tgt is None:
        return None
    
    for inp, out in train_pairs:
        arr_i = np.array(inp); arr_o = np.array(out)
        if arr_i.shape != arr_o.shape:
            return None
    
    # Extract template shape from first pair
    arr = np.array(train_pairs[0][0])
    t_cells = list(zip(*np.where(arr == tc)))
    if not t_cells:
        return None
    tr1 = min(r for r, c in t_cells); tr2 = max(r for r, c in t_cells)
    tc1 = min(c for r, c in t_cells); tc2 = max(c for r, c in t_cells)
    template = np.zeros((tr2 - tr1 + 1, tc2 - tc1 + 1), dtype=bool)
    for r, c in t_cells:
        template[r - tr1, c - tc1] = True
    
    # Check: does stamping template at each target cell reproduce output?
    for inp, out in train_pairs:
        arr_i = np.array(inp); arr_o = np.array(out)
        h, w = arr_i.shape
        
        # Get template for this pair
        t_cells_p = list(zip(*np.where(arr_i == tc)))
        if not t_cells_p:
            return None
        tr1p = min(r for r, c in t_cells_p); tr2p = max(r for r, c in t_cells_p)
        tc1p = min(c for r, c in t_cells_p); tc2p = max(c for r, c in t_cells_p)
        templ = np.zeros((tr2p - tr1p + 1, tc2p - tc1p + 1), dtype=bool)
        for r, c in t_cells_p:
            templ[r - tr1p, c - tc1p] = True
        
        # Stamp at each target cell
        tgt_cells = list(zip(*np.where(arr_i == tgt)))
        result = arr_i.copy()
        for tr, ttc in tgt_cells:
            for dr in range(templ.shape[0]):
                for dc in range(templ.shape[1]):
                    if templ[dr, dc]:
                        nr, nc = tr + dr, ttc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            result[nr, nc] = tgt
        
        if not np.array_equal(result, arr_o):
            return None
    
    return {'template_color': tc, 'target_color': tgt, 'bg': bg}


def _apply_template_stamp(inp, params):
    tc = params['template_color']
    tgt = params['target_color']
    bg = params['bg']
    arr = np.array(inp)
    h, w = arr.shape
    
    t_cells = list(zip(*np.where(arr == tc)))
    if not t_cells:
        return None
    tr1 = min(r for r, c in t_cells); tr2 = max(r for r, c in t_cells)
    tc1 = min(c for r, c in t_cells); tc2 = max(c for r, c in t_cells)
    templ = np.zeros((tr2 - tr1 + 1, tc2 - tc1 + 1), dtype=bool)
    for r, c in t_cells:
        templ[r - tr1, c - tc1] = True
    
    tgt_cells = list(zip(*np.where(arr == tgt)))
    result = arr.copy()
    for tr, ttc in tgt_cells:
        for dr in range(templ.shape[0]):
            for dc in range(templ.shape[1]):
                if templ[dr, dc]:
                    nr, nc = tr + dr, ttc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = tgt
    
    return [[int(v) for v in row] for row in result]


def _gen_symmetry_repair_ops(train_pairs, bg):
    """Fix symmetry breaks."""
    pieces = []
    
    for sym_type, sym_fn, repair_fn in [
        ('lr', lambda a: np.fliplr(a), lambda a: _repair_sym(a, np.fliplr, bg)),
        ('ud', lambda a: np.flipud(a), lambda a: _repair_sym(a, np.flipud, bg)),
        ('rot90', lambda a: np.rot90(a), lambda a: _repair_sym(a, np.rot90, bg)),
    ]:
        ok = True
        for inp, out in train_pairs:
            result = repair_fn(np.array(inp))
            if result is None or not grid_eq(result.tolist() if hasattr(result, 'tolist') else result, out):
                ok = False; break
        
        if ok:
            _sf = sym_fn
            _bg = bg
            def _apply_sym(inp, sf=_sf, b=_bg):
                r = _repair_sym(np.array(inp), sf, b)
                return r.tolist() if r is not None else None
            pieces.append(CrossPiece(
                f'mc:sym_repair_{sym_type}',
                _apply_sym
            ))
    
    return pieces


def _repair_sym(arr, sym_fn, bg):
    """Repair a grid to be symmetric: where cells differ, use the non-bg value."""
    flipped = sym_fn(arr)
    if flipped.shape != arr.shape:
        return None
    result = arr.copy()
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if arr[r, c] != flipped[r, c]:
                # Take the non-bg value (or flipped if both non-bg)
                if arr[r, c] == bg:
                    result[r, c] = flipped[r, c]
                elif flipped[r, c] == bg:
                    pass  # keep original
                else:
                    # Both non-bg but different: take max (arbitrary but consistent)
                    result[r, c] = max(int(arr[r, c]), int(flipped[r, c]))
    return result


def _gen_color_group_ops(train_pairs, bg):
    """Operations on groups of same-color disconnected cells."""
    pieces = []
    
    # Op: connect same-color clusters (draw lines between centroids)
    # Op: fill bounding box of all same-color cells
    rule = _learn_color_bbox_fill(train_pairs, bg)
    if rule:
        _r = rule
        pieces.append(CrossPiece(
            f'mc:color_bbox_fill',
            lambda inp, r=_r: _apply_color_bbox_fill(inp, r)
        ))
    
    return pieces


def _learn_color_bbox_fill(train_pairs, bg):
    """Fill bounding box of all same-color cells with that color."""
    for inp, out in train_pairs:
        arr_i = np.array(inp); arr_o = np.array(out)
        if arr_i.shape != arr_o.shape:
            return None
        
        h, w = arr_i.shape
        result = arr_i.copy()
        
        for color in set(int(v) for v in arr_i.flatten() if v != bg):
            cells = list(zip(*np.where(arr_i == color)))
            if len(cells) < 2:
                continue
            r1 = min(r for r, c in cells); r2 = max(r for r, c in cells)
            c1 = min(c for r, c in cells); c2 = max(c for r, c in cells)
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if result[r, c] == bg:
                        result[r, c] = color
        
        if not np.array_equal(result, arr_o):
            return None
    
    return {'bg': bg}


def _apply_color_bbox_fill(inp, params):
    bg = params['bg']
    arr = np.array(inp)
    h, w = arr.shape
    result = arr.copy()
    
    for color in set(int(v) for v in arr.flatten() if v != bg):
        cells = list(zip(*np.where(arr == color)))
        if len(cells) < 2:
            continue
        r1 = min(r for r, c in cells); r2 = max(r for r, c in cells)
        c1 = min(c for r, c in cells); c2 = max(c for r, c in cells)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if result[r, c] == bg:
                    result[r, c] = color
    
    return [[int(v) for v in row] for row in result]


# ============================================================
# Main Entry Point
# ============================================================

def generate_meta_cross_pieces(
    train_pairs: List[Tuple[Grid, Grid]]
) -> List[CrossPiece]:
    """Main entry: detect concepts → assign roles → generate operations."""
    if not HAS_NUMPY:
        return []
    
    try:
        bg = most_common_color(train_pairs[0][0])
        inp0 = train_pairs[0][0]
        
        # Step 1: Detect concepts
        sig = detect_concepts(inp0, bg)
        
        # Step 2: Assign roles
        roles = assign_roles(inp0, bg, sig)
        
        # Step 3: Generate operation trees
        pieces = generate_operation_trees(train_pairs, sig, roles)
        
        return pieces
    except Exception:
        return []
