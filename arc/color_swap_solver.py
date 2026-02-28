"""
arc/color_swap_solver.py — Color Swap Solver

Targets tasks where input→output involves at most 2 types of color changes.
Learns the CONDITION that determines which cells change.

Strategy: try multiple condition hypotheses, pick the one that works on all training pairs.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter, defaultdict
from scipy import ndimage

Grid = List[List[int]]


def _label_objects(grid: np.ndarray, bg: int) -> Tuple[np.ndarray, int]:
    nonbg = (grid != bg).astype(int)
    labeled, n = ndimage.label(nonbg)
    return labeled, n


def _label_per_color(grid: np.ndarray) -> np.ndarray:
    """Each color gets separate connected components"""
    h, w = grid.shape
    result = np.zeros((h, w), dtype=int)
    obj_id = 0
    for color in range(10):
        mask = (grid == color)
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        for lbl in range(1, n + 1):
            obj_id += 1
            result[labeled == lbl] = obj_id
    return result


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.flatten())
    return counts.most_common(1)[0][0]


def _object_properties(grid: np.ndarray, labeled: np.ndarray) -> Dict:
    """Compute properties for each object"""
    props = {}
    obj_ids = set(int(v) for v in labeled.flatten()) - {0}
    h, w = grid.shape
    
    for oid in obj_ids:
        mask = (labeled == oid)
        rows, cols = np.where(mask)
        area = len(rows)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        bbox_h, bbox_w = r1 - r0 + 1, c1 - c0 + 1
        centroid_r, centroid_c = rows.mean(), cols.mean()
        colors = Counter(int(grid[r, c]) for r, c in zip(rows, cols))
        dominant = colors.most_common(1)[0][0]
        border = bool(r0 == 0 or r1 == h-1 or c0 == 0 or c1 == w-1)
        
        # Convexity: area / bbox_area
        bbox_area = bbox_h * bbox_w
        convexity = area / bbox_area if bbox_area > 0 else 0
        
        props[oid] = {
            'area': area, 'bbox': (r0, c0, r1, c1),
            'bbox_h': bbox_h, 'bbox_w': bbox_w,
            'centroid': (centroid_r, centroid_c),
            'color': dominant, 'colors': colors,
            'border': border, 'convexity': convexity,
        }
    return props


# ============================================================
# Hypothesis generators — each returns a cell classifier
# ============================================================

def _hyp_obj_convexity_fill(grid, labeled, props, bg):
    """Hypothesis: fill concave parts of objects (convex hull completion)"""
    h, w = grid.shape
    fill_cells = set()
    
    obj_ids = set(int(v) for v in labeled.flatten()) - {0}
    for oid in obj_ids:
        if oid not in props:
            continue
        p = props[oid]
        r0, c0, r1, c1 = p['bbox']
        mask = (labeled == oid)
        
        # For each row in bbox, find leftmost and rightmost cell
        for r in range(r0, r1 + 1):
            row_cells = [c for c in range(c0, c1 + 1) if mask[r, c]]
            if len(row_cells) >= 2:
                left, right = min(row_cells), max(row_cells)
                for c in range(left, right + 1):
                    if not mask[r, c]:
                        fill_cells.add((r, c))
        
        # For each col in bbox, find topmost and bottommost
        for c in range(c0, c1 + 1):
            col_cells = [r for r in range(r0, r1 + 1) if mask[r, c]]
            if len(col_cells) >= 2:
                top, bottom = min(col_cells), max(col_cells)
                for r in range(top, bottom + 1):
                    if not mask[r, c]:
                        fill_cells.add((r, c))
    
    return fill_cells


def _hyp_symmetry_repair(grid, labeled, props, bg):
    """Hypothesis: make each object symmetric (horizontal, vertical, or both)"""
    h, w = grid.shape
    changes = {}  # (r, c) -> new_color
    
    obj_ids = set(int(v) for v in labeled.flatten()) - {0}
    for oid in obj_ids:
        if oid not in props:
            continue
        p = props[oid]
        mask = (labeled == oid)
        r0, c0, r1, c1 = p['bbox']
        cr = (r0 + r1) / 2
        cc = (c0 + c1) / 2
        color = p['color']
        
        # Try horizontal symmetry
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mr = int(2 * cr - r)
                if r0 <= mr <= r1:
                    if mask[r, c] and not mask[mr, c]:
                        changes[(mr, c)] = color
                    elif mask[mr, c] and not mask[r, c]:
                        changes[(r, c)] = color
        
        # Try vertical symmetry
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mc = int(2 * cc - c)
                if c0 <= mc <= c1:
                    if mask[r, c] and not mask[r, mc]:
                        changes[(r, mc)] = color
                    elif mask[r, mc] and not mask[r, c]:
                        changes[(r, c)] = color
    
    return changes


def _hyp_corner_extend(grid, labeled, props, bg):
    """Hypothesis: L-shaped objects get corners filled/extended"""
    h, w = grid.shape
    changes = {}
    
    obj_ids = set(int(v) for v in labeled.flatten()) - {0}
    for oid in obj_ids:
        if oid not in props:
            continue
        mask = (labeled == oid)
        p = props[oid]
        color = p['color']
        r0, c0, r1, c1 = p['bbox']
        
        # Find "elbow" of L-shape: cells where object turns 90 degrees
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not mask[r, c]:
                    # Count adjacent object cells
                    adj = 0
                    dirs = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                            adj += 1
                            dirs.append((dr, dc))
                    # If exactly 2 adjacent and they form an L (perpendicular)
                    if adj == 2:
                        (d1r, d1c), (d2r, d2c) = dirs
                        if d1r * d2r + d1c * d2c == 0:  # perpendicular
                            changes[(r, c)] = color
    
    return changes


def _hyp_protrusion_trim(grid, labeled, props, bg):
    """Hypothesis: trim single-cell protrusions from objects"""
    h, w = grid.shape
    changes = {}
    
    obj_ids = set(int(v) for v in labeled.flatten()) - {0}
    for oid in obj_ids:
        if oid not in props:
            continue
        mask = (labeled == oid)
        
        for r, c in zip(*np.where(mask)):
            # Count N4 neighbors in same object
            n4_same = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                          if 0 <= r+dr < h and 0 <= c+dc < w and mask[r+dr, c+dc])
            if n4_same == 1:  # Only one neighbor = protrusion
                changes[(r, c)] = bg
    
    return changes


def _hyp_line_extend_to_border(grid, labeled, props, bg):
    """Hypothesis: extend lines from object tips to grid border or other objects"""
    h, w = grid.shape
    changes = {}
    
    obj_ids = set(int(v) for v in labeled.flatten()) - {0}
    for oid in obj_ids:
        if oid not in props:
            continue
        mask = (labeled == oid)
        color = p = props[oid]['color']
        
        # Find tip cells (exactly 1 neighbor in object)
        tips = []
        for r, c in zip(*np.where(mask)):
            neighbors = [(dr, dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                         if 0 <= r+dr < h and 0 <= c+dc < w and mask[r+dr, c+dc]]
            if len(neighbors) == 1:
                # Direction away from neighbor
                nr_dr, nr_dc = neighbors[0]
                ext_dr, ext_dc = -nr_dr, -nr_dc
                tips.append((r, c, ext_dr, ext_dc))
        
        # Extend each tip
        for r, c, dr, dc in tips:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                changes[(nr, nc)] = color
                nr += dr
                nc += dc
    
    return changes


def _hyp_marker_spread(grid, labeled, props, bg):
    """Hypothesis: special marker cells spread their color to nearby bg cells"""
    h, w = grid.shape
    changes = {}
    
    # Find "marker" colors (appear rarely, only 1-2 cells)
    color_counts = Counter(int(v) for v in grid.flatten())
    rare_colors = [c for c, cnt in color_counts.items() if c != bg and cnt <= 3]
    
    for rare_c in rare_colors:
        positions = list(zip(*np.where(grid == rare_c)))
        for r, c in positions:
            # Find which object this marker is inside/near
            oid = int(labeled[r, c])
            if oid > 0:
                continue  # marker is part of an object, skip
            
            # Spread to N4 bg cells
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == bg:
                    changes[(nr, nc)] = rare_c
    
    return changes


def _hyp_separator_panel_transform(grid, bg):
    """Hypothesis: separator divides grid into panels; transform based on panel comparison"""
    h, w = grid.shape
    
    # Find separator rows/cols
    sep_rows = []
    sep_cols = []
    sep_color = None
    
    for r in range(h):
        if len(set(grid[r])) == 1 and grid[r][0] != bg:
            sep_rows.append(r)
            sep_color = int(grid[r][0])
    
    for c in range(w):
        col = grid[:, c]
        if len(set(col)) == 1 and col[0] != bg:
            sep_cols.append(c)
            if sep_color is None:
                sep_color = int(col[0])
    
    if not sep_rows and not sep_cols:
        return None
    
    # Split into panels
    row_bounds = [0] + [r for r in sep_rows] + [h]
    col_bounds = [0] + [c for c in sep_cols] + [w]
    
    panels = []
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r0, r1 = row_bounds[i], row_bounds[i+1]
            c0, c1 = col_bounds[j], col_bounds[j+1]
            if r0 in sep_rows: r0 += 1
            if c0 in sep_cols: c0 += 1
            if r1 > r0 and c1 > c0:
                panels.append((r0, c0, r1, c1))
    
    return panels


# ============================================================
# Main solver: try all hypotheses, verify on training
# ============================================================

def learn_color_swap_rule(train_pairs: List[Tuple[Grid, Grid]], 
                          bg: int = None) -> Optional[Dict]:
    """Learn which color swap rule applies"""
    
    # Check same-size and low-diff
    for inp, out in train_pairs:
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
    
    np_pairs = [(np.array(i), np.array(o)) for i, o in train_pairs]
    
    if bg is None:
        bg = _get_bg(np_pairs[0][0])
    
    # Collect all changes across training pairs
    all_changes = []
    for inp, out in np_pairs:
        diff_mask = inp != out
        n_diff = diff_mask.sum()
        total = inp.size
        if n_diff == 0 or n_diff / total > 0.20:
            return None  # Too many changes or no changes
        
        changes = {}
        for r, c in zip(*np.where(diff_mask)):
            changes[(int(r), int(c))] = (int(inp[r, c]), int(out[r, c]))
        all_changes.append(changes)
    
    # Try each hypothesis
    hypotheses = [
        ('corner_extend', _try_corner_extend),
        ('symmetry_h', lambda pairs, bg: _try_symmetry(pairs, bg, 'h')),
        ('symmetry_v', lambda pairs, bg: _try_symmetry(pairs, bg, 'v')),
        ('symmetry_hv', lambda pairs, bg: _try_symmetry(pairs, bg, 'hv')),
        ('convex_fill', _try_convex_fill),
        ('protrusion_trim', _try_protrusion_trim),
        ('obj_interior_color', _try_obj_interior_recolor),
        ('odd_cell_in_obj', _try_odd_cell_recolor),
        ('border_cell_recolor', _try_border_cell_recolor),
        ('ray_from_dot', _try_ray_from_dot),
    ]
    
    for name, try_fn in hypotheses:
        try:
            rule = try_fn(np_pairs, bg)
            if rule is not None:
                rule['type'] = name
                rule['bg'] = bg
                return rule
        except Exception:
            continue
    
    return None


def _try_corner_extend(np_pairs, bg):
    """L-shaped corner filling"""
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        predicted_changes = _hyp_corner_extend(inp, labeled, props, bg)
        
        # Compare with actual changes
        actual_changes = {}
        diff_mask = inp != out
        for r, c in zip(*np.where(diff_mask)):
            actual_changes[(int(r), int(c))] = int(out[r, c])
        
        if set(predicted_changes.keys()) != set(actual_changes.keys()):
            return None
        for pos in predicted_changes:
            if predicted_changes[pos] != actual_changes[pos]:
                return None
    
    return {'strategy': 'corner_extend'}


def _try_symmetry(np_pairs, bg, axis):
    """Symmetry completion for objects"""
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        h, w = inp.shape
        
        predicted = inp.copy()
        obj_ids = set(int(v) for v in labeled.flatten()) - {0}
        
        for oid in obj_ids:
            if oid not in props:
                continue
            mask = (labeled == oid)
            p = props[oid]
            r0, c0, r1, c1 = p['bbox']
            color = p['color']
            cr = (r0 + r1) / 2.0
            cc = (c0 + c1) / 2.0
            
            if axis in ('h', 'hv'):
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        mr = round(2 * cr - r)
                        if r0 <= mr <= r1:
                            if mask[r, c] and 0 <= mr < h and not mask[mr, c]:
                                predicted[mr, c] = color
            
            if axis in ('v', 'hv'):
                # Re-read mask after h-symmetry
                if axis == 'hv':
                    mask = (predicted != bg) & ((labeled == oid) | (predicted != inp))
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        mc = round(2 * cc - c)
                        if c0 <= mc <= c1:
                            if mask[r, c] and 0 <= mc < w and predicted[r, mc] == bg:
                                predicted[r, mc] = color
        
        if not np.array_equal(predicted, out):
            return None
    
    return {'strategy': f'symmetry_{axis}'}


def _try_convex_fill(np_pairs, bg):
    """Fill concave parts of objects"""
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        
        fill_cells = _hyp_obj_convexity_fill(inp, labeled, props, bg)
        
        predicted = inp.copy()
        for r, c in fill_cells:
            # Find which object's color to use
            # Check N4 neighbors for object color
            h, w = inp.shape
            neighbor_oids = set()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and labeled[nr, nc] > 0:
                    neighbor_oids.add(int(labeled[nr, nc]))
            if neighbor_oids:
                oid = min(neighbor_oids)  # pick one
                if oid in props:
                    predicted[r, c] = props[oid]['color']
        
        if not np.array_equal(predicted, out):
            return None
    
    return {'strategy': 'convex_fill'}


def _try_protrusion_trim(np_pairs, bg):
    """Trim single-cell protrusions"""
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        h, w = inp.shape
        
        predicted = inp.copy()
        changes = _hyp_protrusion_trim(inp, labeled, props, bg)
        for (r, c), color in changes.items():
            predicted[r, c] = color
        
        if not np.array_equal(predicted, out):
            return None
    
    return {'strategy': 'protrusion_trim'}


def _try_obj_interior_recolor(np_pairs, bg):
    """Interior cells of objects get recolored"""
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        h, w = inp.shape
        
        # Find changes
        diff_mask = inp != out
        if not diff_mask.any():
            return None
        
        # Check if all changed cells are interior of some object
        for r, c in zip(*np.where(diff_mask)):
            oid = int(labeled[r, c])
            if oid == 0:
                return None  # changed cell not in object
            # Check if interior (all N4 in same object)
            n4_same = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                          if 0 <= r+dr < h and 0 <= c+dc < w and labeled[r+dr, c+dc] == oid)
            n4_total = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                           if 0 <= r+dr < h and 0 <= c+dc < w)
            if n4_same < n4_total:
                return None  # Not interior
    
    # Learn the recolor rule: what color do interior cells become?
    # Collect (obj_color, obj_area_bucket, old_color) → new_color
    rule_map = {}
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        h, w = inp.shape
        
        diff_mask = inp != out
        for r, c in zip(*np.where(diff_mask)):
            oid = int(labeled[r, c])
            if oid in props:
                obj_color = props[oid]['color']
                area = props[oid]['area']
                ab = 0 if area < 5 else (1 if area < 20 else 2)
                key = (obj_color, ab, int(inp[r, c]))
                val = int(out[r, c])
                if key in rule_map and rule_map[key] != val:
                    return None
                rule_map[key] = val
    
    return {'strategy': 'obj_interior_recolor', 'rule_map': rule_map}


def _try_odd_cell_recolor(np_pairs, bg):
    """Odd-one-out cell within object gets recolored"""
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        h, w = inp.shape
        
        diff_mask = inp != out
        
        for r, c in zip(*np.where(diff_mask)):
            oid = int(labeled[r, c])
            if oid == 0:
                return None
            
            # Check if this cell has a different color from most of its object
            if oid in props:
                dominant = props[oid]['color']
                if int(inp[r, c]) == dominant:
                    return None  # It's not the odd one
    
    # Learn: odd cells get recolored to what?
    rule_map = {}
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        props = _object_properties(inp, labeled)
        diff_mask = inp != out
        
        for r, c in zip(*np.where(diff_mask)):
            oid = int(labeled[r, c])
            if oid in props:
                dominant = props[oid]['color']
                key = (int(inp[r, c]), dominant)
                val = int(out[r, c])
                if key in rule_map and rule_map[key] != val:
                    return None
                rule_map[key] = val
    
    return {'strategy': 'odd_cell_recolor', 'rule_map': rule_map}


def _try_border_cell_recolor(np_pairs, bg):
    """Border cells of objects (edge touching bg) get recolored"""
    for inp, out in np_pairs:
        labeled, n = _label_objects(inp, bg)
        h, w = inp.shape
        
        diff_mask = inp != out
        
        for r, c in zip(*np.where(diff_mask)):
            oid = int(labeled[r, c])
            if oid == 0:
                return None
            # Must be border of object (at least one N4 = bg or OOB)
            has_bg_neighbor = any(
                (not (0 <= r+dr < h and 0 <= c+dc < w)) or labeled[r+dr, c+dc] != oid
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if not has_bg_neighbor:
                return None
    
    return {'strategy': 'border_cell_recolor'}


def _try_ray_from_dot(np_pairs, bg):
    """Dots shoot rays in cardinal directions"""
    for inp, out in np_pairs:
        h, w = inp.shape
        diff_mask = inp != out
        
        # Find isolated dots in input (single non-bg cells with no same-color N4)
        dots = []
        for r in range(h):
            for c in range(w):
                if inp[r, c] == bg:
                    continue
                n4_same = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                              if 0 <= r+dr < h and 0 <= c+dc < w and inp[r+dr, c+dc] == inp[r, c])
                if n4_same == 0:
                    dots.append((r, c, int(inp[r, c])))
        
        if not dots:
            return None
        
        # Check if changes are along rays from dots
        predicted = inp.copy()
        for r, c, color in dots:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w:
                    if inp[nr, nc] != bg:
                        break
                    predicted[nr, nc] = color
                    nr += dr
                    nc += dc
        
        if not np.array_equal(predicted, out):
            return None
    
    return {'strategy': 'ray_from_dot'}


# ============================================================
# Apply functions
# ============================================================

def apply_color_swap_rule(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply learned color swap rule"""
    grid = np.array(inp)
    bg = rule.get('bg', 0)
    strategy = rule['strategy']
    
    if strategy == 'corner_extend':
        labeled, n = _label_objects(grid, bg)
        props = _object_properties(grid, labeled)
        changes = _hyp_corner_extend(grid, labeled, props, bg)
        result = grid.copy()
        for (r, c), color in changes.items():
            result[r, c] = color
        return result.tolist()
    
    elif strategy.startswith('symmetry_'):
        axis = strategy.split('_')[1]
        labeled, n = _label_objects(grid, bg)
        props = _object_properties(grid, labeled)
        h, w = grid.shape
        result = grid.copy()
        
        obj_ids = set(int(v) for v in labeled.flatten()) - {0}
        for oid in obj_ids:
            if oid not in props:
                continue
            mask = (labeled == oid)
            p = props[oid]
            r0, c0, r1, c1 = p['bbox']
            color = p['color']
            cr = (r0 + r1) / 2.0
            cc = (c0 + c1) / 2.0
            
            if axis in ('h', 'hv'):
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        mr = round(2 * cr - r)
                        if r0 <= mr <= r1 and 0 <= mr < h:
                            if mask[r, c] and not mask[mr, c]:
                                result[mr, c] = color
            
            if axis in ('v', 'hv'):
                mask2 = (result != bg) & ((labeled == oid) | (result != grid))
                for r in range(r0, r1 + 1):
                    for c in range(c0, c1 + 1):
                        mc = round(2 * cc - c)
                        if c0 <= mc <= c1 and 0 <= mc < w:
                            if mask2[r, c] and result[r, mc] == bg:
                                result[r, mc] = color
        
        return result.tolist()
    
    elif strategy == 'convex_fill':
        labeled, n = _label_objects(grid, bg)
        props = _object_properties(grid, labeled)
        fill_cells = _hyp_obj_convexity_fill(grid, labeled, props, bg)
        h, w = grid.shape
        result = grid.copy()
        for r, c in fill_cells:
            neighbor_oids = set()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and labeled[nr, nc] > 0:
                    neighbor_oids.add(int(labeled[nr, nc]))
            if neighbor_oids:
                oid = min(neighbor_oids)
                if oid in props:
                    result[r, c] = props[oid]['color']
        return result.tolist()
    
    elif strategy == 'protrusion_trim':
        labeled, n = _label_objects(grid, bg)
        props = _object_properties(grid, labeled)
        changes = _hyp_protrusion_trim(grid, labeled, props, bg)
        result = grid.copy()
        for (r, c), color in changes.items():
            result[r, c] = color
        return result.tolist()
    
    elif strategy == 'obj_interior_recolor':
        rule_map = rule['rule_map']
        labeled, n = _label_objects(grid, bg)
        props = _object_properties(grid, labeled)
        h, w = grid.shape
        result = grid.copy()
        
        for r in range(h):
            for c in range(w):
                oid = int(labeled[r, c])
                if oid == 0:
                    continue
                # Check if interior
                n4_same = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                              if 0 <= r+dr < h and 0 <= c+dc < w and labeled[r+dr, c+dc] == oid)
                n4_total = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                               if 0 <= r+dr < h and 0 <= c+dc < w)
                if n4_same == n4_total:  # interior
                    if oid in props:
                        obj_color = props[oid]['color']
                        area = props[oid]['area']
                        ab = 0 if area < 5 else (1 if area < 20 else 2)
                        key = (obj_color, ab, int(grid[r, c]))
                        if key in rule_map:
                            result[r, c] = rule_map[key]
        
        return result.tolist()
    
    elif strategy == 'odd_cell_recolor':
        rule_map = rule['rule_map']
        labeled, n = _label_objects(grid, bg)
        props = _object_properties(grid, labeled)
        h, w = grid.shape
        result = grid.copy()
        
        for r in range(h):
            for c in range(w):
                oid = int(labeled[r, c])
                if oid == 0 or oid not in props:
                    continue
                dominant = props[oid]['color']
                cell_color = int(grid[r, c])
                if cell_color != dominant:
                    key = (cell_color, dominant)
                    if key in rule_map:
                        result[r, c] = rule_map[key]
        
        return result.tolist()
    
    elif strategy == 'ray_from_dot':
        h, w = grid.shape
        result = grid.copy()
        
        dots = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bg:
                    continue
                n4_same = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                              if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr, c+dc] == grid[r, c])
                if n4_same == 0:
                    dots.append((r, c, int(grid[r, c])))
        
        for r, c, color in dots:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w:
                    if grid[nr, nc] != bg:
                        break
                    result[nr, nc] = color
                    nr += dr
                    nc += dc
        
        return result.tolist()
    
    return None


# ============================================================
# Integration: generate CrossPiece objects
# ============================================================

def generate_color_swap_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> list:
    """Generate CrossPiece objects for cross_engine integration"""
    from arc.cross_engine import CrossPiece
    from arc.grid import most_common_color
    
    pieces = []
    if not train_pairs:
        return pieces
    
    bg = most_common_color(train_pairs[0][0])
    
    rule = learn_color_swap_rule(train_pairs, bg)
    if rule is not None:
        pieces.append(CrossPiece(
            f'color_swap:{rule["strategy"]}',
            lambda inp, _r=rule: apply_color_swap_rule(inp, _r)
        ))
    
    return pieces
