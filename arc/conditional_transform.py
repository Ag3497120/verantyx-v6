"""
arc/conditional_transform.py — Conditional Object Transforms for ARC-AGI-2

Handles per-object conditional transforms:
1. Object interior fill (hollow -> filled)
2. Line extension from objects (extend colored cells in cardinal directions)
3. Object-conditional recolor (based on size/shape/position)
4. Boundary completion (complete partial rectangles/shapes)
5. Connect objects (draw lines between same-colored objects)
6. Background to color based on nearby objects (bg→color pattern)

Addresses ver=0 tasks with:
- 48 bg→color pattern tasks
- 89 other_recolor pattern tasks
- Many same_size_small_diff tasks requiring per-object transforms
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.objects import detect_objects, ArcObject


# ============================================================
# Object Interior Fill
# ============================================================

def _is_hollow_object(obj: ArcObject, grid: Grid, bg: int) -> bool:
    """Check if object has hollow interior (bg cells inside bbox)"""
    r1, c1, r2, c2 = obj.bbox
    obj_cells = set(obj.cells)

    # Check for bg cells inside bbox that are not on the border
    for r in range(r1 + 1, r2):
        for c in range(c1 + 1, c2):
            if (r, c) not in obj_cells and grid[r][c] == bg:
                return True
    return False


def _fill_interior(obj: ArcObject, grid: Grid, bg: int, fill_color: int) -> Grid:
    """Fill interior of object (flood fill within bbox)"""
    result = [row[:] for row in grid]
    r1, c1, r2, c2 = obj.bbox
    obj_cells = set(obj.cells)

    # Flood fill from each bg cell inside bbox
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            if (r, c) not in obj_cells and grid[r][c] == bg:
                result[r][c] = fill_color

    return result


def learn_fill_object_interior(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn to fill hollow object interiors"""
    for bg in [0, most_common_color(train_pairs[0][0])]:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            if grid_shape(out) != (h, w):
                ok = False
                break

            objs = detect_objects(inp, bg)
            result = [row[:] for row in inp]

            for obj in objs:
                if _is_hollow_object(obj, inp, bg):
                    # Fill with object's color
                    result = _fill_interior(obj, result, bg, obj.color)

            if not grid_eq(result, out):
                ok = False
                break

        if ok:
            return {'type': 'fill_interior', 'bg': bg}

    return None


def apply_fill_object_interior(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply interior fill transform"""
    bg = params['bg']
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]

    for obj in objs:
        if _is_hollow_object(obj, inp, bg):
            result = _fill_interior(obj, result, bg, obj.color)

    return result


# ============================================================
# Line Extension from Objects
# ============================================================

def _extend_line_from_object(grid: Grid, obj: ArcObject, bg: int,
                             direction: str) -> Grid:
    """Extend colored cells in cardinal direction until hitting boundary/other color"""
    result = [row[:] for row in grid]
    h, w = grid_shape(grid)

    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
        'all': [(-1, 0), (1, 0), (0, -1), (0, 1)]
    }

    if direction == 'all':
        dirs = directions['all']
    else:
        dirs = [directions[direction]]

    for r, c in obj.cells:
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            while 0 <= nr < h and 0 <= nc < w and result[nr][nc] == bg:
                result[nr][nc] = obj.color
                nr, nc = nr + dr, nc + dc

    return result


def learn_line_extension_from_objects(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn to extend lines from objects in specific directions"""
    for bg in [0, most_common_color(train_pairs[0][0])]:
        for direction in ['all', 'up', 'down', 'left', 'right']:
            ok = True
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                if grid_shape(out) != (h, w):
                    ok = False
                    break

                objs = detect_objects(inp, bg)
                result = [row[:] for row in inp]

                for obj in objs:
                    result = _extend_line_from_object(result, obj, bg, direction)

                if not grid_eq(result, out):
                    ok = False
                    break

            if ok:
                return {'type': 'line_extend_objects', 'bg': bg, 'direction': direction}

    return None


def apply_line_extension_from_objects(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply line extension from objects"""
    bg = params['bg']
    direction = params['direction']
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]

    for obj in objs:
        result = _extend_line_from_object(result, obj, bg, direction)

    return result


# ============================================================
# Background to Color Based on Nearby Objects
# ============================================================

def _get_nearby_object_color(r: int, c: int, objs: List[ArcObject],
                             h: int, w: int) -> Optional[int]:
    """Get color of nearest object to position (r, c)"""
    min_dist = float('inf')
    nearest_color = None

    for obj in objs:
        for or_, oc in obj.cells:
            dist = abs(r - or_) + abs(c - oc)
            if dist < min_dist:
                min_dist = dist
                nearest_color = obj.color

    return nearest_color


def learn_bg_to_nearby_color(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn to color background cells based on nearby objects

    Pattern: Background cells become colored based on the nearest object's color
    Common in 'projection' or 'influence' tasks.
    """
    for bg in [0, most_common_color(train_pairs[0][0])]:
        # Try different strategies
        for strategy in ['nearest_object', 'surrounding_box', 'aligned_object']:
            ok = True

            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                if grid_shape(out) != (h, w):
                    ok = False
                    break

                objs = detect_objects(inp, bg)
                if not objs:
                    ok = False
                    break

                result = [row[:] for row in inp]

                if strategy == 'nearest_object':
                    # Color each bg cell with nearest object's color
                    for r in range(h):
                        for c in range(w):
                            if inp[r][c] == bg:
                                color = _get_nearby_object_color(r, c, objs, h, w)
                                if color is not None:
                                    result[r][c] = color

                elif strategy == 'surrounding_box':
                    # For each object, fill surrounding box with object color
                    for obj in objs:
                        r1, c1, r2, c2 = obj.bbox
                        # Expand bbox by 1
                        for r in range(max(0, r1 - 1), min(h, r2 + 2)):
                            for c in range(max(0, c1 - 1), min(w, c2 + 2)):
                                if inp[r][c] == bg:
                                    result[r][c] = obj.color

                elif strategy == 'aligned_object':
                    # Fill bg cells that are row/col aligned with object
                    for obj in objs:
                        r1, c1, r2, c2 = obj.bbox
                        # Fill aligned columns
                        for c in range(c1, c2 + 1):
                            for r in range(h):
                                if inp[r][c] == bg:
                                    result[r][c] = obj.color
                        # Fill aligned rows
                        for r in range(r1, r2 + 1):
                            for c in range(w):
                                if inp[r][c] == bg:
                                    result[r][c] = obj.color

                if not grid_eq(result, out):
                    ok = False
                    break

            if ok:
                return {'type': 'bg_to_color', 'bg': bg, 'strategy': strategy}

    return None


def apply_bg_to_nearby_color(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply background to color transform"""
    bg = params['bg']
    strategy = params['strategy']
    h, w = grid_shape(inp)
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]

    if strategy == 'nearest_object':
        for r in range(h):
            for c in range(w):
                if inp[r][c] == bg:
                    color = _get_nearby_object_color(r, c, objs, h, w)
                    if color is not None:
                        result[r][c] = color

    elif strategy == 'surrounding_box':
        for obj in objs:
            r1, c1, r2, c2 = obj.bbox
            for r in range(max(0, r1 - 1), min(h, r2 + 2)):
                for c in range(max(0, c1 - 1), min(w, c2 + 2)):
                    if inp[r][c] == bg:
                        result[r][c] = obj.color

    elif strategy == 'aligned_object':
        for obj in objs:
            r1, c1, r2, c2 = obj.bbox
            for c in range(c1, c2 + 1):
                for r in range(h):
                    if inp[r][c] == bg:
                        result[r][c] = obj.color
            for r in range(r1, r2 + 1):
                for c in range(w):
                    if inp[r][c] == bg:
                        result[r][c] = obj.color

    return result


# ============================================================
# Boundary/Shape Completion
# ============================================================

def _detect_partial_rectangle(obj: ArcObject, grid: Grid) -> Optional[Tuple[int, int, int, int]]:
    """Detect if object is partial rectangle and return completed bbox"""
    r1, c1, r2, c2 = obj.bbox

    # Check if object has cells forming an incomplete rectangle
    # (e.g., only edges, missing corners, etc.)
    obj_cells = set(obj.cells)

    # Count cells on edges vs interior
    edge_cells = 0
    for r, c in obj.cells:
        if r == r1 or r == r2 or c == c1 or c == c2:
            edge_cells += 1

    # If mostly edge cells, it's likely a partial rectangle
    if edge_cells >= len(obj.cells) * 0.5:
        return (r1, c1, r2, c2)

    return None


def learn_complete_boundaries(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn to complete partial rectangles/shapes"""
    for bg in [0, most_common_color(train_pairs[0][0])]:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            if grid_shape(out) != (h, w):
                ok = False
                break

            objs = detect_objects(inp, bg)
            result = [row[:] for row in inp]

            for obj in objs:
                bbox = _detect_partial_rectangle(obj, inp)
                if bbox:
                    r1, c1, r2, c2 = bbox
                    # Complete the rectangle outline
                    for r in range(r1, r2 + 1):
                        result[r][c1] = obj.color
                        result[r][c2] = obj.color
                    for c in range(c1, c2 + 1):
                        result[r1][c] = obj.color
                        result[r2][c] = obj.color

            if not grid_eq(result, out):
                ok = False
                break

        if ok:
            return {'type': 'complete_boundary', 'bg': bg}

    return None


def apply_complete_boundaries(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply boundary completion"""
    bg = params['bg']
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]

    for obj in objs:
        bbox = _detect_partial_rectangle(obj, inp)
        if bbox:
            r1, c1, r2, c2 = bbox
            for r in range(r1, r2 + 1):
                result[r][c1] = obj.color
                result[r][c2] = obj.color
            for c in range(c1, c2 + 1):
                result[r1][c] = obj.color
                result[r2][c] = obj.color

    return result


# ============================================================
# Move Objects to Edge Positions Based on Their Location
# ============================================================

def learn_move_objects_to_edge(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn to move objects to grid edges based on position

    Pattern: Objects move diagonally inward from corners (22208ba4 pattern)
    """
    for bg in [0, 7, most_common_color(train_pairs[0][0])]:  # Try common backgrounds
        # Try moving objects 1 step diagonally inward from corners
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            if grid_shape(out) != (h, w):
                ok = False
                break

            result = [[bg] * w for _ in range(h)]

            # Find all non-bg cells
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != bg:
                        color = inp[r][c]
                        # Determine movement based on position
                        # If on edge, move inward by 1
                        new_r, new_c = r, c

                        # On top edge
                        if r == 0:
                            new_r = 1
                        # On bottom edge
                        elif r == h - 1:
                            new_r = h - 2

                        # On left edge
                        if c == 0:
                            new_c = 1
                        # On right edge
                        elif c == w - 1:
                            new_c = w - 2

                        result[new_r][new_c] = color

            if not grid_eq(result, out):
                ok = False
                break

        if ok:
            return {'type': 'move_inward_from_edge', 'bg': bg}

    return None


def apply_move_objects_to_edge(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply move inward from edge transform"""
    bg = params['bg']
    h, w = grid_shape(inp)
    result = [[bg] * w for _ in range(h)]

    # Find all non-bg cells and move them inward from edges
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                color = inp[r][c]
                new_r, new_c = r, c

                # On top edge
                if r == 0:
                    new_r = 1
                # On bottom edge
                elif r == h - 1:
                    new_r = h - 2

                # On left edge
                if c == 0:
                    new_c = 1
                # On right edge
                elif c == w - 1:
                    new_c = w - 2

                result[new_r][new_c] = color

    return result


# ============================================================
# Main Learning Function
# ============================================================

def learn_conditional_object_transform(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn property→transform mapping for conditional object transforms

    Tries various conditional transform strategies in priority order.
    """
    strategies = [
        ('move_to_edge', learn_move_objects_to_edge, apply_move_objects_to_edge),
        ('fill_interior', learn_fill_object_interior, apply_fill_object_interior),
        ('line_extend', learn_line_extension_from_objects, apply_line_extension_from_objects),
        ('bg_to_color', learn_bg_to_nearby_color, apply_bg_to_nearby_color),
        ('complete_boundary', learn_complete_boundaries, apply_complete_boundaries),
    ]

    for name, learn_fn, apply_fn in strategies:
        try:
            rule = learn_fn(train_pairs)
            if rule is not None:
                # Verify it works on all training pairs
                ok = True
                for inp, out in train_pairs:
                    result = apply_fn(inp, rule)
                    if result is None or not grid_eq(result, out):
                        ok = False
                        break

                if ok:
                    rule['strategy'] = name
                    return rule
        except Exception:
            continue

    return None


def apply_conditional_object_transform(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply learned conditional object transform"""
    strategy = params.get('strategy', params.get('type'))

    if strategy == 'fill_interior':
        return apply_fill_object_interior(inp, params)
    elif strategy == 'line_extend' or params.get('type') == 'line_extend_objects':
        return apply_line_extension_from_objects(inp, params)
    elif strategy == 'bg_to_color':
        return apply_bg_to_nearby_color(inp, params)
    elif strategy == 'complete_boundary':
        return apply_complete_boundaries(inp, params)
    elif strategy == 'move_to_edge':
        return apply_move_objects_to_edge(inp, params)

    return None
