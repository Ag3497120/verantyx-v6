"""
arc/flood_fill.py — Flood Fill Region Primitive for ARC-AGI-2

Handles tasks where:
1. Background regions enclosed by walls/objects are filled with a specific color
2. Fill color is determined by nearby object color, enclosed object, or wall color
3. Multiple variants: fill all bg in enclosed areas, fill based on touching object, etc.

Common patterns:
- 0e671a1a: Connect colored dots with lines (wall-drawing)
- 1e32b0e9: Fill enclosed bg regions with wall color
- Others: Fill regions based on nearby/enclosed object properties
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
from collections import deque
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.objects import detect_objects, ArcObject


def learn_flood_fill_region(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn flood fill rule: fill enclosed bg regions with color from nearby/enclosed object.

    Detects several variants:
    1. Connect dots with rectangular frame (0e671a1a pattern)
    2. Fill enclosed bg regions with wall color (1e32b0e9pattern)
    3. Fill based on enclosed object color
    4. Fill all bg in enclosed areas with single color
    """

    # Try variant 1: Connect colored dots with rectangular frames
    rule = _learn_connect_dots_with_frames(train_pairs)
    if rule:
        return rule

    # Try variant 2: Fill enclosed regions with wall/boundary color
    rule = _learn_fill_enclosed_with_wall_color(train_pairs)
    if rule:
        return rule

    # Try variant 3: Fill enclosed regions based on nearby object color
    rule = _learn_fill_by_nearest_object(train_pairs)
    if rule:
        return rule

    # Try variant 4: Fill enclosed bg regions with fixed color
    rule = _learn_fill_enclosed_fixed_color(train_pairs)
    if rule:
        return rule

    return None


def apply_flood_fill_region(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply learned flood fill rule"""
    rule_type = params['type']

    if rule_type == 'connect_dots_frames':
        return _apply_connect_dots_with_frames(inp, params)
    elif rule_type == 'fill_enclosed_wall':
        return _apply_fill_enclosed_with_wall_color(inp, params)
    elif rule_type == 'fill_by_nearest':
        return _apply_fill_by_nearest_object(inp, params)
    elif rule_type == 'fill_enclosed_fixed':
        return _apply_fill_enclosed_fixed_color(inp, params)

    return None


# ============================================================
# Variant 1: Connect dots with rectangular frames (0e671a1a)
# ============================================================

def _learn_connect_dots_with_frames(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: draw rectangular frames connecting colored dots with a specific color.

    Pattern: colored dots (non-bg) get connected by rectangular borders.
    """
    for bg in [0]:
        for frame_color in range(1, 10):
            ok = True

            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                if grid_shape(out) != (h, w):
                    ok = False
                    break

                # Find all non-bg dots
                dots = []
                for r in range(h):
                    for c in range(w):
                        if inp[r][c] != bg:
                            dots.append((r, c, inp[r][c]))

                if len(dots) < 2:
                    ok = False
                    break

                # Build expected output: connect dots with frames
                result = [row[:] for row in inp]

                # For each pair of dots, draw rectangular frame
                for i in range(len(dots)):
                    for j in range(i + 1, len(dots)):
                        r1, c1, color1 = dots[i]
                        r2, c2, color2 = dots[j]

                        min_r, max_r = min(r1, r2), max(r1, r2)
                        min_c, max_c = min(c1, c2), max(c1, c2)

                        # Draw horizontal lines
                        for c in range(min_c, max_c + 1):
                            if result[min_r][c] == bg:
                                result[min_r][c] = frame_color
                            if result[max_r][c] == bg:
                                result[max_r][c] = frame_color

                        # Draw vertical lines
                        for r in range(min_r, max_r + 1):
                            if result[r][min_c] == bg:
                                result[r][min_c] = frame_color
                            if result[r][max_c] == bg:
                                result[r][max_c] = frame_color

                if not grid_eq(result, out):
                    ok = False
                    break

            if ok:
                return {'type': 'connect_dots_frames', 'bg': bg, 'frame_color': frame_color}

    return None


def _apply_connect_dots_with_frames(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply: connect dots with rectangular frames"""
    bg = params['bg']
    frame_color = params['frame_color']
    h, w = grid_shape(inp)

    dots = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                dots.append((r, c, inp[r][c]))

    result = [row[:] for row in inp]

    for i in range(len(dots)):
        for j in range(i + 1, len(dots)):
            r1, c1, _ = dots[i]
            r2, c2, _ = dots[j]

            min_r, max_r = min(r1, r2), max(r1, r2)
            min_c, max_c = min(c1, c2), max(c1, c2)

            for c in range(min_c, max_c + 1):
                if result[min_r][c] == bg:
                    result[min_r][c] = frame_color
                if result[max_r][c] == bg:
                    result[max_r][c] = frame_color

            for r in range(min_r, max_r + 1):
                if result[r][min_c] == bg:
                    result[r][min_c] = frame_color
                if result[r][max_c] == bg:
                    result[r][max_c] = frame_color

    return result


# ============================================================
# Variant 2: Fill enclosed regions with wall/boundary color
# ============================================================

def _learn_fill_enclosed_with_wall_color(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: fill enclosed bg regions with the color of the enclosing wall/grid lines.

    Pattern: Grid divided by walls/lines, fill bg cells in each region with wall color.
    """
    for bg in [0]:
        # Try to identify wall colors (colors that form grid lines)
        first_inp = train_pairs[0][0]
        h, w = grid_shape(first_inp)

        # Detect potential wall colors (appear in full rows/columns)
        wall_colors = set()
        for r in range(h):
            if len(set(first_inp[r])) == 1 and first_inp[r][0] != bg:
                wall_colors.add(first_inp[r][0])
        for c in range(w):
            col = [first_inp[r][c] for r in range(h)]
            if len(set(col)) == 1 and col[0] != bg:
                wall_colors.add(col[0])

        if not wall_colors:
            continue

        for wall_color in wall_colors:
            ok = True

            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                if grid_shape(out) != (h, w):
                    ok = False
                    break

                # Fill enclosed regions with wall color
                result = [row[:] for row in inp]

                # Find all bg regions and determine if they're enclosed
                visited = [[False] * w for _ in range(h)]

                for r in range(h):
                    for c in range(w):
                        if visited[r][c] or inp[r][c] != bg:
                            continue

                        # BFS to find region
                        region = []
                        queue = deque([(r, c)])
                        visited[r][c] = True
                        touches_border = False

                        while queue:
                            cr, cc = queue.popleft()
                            region.append((cr, cc))

                            if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                                touches_border = True

                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and inp[nr][nc] == bg:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))

                        # If region doesn't touch border, it's enclosed - fill it
                        if not touches_border:
                            for rr, cc in region:
                                result[rr][cc] = wall_color

                if not grid_eq(result, out):
                    ok = False
                    break

            if ok:
                return {'type': 'fill_enclosed_wall', 'bg': bg, 'wall_color': wall_color}

    return None


def _apply_fill_enclosed_with_wall_color(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply: fill enclosed bg regions with wall color"""
    bg = params['bg']
    wall_color = params['wall_color']
    h, w = grid_shape(inp)

    result = [row[:] for row in inp]
    visited = [[False] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if visited[r][c] or inp[r][c] != bg:
                continue

            region = []
            queue = deque([(r, c)])
            visited[r][c] = True
            touches_border = False

            while queue:
                cr, cc = queue.popleft()
                region.append((cr, cc))

                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and inp[nr][nc] == bg:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

            if not touches_border:
                for rr, cc in region:
                    result[rr][cc] = wall_color

    return result


# ============================================================
# Variant 3: Fill regions based on nearest/touching object color
# ============================================================

def _learn_fill_by_nearest_object(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: fill enclosed bg regions with color of nearest/touching object."""
    for bg in [0]:
        ok = True

        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            if grid_shape(out) != (h, w):
                ok = False
                break

            # Find all bg regions
            visited = [[False] * w for _ in range(h)]
            result = [row[:] for row in inp]

            for r in range(h):
                for c in range(w):
                    if visited[r][c] or inp[r][c] != bg:
                        continue

                    region = []
                    queue = deque([(r, c)])
                    visited[r][c] = True
                    touches_border = False
                    neighbor_colors = set()

                    while queue:
                        cr, cc = queue.popleft()
                        region.append((cr, cc))

                        if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                            touches_border = True

                        # Check neighbors for non-bg colors
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if inp[nr][nc] != bg:
                                    neighbor_colors.add(inp[nr][nc])
                                elif not visited[nr][nc]:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))

                    # If enclosed and has neighboring colors, fill with most common neighbor
                    if not touches_border and neighbor_colors:
                        fill_color = min(neighbor_colors)  # Pick consistently
                        for rr, cc in region:
                            result[rr][cc] = fill_color

            if not grid_eq(result, out):
                ok = False
                break

        if ok:
            return {'type': 'fill_by_nearest', 'bg': bg}

    return None


def _apply_fill_by_nearest_object(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply: fill regions by nearest object color"""
    bg = params['bg']
    h, w = grid_shape(inp)

    visited = [[False] * w for _ in range(h)]
    result = [row[:] for row in inp]

    for r in range(h):
        for c in range(w):
            if visited[r][c] or inp[r][c] != bg:
                continue

            region = []
            queue = deque([(r, c)])
            visited[r][c] = True
            touches_border = False
            neighbor_colors = set()

            while queue:
                cr, cc = queue.popleft()
                region.append((cr, cc))

                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if inp[nr][nc] != bg:
                            neighbor_colors.add(inp[nr][nc])
                        elif not visited[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

            if not touches_border and neighbor_colors:
                fill_color = min(neighbor_colors)
                for rr, cc in region:
                    result[rr][cc] = fill_color

    return result


# ============================================================
# Variant 4: Fill all enclosed bg regions with fixed color
# ============================================================

def _learn_fill_enclosed_fixed_color(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: fill all enclosed bg regions with a specific fixed color."""
    for bg in [0]:
        for fill_color in range(1, 10):
            ok = True

            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                if grid_shape(out) != (h, w):
                    ok = False
                    break

                visited = [[False] * w for _ in range(h)]
                result = [row[:] for row in inp]

                for r in range(h):
                    for c in range(w):
                        if visited[r][c] or inp[r][c] != bg:
                            continue

                        region = []
                        queue = deque([(r, c)])
                        visited[r][c] = True
                        touches_border = False

                        while queue:
                            cr, cc = queue.popleft()
                            region.append((cr, cc))

                            if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                                touches_border = True

                            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and inp[nr][nc] == bg:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))

                        if not touches_border:
                            for rr, cc in region:
                                result[rr][cc] = fill_color

                if not grid_eq(result, out):
                    ok = False
                    break

            if ok:
                return {'type': 'fill_enclosed_fixed', 'bg': bg, 'fill_color': fill_color}

    return None


def _apply_fill_enclosed_fixed_color(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply: fill enclosed regions with fixed color"""
    bg = params['bg']
    fill_color = params['fill_color']
    h, w = grid_shape(inp)

    visited = [[False] * w for _ in range(h)]
    result = [row[:] for row in inp]

    for r in range(h):
        for c in range(w):
            if visited[r][c] or inp[r][c] != bg:
                continue

            region = []
            queue = deque([(r, c)])
            visited[r][c] = True
            touches_border = False

            while queue:
                cr, cc = queue.popleft()
                region.append((cr, cc))

                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and inp[nr][nc] == bg:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

            if not touches_border:
                for rr, cc in region:
                    result[rr][cc] = fill_color

    return result
