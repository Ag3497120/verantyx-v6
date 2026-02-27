"""
arc/pattern_tiling_solver.py — Pattern Tiling/Replication Solver

Pattern: Repeating objects or patterns across the grid
Examples: 045e512c, 0a938d79

Key insights:
- Detect repeating patterns in the output
- Patterns may be objects that get tiled horizontally/vertically
- Sparse input pixels define positions where pattern repeats
"""

from typing import List, Tuple, Optional
from arc.grid import Grid, grid_shape, most_common_color
from arc.objects import detect_objects


def learn_pattern_tiling_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[dict]:
    """
    Learn if the task involves tiling/repeating patterns.

    Returns rule params if pattern detected, None otherwise.
    """
    if not train_pairs:
        return None

    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        if grid_shape(out) != (h, w):
            return None  # Only same-size transforms

    # Analyze first training pair
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    bg = most_common_color(inp0)

    # Check if output has repeating patterns
    # Look for horizontal or vertical repetition

    # Try to detect if there's a small pattern that repeats
    # Check for vertical stripe patterns (columns that repeat)
    col_patterns = []
    for c in range(w):
        col = tuple(out0[r][c] for r in range(h))
        col_patterns.append(col)

    # Check if columns repeat with a period
    for period in range(1, w // 2 + 1):
        repeats = True
        for c in range(w):
            if col_patterns[c] != col_patterns[c % period]:
                repeats = False
                break
        if repeats and period < w:
            # Found vertical stripe pattern
            return {
                'type': 'vertical_stripe',
                'period': period
            }

    # Check for horizontal stripe patterns (rows that repeat)
    row_patterns = []
    for r in range(h):
        row = tuple(out0[r])
        row_patterns.append(row)

    for period in range(1, h // 2 + 1):
        repeats = True
        for r in range(h):
            if row_patterns[r] != row_patterns[r % period]:
                repeats = False
                break
        if repeats and period < h:
            # Found horizontal stripe pattern
            return {
                'type': 'horizontal_stripe',
                'period': period
            }

    # Check for object tiling
    # Detect if small objects in input become larger repeated patterns in output
    inp_objs = detect_objects(inp0)
    out_objs = detect_objects(out0)

    if len(inp_objs) > 0 and len(out_objs) > len(inp_objs):
        # More objects in output - could be tiling
        return {
            'type': 'object_tiling'
        }

    return None


def apply_vertical_stripe_tiling(inp: Grid, period: int) -> Grid:
    """
    Apply vertical stripe pattern tiling.

    Uses the first 'period' columns as a template and repeats them.
    """
    h, w = grid_shape(inp)
    out = [row[:] for row in inp]

    # Get the template from first period columns
    for r in range(h):
        for c in range(w):
            source_c = c % period
            out[r][c] = inp[r][source_c]

    return out


def apply_horizontal_stripe_tiling(inp: Grid, period: int) -> Grid:
    """
    Apply horizontal stripe pattern tiling.

    Uses the first 'period' rows as a template and repeats them.
    """
    h, w = grid_shape(inp)
    out = [row[:] for row in inp]

    # Get the template from first period rows
    for r in range(h):
        source_r = r % period
        out[r] = inp[source_r][:]

    return out


def apply_object_tiling(inp: Grid) -> Grid:
    """
    Detect objects and tile them across the grid.
    """
    h, w = grid_shape(inp)
    bg = most_common_color(inp)
    out = [[bg] * w for _ in range(h)]

    # Detect objects
    objs = detect_objects(inp)
    if not objs:
        return inp

    # For each object, try to tile it
    for obj in objs:
        if len(obj['pixels']) < 3:
            continue  # Skip very small objects

        # Get bounding box
        min_r = min(p[0] for p in obj['pixels'])
        max_r = max(p[0] for p in obj['pixels'])
        min_c = min(p[1] for p in obj['pixels'])
        max_c = max(p[1] for p in obj['pixels'])

        obj_h = max_r - min_r + 1
        obj_w = max_c - min_c + 1

        # Tile this object across the grid
        for base_r in range(0, h, obj_h):
            for base_c in range(0, w, obj_w):
                for pr, pc in obj['pixels']:
                    new_r = base_r + (pr - min_r)
                    new_c = base_c + (pc - min_c)
                    if 0 <= new_r < h and 0 <= new_c < w:
                        out[new_r][new_c] = inp[pr][pc]

    return out


def generate_pattern_tiling_pieces(train_pairs):
    """Generate CrossPiece candidates for pattern tiling."""
    from arc.cross_engine import CrossPiece

    rule = learn_pattern_tiling_rule(train_pairs)
    if rule is None:
        return []

    pieces = []

    if rule['type'] == 'vertical_stripe':
        pieces.append(CrossPiece(
            'vertical_stripe_tiling',
            lambda inp, p=rule['period']: apply_vertical_stripe_tiling(inp, p)
        ))
    elif rule['type'] == 'horizontal_stripe':
        pieces.append(CrossPiece(
            'horizontal_stripe_tiling',
            lambda inp, p=rule['period']: apply_horizontal_stripe_tiling(inp, p)
        ))
    elif rule['type'] == 'object_tiling':
        pieces.append(CrossPiece(
            'object_tiling',
            apply_object_tiling
        ))

    return pieces
