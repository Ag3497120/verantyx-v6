"""
arc/rect_boundary_solver.py — Rectangular Boundary Solver

Pattern: Detect colored corner markers and draw rectangles connecting them
Example task: 0e671a1a

Key insight:
- Find colored non-background pixels that could be corners
- Try to form rectangles from pairs/groups of corners
- Draw boundary lines (in a connecting color) between corners
"""

from typing import List, Tuple, Optional
from arc.grid import Grid, grid_shape, most_common_color


def learn_rect_boundary_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[dict]:
    """
    Learn if the task draws rectangular boundaries between colored markers.

    Returns rule params if pattern detected, None otherwise.
    """
    if not train_pairs:
        return None

    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        if grid_shape(out) != (h, w):
            return None  # Only same-size transforms

    # Check if output has new lines connecting existing colored pixels
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)

    # Find all non-bg pixels in input
    inp_colored = set()
    for r in range(len(inp0)):
        for c in range(len(inp0[0])):
            if inp0[r][c] != bg:
                inp_colored.add((r, c))

    if len(inp_colored) < 2:
        return None

    # Check if output has new pixels forming lines
    new_pixels = []
    new_colors = set()
    for r in range(len(out0)):
        for c in range(len(out0[0])):
            if out0[r][c] != bg and (r, c) not in inp_colored:
                new_pixels.append((r, c, out0[r][c]))
                new_colors.add(out0[r][c])

    if not new_pixels:
        return None

    # The new color should be consistent (single color for boundaries)
    if len(new_colors) != 1:
        return None

    boundary_color = new_colors.pop()

    # Verify this works across all training pairs
    for inp, out in train_pairs:
        bg = most_common_color(inp)
        inp_colored = set()
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] != bg:
                    inp_colored.add((r, c))

        # Check new pixels in output
        has_new = False
        for r in range(len(out)):
            for c in range(len(out[0])):
                if out[r][c] != bg and (r, c) not in inp_colored:
                    if out[r][c] != boundary_color:
                        return None  # Boundary color should be consistent
                    has_new = True

        if not has_new:
            return None

    return {'boundary_color': boundary_color}


def apply_rect_boundary_rule(inp: Grid, boundary_color: int) -> Grid:
    """
    Apply rectangular boundary drawing:
    1. Find all colored non-bg pixels (potential corners)
    2. Form rectangles from subsets of corners
    3. Draw boundaries connecting corners
    """
    h, w = grid_shape(inp)
    bg = most_common_color(inp)

    # Copy input
    out = [row[:] for row in inp]

    # Find all colored pixels
    colored = []
    for r in range(h):
        for c in range(w):
            if inp[r][c] != bg:
                colored.append((r, c, inp[r][c]))

    if len(colored) < 2:
        return out

    # Strategy: Find rectangular regions formed by colored pixels
    # and draw boundary lines connecting them

    # Try finding axis-aligned rectangles
    positions = [(r, c) for r, c, _ in colored]

    # For each pair of points, check if they can form a rectangle
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            r1, c1 = positions[i]
            r2, c2 = positions[j]

            # Check if they're on different rows and columns
            if r1 != r2 and c1 != c2:
                # Draw horizontal and vertical lines connecting them
                # Horizontal lines
                min_c, max_c = min(c1, c2), max(c1, c2)
                for c in range(min_c, max_c + 1):
                    if out[r1][c] == bg:
                        out[r1][c] = boundary_color
                    if out[r2][c] == bg:
                        out[r2][c] = boundary_color

                # Vertical lines
                min_r, max_r = min(r1, r2), max(r1, r2)
                for r in range(min_r, max_r + 1):
                    if out[r][c1] == bg:
                        out[r][c1] = boundary_color
                    if out[r][c2] == bg:
                        out[r][c2] = boundary_color

    return out


def generate_rect_boundary_pieces(train_pairs):
    """Generate CrossPiece candidates for rectangular boundary drawing."""
    from arc.cross_engine import CrossPiece

    rule = learn_rect_boundary_rule(train_pairs)
    if rule is None:
        return []

    return [CrossPiece(
        'rect_boundary',
        lambda inp, bc=rule['boundary_color']: apply_rect_boundary_rule(inp, bc)
    )]
