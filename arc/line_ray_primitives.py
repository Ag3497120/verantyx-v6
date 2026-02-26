"""
arc/line_ray_primitives.py — Line and ray extension primitives

Handles tasks where lines/rays are drawn from colored cells/objects.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.objects import detect_objects


def learn_line_ray_from_objects(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: draw lines/rays from colored cells along rows/cols until hitting boundary or other color.

    Variants:
    - Ray direction: horizontal, vertical, or both (cross)
    - Ray behavior: until boundary, until hit non-bg, bidirectional
    - Ray color: same as source, or fixed color
    """
    if not train_pairs:
        return None

    # Must be same size
    inp0, out0 = train_pairs[0]
    if grid_shape(inp0) != grid_shape(out0):
        return None

    bg = most_common_color(inp0)

    # Try different strategies
    strategies = [
        ('cross_ray_same_color', _try_cross_ray_same),
        ('cross_ray_until_hit', _try_cross_ray_until_hit),
        ('h_ray_same', _try_h_ray_same),
        ('v_ray_same', _try_v_ray_same),
        ('cross_from_dots', _try_cross_from_dots),
    ]

    for strategy_name, strategy_fn in strategies:
        rule = strategy_fn(train_pairs, bg)
        if rule:
            rule['type'] = strategy_name
            rule['bg'] = bg
            # Verify on all pairs
            ok = True
            for inp, out in train_pairs:
                result = apply_line_ray_from_objects(inp, rule)
                if result is None or not grid_eq(result, out):
                    ok = False
                    break
            if ok:
                return rule

    return None


def _try_cross_ray_same(train_pairs, bg):
    """Try: from each non-bg cell, draw cross (H+V lines) in same color until boundary"""
    for inp, out in train_pairs:
        ai = np.array(inp)
        ao = np.array(out)
        h, w = ai.shape

        expected = ai.copy()

        # From each non-bg cell, draw rays
        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]
                    # Draw horizontal line
                    for c2 in range(w):
                        if expected[r, c2] == bg:
                            expected[r, c2] = color
                    # Draw vertical line
                    for r2 in range(h):
                        if expected[r2, c] == bg:
                            expected[r2, c] = color

        if not np.array_equal(expected, ao):
            return None

    return {}


def _try_cross_ray_until_hit(train_pairs, bg):
    """Try: from each non-bg cell, draw cross until hitting another non-bg cell"""
    for inp, out in train_pairs:
        ai = np.array(inp)
        ao = np.array(out)
        h, w = ai.shape

        expected = ai.copy()

        # From each non-bg cell, draw rays until hit
        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]

                    # Draw right
                    for c2 in range(c + 1, w):
                        if ai[r, c2] != bg:
                            break
                        expected[r, c2] = color

                    # Draw left
                    for c2 in range(c - 1, -1, -1):
                        if ai[r, c2] != bg:
                            break
                        expected[r, c2] = color

                    # Draw down
                    for r2 in range(r + 1, h):
                        if ai[r2, c] != bg:
                            break
                        expected[r2, c] = color

                    # Draw up
                    for r2 in range(r - 1, -1, -1):
                        if ai[r2, c] != bg:
                            break
                        expected[r2, c] = color

        if not np.array_equal(expected, ao):
            return None

    return {'mode': 'until_hit'}


def _try_h_ray_same(train_pairs, bg):
    """Try: horizontal rays only"""
    for inp, out in train_pairs:
        ai = np.array(inp)
        ao = np.array(out)
        h, w = ai.shape

        expected = ai.copy()

        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]
                    for c2 in range(w):
                        if expected[r, c2] == bg:
                            expected[r, c2] = color

        if not np.array_equal(expected, ao):
            return None

    return {'direction': 'horizontal'}


def _try_v_ray_same(train_pairs, bg):
    """Try: vertical rays only"""
    for inp, out in train_pairs:
        ai = np.array(inp)
        ao = np.array(out)
        h, w = ai.shape

        expected = ai.copy()

        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]
                    for r2 in range(h):
                        if expected[r2, c] == bg:
                            expected[r2, c] = color

        if not np.array_equal(expected, ao):
            return None

    return {'direction': 'vertical'}


def _try_cross_from_dots(train_pairs, bg):
    """Try: cross rays from single-cell objects (dots) only"""
    for inp, out in train_pairs:
        objs = detect_objects(inp, bg)
        dots = [o for o in objs if o.size == 1]

        if not dots:
            return None

        ai = np.array(inp)
        ao = np.array(out)
        h, w = ai.shape

        expected = ai.copy()

        for dot in dots:
            r, c = dot.cells[0]
            color = dot.color

            # Draw cross from this dot
            for c2 in range(w):
                if expected[r, c2] == bg:
                    expected[r, c2] = color
            for r2 in range(h):
                if expected[r2, c] == bg:
                    expected[r2, c] = color

        if not np.array_equal(expected, ao):
            return None

    return {'mode': 'from_dots'}


def apply_line_ray_from_objects(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply line/ray extension"""
    strategy = params['type']
    bg = params['bg']

    ai = np.array(inp)
    h, w = ai.shape
    result = ai.copy()

    if strategy == 'cross_ray_same_color':
        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]
                    for c2 in range(w):
                        if result[r, c2] == bg:
                            result[r, c2] = color
                    for r2 in range(h):
                        if result[r2, c] == bg:
                            result[r2, c] = color

    elif strategy == 'cross_ray_until_hit':
        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]

                    for c2 in range(c + 1, w):
                        if ai[r, c2] != bg:
                            break
                        result[r, c2] = color

                    for c2 in range(c - 1, -1, -1):
                        if ai[r, c2] != bg:
                            break
                        result[r, c2] = color

                    for r2 in range(r + 1, h):
                        if ai[r2, c] != bg:
                            break
                        result[r2, c] = color

                    for r2 in range(r - 1, -1, -1):
                        if ai[r2, c] != bg:
                            break
                        result[r2, c] = color

    elif strategy == 'h_ray_same':
        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]
                    for c2 in range(w):
                        if result[r, c2] == bg:
                            result[r, c2] = color

    elif strategy == 'v_ray_same':
        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg:
                    color = ai[r, c]
                    for r2 in range(h):
                        if result[r2, c] == bg:
                            result[r2, c] = color

    elif strategy == 'cross_from_dots':
        objs = detect_objects(inp, bg)
        dots = [o for o in objs if o.size == 1]

        for dot in dots:
            r, c = dot.cells[0]
            color = dot.color

            for c2 in range(w):
                if result[r, c2] == bg:
                    result[r, c2] = color
            for r2 in range(h):
                if result[r2, c] == bg:
                    result[r2, c] = color

    return result.tolist()


def learn_fill_object_interior(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: fill interior (enclosed) regions of objects"""
    if not train_pairs:
        return None

    inp0, out0 = train_pairs[0]
    if grid_shape(inp0) != grid_shape(out0):
        return None

    bg = most_common_color(inp0)

    # Try different fill strategies
    # Strategy 1: fill all enclosed bg regions
    ok = True
    for inp, out in train_pairs:
        result = _fill_enclosed_regions(inp, bg)
        if result is None or not grid_eq(result, out):
            ok = False
            break

    if ok:
        return {'type': 'fill_enclosed', 'bg': bg}

    # Strategy 2: fill interiors of objects (per-object)
    ok = True
    for inp, out in train_pairs:
        result = _fill_object_bboxes(inp, bg)
        if result is None or not grid_eq(result, out):
            ok = False
            break

    if ok:
        return {'type': 'fill_object_bbox', 'bg': bg}

    return None


def _fill_enclosed_regions(inp: Grid, bg: int) -> Grid:
    """Fill all bg regions that are not connected to border"""
    from scipy import ndimage

    ai = np.array(inp)
    h, w = ai.shape

    # Find all bg regions
    bg_mask = ai == bg
    labeled, n = ndimage.label(bg_mask)

    # Find which labels touch the border
    border_labels = set()
    for r in range(h):
        if bg_mask[r, 0]:
            border_labels.add(labeled[r, 0])
        if bg_mask[r, w - 1]:
            border_labels.add(labeled[r, w - 1])
    for c in range(w):
        if bg_mask[0, c]:
            border_labels.add(labeled[0, c])
        if bg_mask[h - 1, c]:
            border_labels.add(labeled[h - 1, c])

    # Fill enclosed regions (not touching border)
    result = ai.copy()
    for r in range(h):
        for c in range(w):
            if bg_mask[r, c] and labeled[r, c] not in border_labels:
                # Fill with color of surrounding object
                # Find nearest non-bg color
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and ai[nr, nc] != bg:
                        result[r, c] = ai[nr, nc]
                        break

    return result.tolist()


def _fill_object_bboxes(inp: Grid, bg: int) -> Grid:
    """Fill bounding boxes of all objects"""
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]

    for obj in objs:
        r1, c1, r2, c2 = obj.bbox
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                result[r][c] = obj.color

    return result


def apply_fill_object_interior(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply interior fill"""
    fill_type = params['type']
    bg = params['bg']

    if fill_type == 'fill_enclosed':
        return _fill_enclosed_regions(inp, bg)
    elif fill_type == 'fill_object_bbox':
        return _fill_object_bboxes(inp, bg)

    return None
