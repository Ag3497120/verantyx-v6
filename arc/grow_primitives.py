"""
arc/grow_primitives.py — Primitives for growing/scaling transformations

Handles tasks where output is NxN times larger than input.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def learn_grow_via_self_stamp(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: output = HxW grid of input-sized blocks, where each block at (r,c)
    is the entire input if a trigger condition is met, else background.

    Pattern: input is NxM, output is (N*K)x(M*K), where K is determined per task.
    Each KxK block corresponds to one input cell.
    If input[r,c] meets condition, block[r,c] = entire input, else = bg.
    """
    if not train_pairs:
        return None

    inp0 = np.array(train_pairs[0][0])
    out0 = np.array(train_pairs[0][1])
    ih, iw = inp0.shape
    oh, ow = out0.shape

    # Quick checks
    if ih > 10 or iw > 10:
        return None  # Too large for self-stamp

    if oh % ih != 0 or ow % iw != 0:
        return None

    scale_h = oh // ih
    scale_w = ow // iw

    if scale_h != scale_w or scale_h != ih or scale_w != iw:
        # Must be h*h x w*w
        return None

    K = ih  # Block size

    # Try different trigger conditions
    bg = most_common_color(out0)

    # Strategy 1: trigger_color - if input[r,c] == trigger_color, stamp input
    all_colors = set(int(v) for v in inp0.flatten())

    for trigger_color in all_colors:
        if trigger_color == bg:
            continue

        ok = True
        for inp, out in train_pairs:
            ai = np.array(inp)
            ao = np.array(out)
            _h, _w = ai.shape

            if ao.shape != (_h * _h, _w * _w):
                ok = False
                break

            # Build expected output
            expected = np.full((_h * _h, _w * _w), bg, dtype=int)
            for r in range(_h):
                for c in range(_w):
                    if ai[r, c] == trigger_color:
                        expected[r * _h:(r + 1) * _h, c * _w:(c + 1) * _w] = ai

            if not np.array_equal(expected, ao):
                ok = False
                break

        if ok:
            return {
                'type': 'grow_self_stamp',
                'trigger_color': int(trigger_color),
                'bg': int(bg)
            }

    # Strategy 2: trigger_nonzero - if input[r,c] != bg_inp, stamp input
    bg_inp = most_common_color(train_pairs[0][0])
    ok = True
    for inp, out in train_pairs:
        ai = np.array(inp)
        ao = np.array(out)
        _h, _w = ai.shape

        if ao.shape != (_h * _h, _w * _w):
            ok = False
            break

        expected = np.full((_h * _h, _w * _w), bg, dtype=int)
        for r in range(_h):
            for c in range(_w):
                if ai[r, c] != bg_inp:
                    expected[r * _h:(r + 1) * _h, c * _w:(c + 1) * _w] = ai

        if not np.array_equal(expected, ao):
            ok = False
            break

    if ok:
        return {
            'type': 'grow_self_stamp',
            'trigger': 'nonzero',
            'bg_inp': int(bg_inp),
            'bg': int(bg)
        }

    # Strategy 3: trigger_most_common - if input[r,c] == most_common_color(input), stamp input
    # Output bg is always 0 for this pattern
    ok = True
    for inp, out in train_pairs:
        ai = np.array(inp)
        ao = np.array(out)
        _h, _w = ai.shape

        if ao.shape != (_h * _h, _w * _w):
            ok = False
            break

        most_common = most_common_color(inp)
        expected = np.full((_h * _h, _w * _w), 0, dtype=int)  # Always 0 bg
        for r in range(_h):
            for c in range(_w):
                if ai[r, c] == most_common:
                    expected[r * _h:(r + 1) * _h, c * _w:(c + 1) * _w] = ai

        if not np.array_equal(expected, ao):
            ok = False
            break

    if ok:
        return {
            'type': 'grow_self_stamp',
            'trigger': 'most_common',
            'bg': 0  # Always 0 bg
        }

    return None


def apply_grow_via_self_stamp(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply self-stamp growth"""
    ai = np.array(inp)
    h, w = ai.shape

    bg = params.get('bg', 0)
    result = np.full((h * h, w * w), bg, dtype=int)

    if 'trigger_color' in params:
        trigger_color = params['trigger_color']
        for r in range(h):
            for c in range(w):
                if ai[r, c] == trigger_color:
                    result[r * h:(r + 1) * h, c * w:(c + 1) * w] = ai

    elif params.get('trigger') == 'nonzero':
        bg_inp = params['bg_inp']
        for r in range(h):
            for c in range(w):
                if ai[r, c] != bg_inp:
                    result[r * h:(r + 1) * h, c * w:(c + 1) * w] = ai

    elif params.get('trigger') == 'most_common':
        from arc.grid import most_common_color as mcc
        most_common = mcc(inp)
        for r in range(h):
            for c in range(w):
                if ai[r, c] == most_common:
                    result[r * h:(r + 1) * h, c * w:(c + 1) * w] = ai

    return result.tolist()


def learn_grow_color_template(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: each input cell → KxK block with per-cell-position pattern.

    The pattern is NOT color-specific but position-specific within the training examples.
    We learn what each input cell position maps to in the output.
    """
    if not train_pairs:
        return None

    inp0 = np.array(train_pairs[0][0])
    out0 = np.array(train_pairs[0][1])
    ih, iw = inp0.shape
    oh, ow = out0.shape

    if oh % ih != 0 or ow % iw != 0:
        return None

    kh = oh // ih
    kw = ow // iw

    if kh != kw or kh <= 1:
        return None

    K = kh

    # For first pair, learn the template mapping for EACH position
    # template[r][c] = KxK pattern that input[r,c] expands to
    template = {}

    for r in range(ih):
        for c in range(iw):
            block = out0[r * K:(r + 1) * K, c * K:(c + 1) * K]
            template[(r, c)] = block.copy()

    # Verify this template works for all pairs
    ok = True
    for inp, out in train_pairs[1:]:
        ai = np.array(inp)
        ao = np.array(out)
        _ih, _iw = ai.shape
        _oh, _ow = ao.shape

        if _oh != _ih * K or _ow != _iw * K:
            ok = False
            break

        # Check if the template matches
        for r in range(_ih):
            for c in range(_iw):
                if (r, c) in template:
                    expected_block = template[(r, c)]
                    actual_block = ao[r * K:(r + 1) * K, c * K:(c + 1) * K]

                    if not np.array_equal(expected_block, actual_block):
                        ok = False
                        break
            if not ok:
                break

        if not ok:
            break

    if ok and template:
        return {
            'type': 'grow_position_template',
            'K': K,
            'template': {f"{r},{c}": block.tolist() for (r, c), block in template.items()}
        }

    return None


def apply_grow_color_template(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply position-based template growth"""
    K = params['K']
    template = params['template']

    ai = np.array(inp)
    ih, iw = ai.shape
    result = np.zeros((ih * K, iw * K), dtype=int)

    for r in range(ih):
        for c in range(iw):
            key = f"{r},{c}"
            if key in template:
                block = np.array(template[key])
                result[r * K:(r + 1) * K, c * K:(c + 1) * K] = block

    return result.tolist()


def learn_grow_fixed_color_template(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn: each color → fixed KxK block pattern (color-specific, not position-specific)"""
    if not train_pairs:
        return None

    inp0 = np.array(train_pairs[0][0])
    out0 = np.array(train_pairs[0][1])
    ih, iw = inp0.shape
    oh, ow = out0.shape

    if oh % ih != 0 or ow % iw != 0:
        return None

    kh = oh // ih
    kw = ow // iw

    if kh != kw or kh <= 1:
        return None

    K = kh

    # Learn color → block mapping from first pair
    color_template = {}

    for r in range(ih):
        for c in range(iw):
            color = int(inp0[r, c])
            block = out0[r * K:(r + 1) * K, c * K:(c + 1) * K]
            block_tuple = tuple(block.flatten().tolist())

            if color in color_template:
                if color_template[color] != block_tuple:
                    # Inconsistent mapping for this color
                    return None
            else:
                color_template[color] = block_tuple

    # Verify across all pairs
    ok = True
    for inp, out in train_pairs[1:]:
        ai = np.array(inp)
        ao = np.array(out)
        _ih, _iw = ai.shape

        if ao.shape != (_ih * K, _iw * K):
            ok = False
            break

        for r in range(_ih):
            for c in range(_iw):
                color = int(ai[r, c])
                if color not in color_template:
                    ok = False
                    break

                expected_block = np.array(color_template[color]).reshape(K, K)
                actual_block = ao[r * K:(r + 1) * K, c * K:(c + 1) * K]

                if not np.array_equal(expected_block, actual_block):
                    ok = False
                    break
            if not ok:
                break

    if ok and color_template:
        return {
            'type': 'grow_color_template',
            'K': K,
            'color_template': {int(c): list(p) for c, p in color_template.items()}
        }

    return None


def apply_grow_fixed_color_template(inp: Grid, params: Dict) -> Optional[Grid]:
    """Apply color-based template growth"""
    K = params['K']
    color_template = params['color_template']

    ai = np.array(inp)
    ih, iw = ai.shape
    result = np.zeros((ih * K, iw * K), dtype=int)

    for r in range(ih):
        for c in range(iw):
            color = int(ai[r, c])
            if color not in color_template:
                return None

            block = np.array(color_template[color]).reshape(K, K)
            result[r * K:(r + 1) * K, c * K:(c + 1) * K] = block

    return result.tolist()
