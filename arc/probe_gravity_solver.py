"""
arc/probe_gravity_solver.py — Probe-based gravity solver

Strategies:
1. Corner stacking: objects sorted by distance to corner, stacked with shared corners
2. Uniform gravity: all objects slide in one direction until wall/collision
"""

import numpy as np
from typing import List, Tuple, Optional
from arc.cross3d_probe import measure_objects
from arc.cross_engine import CrossPiece

Grid = List[List[int]]


def _place_mask(r, c, mask, color, result, placed):
    mh, mw = mask.shape
    for mr in range(mh):
        for mc in range(mw):
            if mask[mr, mc]:
                result[r + mr, c + mc] = color
                placed[r + mr, c + mc] = True


def _obj_mask(obj):
    r_min, c_min, r_max, c_max = obj['bbox']
    h = r_max - r_min + 1
    w = c_max - c_min + 1
    mask = np.zeros((h, w), dtype=bool)
    for cr, cc in obj['cells']:
        mask[cr - r_min, cc - c_min] = True
    return mask, h, w


def apply_corner_stack(inp_grid, corner, bg=0):
    inp = np.array(inp_grid, dtype=int)
    H, W = inp.shape
    objs = measure_objects(inp, bg=bg)
    if not objs:
        return inp.tolist()

    target_r = 0 if corner[0] == 0 else H - 1
    target_c = 0 if corner[1] == 0 else W - 1
    objs.sort(key=lambda o: abs(o['center'][0] - target_r) + abs(o['center'][1] - target_c))

    result = np.full_like(inp, bg)
    placed = np.zeros_like(inp, dtype=bool)

    stack_r, stack_c = None, None

    for idx, obj in enumerate(objs):
        mask, obj_h, obj_w = _obj_mask(obj)

        if idx == 0:
            if corner == (0, 0):
                cur_r, cur_c = 0, 0
            elif corner == (0, 1):
                cur_r, cur_c = 0, W - obj_w
            elif corner == (1, 0):
                cur_r, cur_c = H - obj_h, 0
            else:
                cur_r, cur_c = H - obj_h, W - obj_w
        else:
            if corner == (0, 0):
                cur_r, cur_c = stack_r, stack_c
            elif corner == (0, 1):
                cur_r, cur_c = stack_r, stack_c - obj_w + 1
            elif corner == (1, 0):
                cur_r, cur_c = stack_r - obj_h + 1, stack_c
            else:
                cur_r, cur_c = stack_r - obj_h + 1, stack_c - obj_w + 1

        if cur_r < 0 or cur_c < 0 or cur_r + obj_h > H or cur_c + obj_w > W:
            return None

        _place_mask(cur_r, cur_c, mask, obj['color'], result, placed)

        if corner == (0, 0):
            stack_r = cur_r + obj_h - 1
            stack_c = cur_c + obj_w - 1
        elif corner == (0, 1):
            stack_r = cur_r + obj_h - 1
            stack_c = cur_c
        elif corner == (1, 0):
            stack_r = cur_r
            stack_c = cur_c + obj_w - 1
        else:
            stack_r = cur_r
            stack_c = cur_c

    return result.tolist()


def apply_uniform_gravity(inp_grid, direction, bg=0):
    inp = np.array(inp_grid, dtype=int)
    H, W = inp.shape
    dr, dc = direction
    objs = measure_objects(inp, bg=bg)

    if dr > 0: objs.sort(key=lambda o: -o['bbox'][2])
    elif dr < 0: objs.sort(key=lambda o: o['bbox'][0])
    elif dc > 0: objs.sort(key=lambda o: -o['bbox'][3])
    else: objs.sort(key=lambda o: o['bbox'][1])

    result = np.full_like(inp, bg)
    placed = np.zeros_like(inp, dtype=bool)

    for obj in objs:
        mask, obj_h, obj_w = _obj_mask(obj)
        cur_r, cur_c = obj['bbox'][0], obj['bbox'][1]

        while True:
            nr, nc = cur_r + dr, cur_c + dc
            if nr < 0 or nc < 0 or nr + obj_h > H or nc + obj_w > W:
                break
            ok = True
            for mr in range(obj_h):
                for mc in range(obj_w):
                    if mask[mr, mc] and placed[nr + mr, nc + mc]:
                        ok = False
                        break
                if not ok: break
            if not ok: break
            cur_r, cur_c = nr, nc

        _place_mask(cur_r, cur_c, mask, obj['color'], result, placed)

    return result.tolist()


def _verify_all(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None:
            return False
        if np.array_equal(np.array(pred), np.array(out)) is False:
            return False
    return True


def generate_probe_gravity_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    pieces = []
    if not train_pairs:
        return pieces

    bg = 0

    # Try corner stacking
    for corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        fn = lambda inp, c=corner: apply_corner_stack(inp, c, bg)
        if _verify_all(fn, train_pairs):
            pieces.append(CrossPiece(
                name=f"cross3d:corner_stack_{corner}",
                apply_fn=fn,
            ))
            return pieces  # one is enough

    # Try uniform gravity
    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        fn = lambda inp, d=direction: apply_uniform_gravity(inp, d, bg)
        if _verify_all(fn, train_pairs):
            pieces.append(CrossPiece(
                name=f"cross3d:uniform_gravity_{direction}",
                apply_fn=fn,
            ))
            return pieces

    return pieces
