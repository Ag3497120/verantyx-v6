"""
arc/rotating_cross.py — Rotating Cross Engine

Core idea (kofdai design):
After detecting objects, rotate the cross structure itself like celestial bodies.
Every CrossPiece is automatically tried in all 8 orientations (4 rotations × 2 flips).

This means: if a rule works for "stamp pattern to the right", it automatically
also works for "stamp pattern to the left/up/down" and all diagonal variants.

Object detection uses concave/convex boundary mapping from cross3d probe.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from collections import Counter
from scipy.ndimage import label as connected_components
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _g(g): return np.array(g, dtype=int)
def _l(a): return a.tolist()
def _bg(g): return int(Counter(g.flatten()).most_common(1)[0][0])


# ============================================================
# 8 ORIENTATION TRANSFORMS
# ============================================================

def _orientations():
    """Return 8 (forward, inverse) transform pairs for grid rotation"""
    return [
        ('0',     lambda g: g,                    lambda g: g),
        ('90',    lambda g: np.rot90(g, 1),        lambda g: np.rot90(g, 3)),
        ('180',   lambda g: np.rot90(g, 2),        lambda g: np.rot90(g, 2)),
        ('270',   lambda g: np.rot90(g, 3),        lambda g: np.rot90(g, 1)),
        ('fh',    lambda g: g[:, ::-1],            lambda g: g[:, ::-1]),
        ('fv',    lambda g: g[::-1, :],            lambda g: g[::-1, :]),
        ('fh90',  lambda g: np.rot90(g[:, ::-1], 1), lambda g: np.rot90(g, 3)[:, ::-1]),
        ('fv90',  lambda g: np.rot90(g[::-1, :], 1), lambda g: np.rot90(g, 3)[::-1, :]),
    ]


def rotate_piece(piece_fn, orient_name, fwd, inv):
    """Wrap a piece function to work in a rotated frame.
    
    Logic:
    1. Rotate input by `fwd` (move into rotated frame)
    2. Apply piece_fn in that frame
    3. Rotate result back by `inv`
    """
    def rotated_fn(inp_g):
        inp = _g(inp_g)
        rotated_inp = fwd(inp)
        result = piece_fn(_l(rotated_inp))
        if result is None:
            return None
        return _l(inv(_g(result)))
    return rotated_fn


# ============================================================
# OBJECT DETECTION & PER-OBJECT APPLICATION
# ============================================================

def detect_objects(grid, bg):
    """Detect connected components as objects with bbox and properties"""
    mask = (grid != bg).astype(int)
    labeled, n = connected_components(mask)
    
    objects = []
    for i in range(1, n + 1):
        cells = np.where(labeled == i)
        rows, cols = cells
        r1, r2 = int(rows.min()), int(rows.max())
        c1, c2 = int(cols.min()), int(cols.max())
        
        crop = grid[r1:r2+1, c1:c2+1].copy()
        obj_mask = labeled[r1:r2+1, c1:c2+1] == i
        
        colors = Counter(int(grid[r, c]) for r, c in zip(rows, cols))
        
        objects.append({
            'id': i,
            'bbox': (r1, c1, r2, c2),
            'crop': crop,
            'mask': obj_mask,
            'full_mask': (labeled == i),
            'area': len(rows),
            'colors': colors,
            'main_color': colors.most_common(1)[0][0],
            'n_colors': len(colors),
            'center': ((r1 + r2) / 2, (c1 + c2) / 2),
            'shape': (r2 - r1 + 1, c2 - c1 + 1),
        })
    
    return objects


def _object_distance(o1, o2):
    """Manhattan distance between object centers"""
    return abs(o1['center'][0] - o2['center'][0]) + abs(o1['center'][1] - o2['center'][1])


# ============================================================
# MOVEMENT PRIMITIVES (applied per-object)
# ============================================================

def move_to_wall(grid, obj, bg, direction):
    """Move object to the wall in given direction"""
    H, W = grid.shape
    r1, c1, r2, c2 = obj['bbox']
    oh, ow = r2 - r1 + 1, c2 - c1 + 1
    
    if direction == 'up':
        new_r1 = 0
        new_c1 = c1
    elif direction == 'down':
        new_r1 = H - oh
        new_c1 = c1
    elif direction == 'left':
        new_r1 = r1
        new_c1 = 0
    elif direction == 'right':
        new_r1 = r1
        new_c1 = W - ow
    else:
        return None
    
    # Place object
    out = grid.copy()
    # Erase old position
    out[obj['full_mask']] = bg
    # Place at new position
    for dr in range(oh):
        for dc in range(ow):
            if obj['mask'][dr, dc]:
                nr, nc = new_r1 + dr, new_c1 + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = obj['crop'][dr, dc]
    
    return out


def move_toward_nearest(grid, obj, objects, bg):
    """Move object toward its nearest neighbor until touching"""
    if len(objects) < 2:
        return None
    
    H, W = grid.shape
    nearest = min([o for o in objects if o['id'] != obj['id']],
                  key=lambda o: _object_distance(obj, o))
    
    # Direction from obj to nearest
    dr = nearest['center'][0] - obj['center'][0]
    dc = nearest['center'][1] - obj['center'][1]
    
    # Move one step at a time until adjacent
    r1, c1, r2, c2 = obj['bbox']
    oh, ow = r2 - r1 + 1, c2 - c1 + 1
    
    step_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
    step_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
    
    # Slide until touching
    new_r1, new_c1 = r1, c1
    for _ in range(max(H, W)):
        next_r1 = new_r1 + step_r
        next_c1 = new_c1 + step_c
        
        # Check bounds
        if next_r1 < 0 or next_r1 + oh > H or next_c1 < 0 or next_c1 + ow > W:
            break
        
        # Check collision with nearest
        collision = False
        for ddr in range(oh):
            for ddc in range(ow):
                if obj['mask'][ddr, ddc]:
                    nr, nc = next_r1 + ddr, next_c1 + ddc
                    if nearest['full_mask'][nr, nc]:
                        collision = True
                        break
            if collision:
                break
        
        if collision:
            break
        
        new_r1, new_c1 = next_r1, next_c1
    
    if new_r1 == r1 and new_c1 == c1:
        return None  # No movement
    
    out = grid.copy()
    out[obj['full_mask']] = bg
    for ddr in range(oh):
        for ddc in range(ow):
            if obj['mask'][ddr, ddc]:
                out[new_r1 + ddr, new_c1 + ddc] = obj['crop'][ddr, ddc]
    
    return out


def move_all_gravity(grid, bg, direction):
    """Move ALL objects in one direction (gravity)"""
    objects = detect_objects(grid, bg)
    if not objects:
        return grid.copy()
    
    H, W = grid.shape
    out = np.full_like(grid, bg)
    
    # Sort objects by position (process furthest first for the gravity direction)
    if direction == 'down':
        objects.sort(key=lambda o: -o['bbox'][2])
    elif direction == 'up':
        objects.sort(key=lambda o: o['bbox'][0])
    elif direction == 'right':
        objects.sort(key=lambda o: -o['bbox'][3])
    elif direction == 'left':
        objects.sort(key=lambda o: o['bbox'][1])
    
    occupied = np.zeros((H, W), dtype=bool)
    
    for obj in objects:
        r1, c1, r2, c2 = obj['bbox']
        oh, ow = r2 - r1 + 1, c2 - c1 + 1
        
        # Slide until hitting wall or occupied cell
        new_r1, new_c1 = r1, c1
        
        if direction in ('down', 'up'):
            step = 1 if direction == 'down' else -1
            while True:
                next_r1 = new_r1 + step
                if next_r1 < 0 or next_r1 + oh > H:
                    break
                blocked = False
                for dr in range(oh):
                    for dc in range(ow):
                        if obj['mask'][dr, dc] and occupied[next_r1 + dr, new_c1 + dc]:
                            blocked = True
                            break
                    if blocked: break
                if blocked: break
                new_r1 = next_r1
        else:
            step = 1 if direction == 'right' else -1
            while True:
                next_c1 = new_c1 + step
                if next_c1 < 0 or next_c1 + ow > W:
                    break
                blocked = False
                for dr in range(oh):
                    for dc in range(ow):
                        if obj['mask'][dr, dc] and occupied[new_r1 + dr, next_c1 + dc]:
                            blocked = True
                            break
                    if blocked: break
                if blocked: break
                new_c1 = next_c1
        
        # Place
        for dr in range(oh):
            for dc in range(ow):
                if obj['mask'][dr, dc]:
                    out[new_r1 + dr, new_c1 + dc] = obj['crop'][dr, dc]
                    occupied[new_r1 + dr, new_c1 + dc] = True
    
    # Restore non-object cells (separator lines etc)
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg and not any(o['full_mask'][r, c] for o in objects):
                out[r, c] = grid[r, c]
    
    return out


# ============================================================
# CONDITIONAL CROSS PIECES
# ============================================================

def make_conditional_piece(condition_fn, action_a_fn, action_b_fn, name):
    """Create a CrossPiece that applies action_a if condition, else action_b.
    
    condition_fn: (grid, object) -> bool
    action_a_fn: (grid) -> grid  (when True)
    action_b_fn: (grid) -> grid  (when False, can be identity)
    
    The condition itself is a cross piece — composable.
    """
    def apply(inp_g):
        inp = _g(inp_g)
        bg = _bg(inp)
        objects = detect_objects(inp, bg)
        
        out = inp.copy()
        
        for obj in objects:
            if condition_fn(inp, obj, bg):
                # Apply action_a to this object's region
                r1, c1, r2, c2 = obj['bbox']
                sub = inp[r1:r2+1, c1:c2+1].copy()
                result = action_a_fn(_l(sub))
                if result is not None:
                    result = _g(result)
                    if result.shape == sub.shape:
                        for dr in range(result.shape[0]):
                            for dc in range(result.shape[1]):
                                if obj['mask'][dr, dc]:
                                    out[r1+dr, c1+dc] = result[dr, dc]
            elif action_b_fn is not None:
                r1, c1, r2, c2 = obj['bbox']
                sub = inp[r1:r2+1, c1:c2+1].copy()
                result = action_b_fn(_l(sub))
                if result is not None:
                    result = _g(result)
                    if result.shape == sub.shape:
                        for dr in range(result.shape[0]):
                            for dc in range(result.shape[1]):
                                if obj['mask'][dr, dc]:
                                    out[r1+dr, c1+dc] = result[dr, dc]
        
        return _l(out)
    
    return CrossPiece(name=f"cond:{name}", apply_fn=apply)


# ============================================================
# CONDITION PRIMITIVES (also cross pieces)
# ============================================================

def cond_has_color(color):
    return lambda grid, obj, bg: color in obj['colors']

def cond_size_gt(threshold):
    return lambda grid, obj, bg: obj['area'] > threshold

def cond_size_lt(threshold):
    return lambda grid, obj, bg: obj['area'] < threshold

def cond_is_largest(grid, obj, bg):
    objects = detect_objects(grid, bg)
    return obj['area'] == max(o['area'] for o in objects)

def cond_is_smallest(grid, obj, bg):
    objects = detect_objects(grid, bg)
    return obj['area'] == min(o['area'] for o in objects)

def cond_single_color(grid, obj, bg):
    return obj['n_colors'] == 1

def cond_multi_color(grid, obj, bg):
    return obj['n_colors'] > 1

def cond_touches_border(grid, obj, bg):
    H, W = grid.shape
    r1, c1, r2, c2 = obj['bbox']
    return r1 == 0 or c1 == 0 or r2 == H-1 or c2 == W-1

def cond_is_square(grid, obj, bg):
    return obj['shape'][0] == obj['shape'][1]


# ============================================================
# LEARN MOVEMENT RULES
# ============================================================

def _verify(fn, train_pairs):
    for inp, out in train_pairs:
        pred = fn(inp)
        if pred is None or not np.array_equal(_g(pred), _g(out)):
            return False
    return True


def learn_gravity_with_collision(train_pairs):
    """Learn gravity direction with proper object collision handling"""
    for direction in ['down', 'up', 'left', 'right']:
        def apply_fn(inp_g, d=direction):
            inp = _g(inp_g)
            bg = _bg(inp)
            result = move_all_gravity(inp, bg, d)
            return _l(result)
        
        if _verify(apply_fn, train_pairs):
            return CrossPiece(name=f"rot_cross:gravity_{direction}", apply_fn=apply_fn)
    
    return None


def learn_rotated_rules(train_pairs):
    """Try existing rules in all 8 orientations"""
    pieces = []
    
    # For each basic rule, try all orientations
    basic_rules = []
    
    # Gravity with collision
    for direction in ['down', 'up', 'left', 'right']:
        def make_grav(d):
            def fn(inp_g):
                inp = _g(inp_g)
                bg = _bg(inp)
                return _l(move_all_gravity(inp, bg, d))
            return fn
        basic_rules.append((f'gravity_{direction}', make_grav(direction)))
    
    # Move all objects toward nearest
    def move_all_toward_nearest(inp_g):
        inp = _g(inp_g)
        bg = _bg(inp)
        objects = detect_objects(inp, bg)
        
        out = inp.copy()
        for obj in objects:
            result = move_toward_nearest(out, obj, objects, bg)
            if result is not None:
                out = result
                # Re-detect after each move
                objects = detect_objects(out, bg)
        return _l(out)
    
    basic_rules.append(('move_toward_nearest', move_all_toward_nearest))
    
    # Try each in all 8 orientations
    for name, fn in basic_rules:
        for orient_name, fwd, inv in _orientations():
            rotated = rotate_piece(fn, orient_name, fwd, inv)
            if _verify(rotated, train_pairs):
                pieces.append(CrossPiece(
                    name=f"rot_cross:{name}@{orient_name}",
                    apply_fn=rotated
                ))
                return pieces  # Return first match
    
    return pieces


def generate_rotating_cross_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Main entry: generate pieces using rotating cross structure"""
    pieces = []
    if not train_pairs:
        return pieces
    
    inp0, out0 = _g(train_pairs[0][0]), _g(train_pairs[0][1])
    if inp0.shape != out0.shape:
        return pieces
    
    bg = _bg(inp0)
    
    # Quick check: are there multiple objects?
    mask = (inp0 != bg).astype(int)
    _, n_obj = connected_components(mask)
    if n_obj < 2:
        return pieces
    
    # Quick check: does the output have moved objects? (fill+erase pattern)
    diff = (inp0 != out0)
    if diff.sum() == 0:
        return pieces
    
    # 1. Try rotated movement rules (only 4 gravity directions, skip full 8-orient)
    for direction in ['down', 'up', 'left', 'right']:
        def make_fn(d):
            def fn(inp_g):
                inp = _g(inp_g)
                b = _bg(inp)
                return _l(move_all_gravity(inp, b, d))
            return fn
        
        fn = make_fn(direction)
        if _verify(fn, train_pairs):
            pieces.append(CrossPiece(name=f"rot_cross:gravity_{direction}", apply_fn=fn))
            return pieces
    
    # Learn: for each object, what happens to it?
    # Condition: object property → action: recolor to specific color
    conditions = [
        ('largest', cond_is_largest),
        ('smallest', cond_is_smallest),
        ('single_color', cond_single_color),
        ('multi_color', cond_multi_color),
        ('touches_border', cond_touches_border),
        ('is_square', cond_is_square),
    ]
    
    for cond_name, cond_fn in conditions:
        # Learn what color objects matching condition get recolored to
        recolor_map = {}  # True/False -> target_color
        consistent = True
        
        for inp_g, out_g in train_pairs:
            inp, out = _g(inp_g), _g(out_g)
            bg_local = _bg(inp)
            objects = detect_objects(inp, bg_local)
            
            for obj in objects:
                matches = cond_fn(inp, obj, bg_local)
                
                # What color does this object become in output?
                r1, c1, r2, c2 = obj['bbox']
                out_colors = Counter()
                for dr in range(r2-r1+1):
                    for dc in range(c2-c1+1):
                        if obj['mask'][dr, dc]:
                            out_colors[int(out[r1+dr, c1+dc])] += 1
                
                target = out_colors.most_common(1)[0][0]
                
                key = matches
                if key in recolor_map:
                    if recolor_map[key] != target:
                        consistent = False
                        break
                else:
                    recolor_map[key] = target
            
            if not consistent:
                break
        
        if consistent and len(recolor_map) == 2 and recolor_map.get(True) != recolor_map.get(False):
            # We have a conditional recolor rule!
            color_true = recolor_map[True]
            color_false = recolor_map[False]
            
            def make_apply(cf, ct, cf2):
                def apply_fn(inp_g):
                    inp = _g(inp_g)
                    bg_local = _bg(inp)
                    objects = detect_objects(inp, bg_local)
                    out = inp.copy()
                    
                    for obj in objects:
                        target = ct if cf(inp, obj, bg_local) else cf2
                        r1, c1, r2, c2 = obj['bbox']
                        for dr in range(r2-r1+1):
                            for dc in range(c2-c1+1):
                                if obj['mask'][dr, dc]:
                                    out[r1+dr, c1+dc] = target
                    
                    return _l(out)
                return apply_fn
            
            fn = make_apply(cond_fn, color_true, color_false)
            if _verify(fn, train_pairs):
                pieces.append(CrossPiece(
                    name=f"rot_cross:cond_{cond_name}_recolor",
                    apply_fn=fn
                ))
                return pieces
    
    return pieces
