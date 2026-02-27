"""
arc/object_mover.py — Object Movement Engine for Cross Structure

Detects objects as connected components, determines movement direction,
applies movement with collision handling.

Movement strategies (all verified against training pairs):
1. gravity_obj: All objects slide in one direction until hitting wall/other object
2. slide_to_anchor: Movers slide toward static anchor objects
3. reflect_line: Objects reflect across a separator line
4. converge_center: Objects move toward grid center
5. converge_point: Objects move toward a specific point
6. move_toward_same_color: Objects move toward same-color partner
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set, Dict
from collections import deque, Counter
from dataclasses import dataclass, field
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


@dataclass
class Obj:
    """A connected component (object) in the grid."""
    color: int
    cells: Set[Tuple[int, int]]
    
    @property
    def size(self) -> int:
        return len(self.cells)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        rs = [r for r, c in self.cells]
        cs = [c for r, c in self.cells]
        return (min(rs), min(cs), max(rs), max(cs))
    
    @property
    def center(self) -> Tuple[float, float]:
        b = self.bbox
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    
    @property 
    def shape_key(self) -> tuple:
        """Normalized shape (translated to origin)."""
        r0, c0 = min(r for r,c in self.cells), min(c for r,c in self.cells)
        return tuple(sorted((r - r0, c - c0) for r, c in self.cells))


def detect_objects(grid: Grid, bg: int) -> List[Obj]:
    """Detect connected components (4-connected, same color)."""
    h, w = grid_shape(grid)
    visited: Set[Tuple[int, int]] = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                color = grid[r][c]
                comp: Set[Tuple[int, int]] = set()
                q = deque([(r, c)])
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in comp or cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if grid[cr][cc] != color:
                        continue
                    comp.add((cr, cc))
                    visited.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        q.append((cr + dr, cc + dc))
                objects.append(Obj(color=color, cells=comp))
    return objects


def detect_objects_multicolor(grid: Grid, bg: int) -> List[Obj]:
    """Detect connected components (4-connected, any non-bg color)."""
    h, w = grid_shape(grid)
    visited: Set[Tuple[int, int]] = set()
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and (r, c) not in visited:
                comp: Set[Tuple[int, int]] = set()
                q = deque([(r, c)])
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in comp or cr < 0 or cr >= h or cc < 0 or cc >= w:
                        continue
                    if grid[cr][cc] == bg:
                        continue
                    comp.add((cr, cc))
                    visited.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        q.append((cr + dr, cc + dc))
                # Use most common non-bg color as representative
                colors = [grid[r][c] for r, c in comp]
                main_color = Counter(colors).most_common(1)[0][0]
                objects.append(Obj(color=main_color, cells=comp))
    return objects


def _place_objects(h: int, w: int, bg: int, objects: List[Tuple[Obj, int, int, Grid]], 
                   static_cells: Dict[Tuple[int, int], int]) -> Optional[Grid]:
    """Place objects at new positions on a bg grid. dr/dc are offsets."""
    result = [[bg] * w for _ in range(h)]
    
    # Place static cells first
    for (r, c), v in static_cells.items():
        result[r][c] = v
    
    # Place moved objects
    for obj, dr, dc, inp in objects:
        for r, c in obj.cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = inp[r][c]
            else:
                return None  # Out of bounds
    return result


def _slide_object(obj: Obj, dr: int, dc: int, h: int, w: int,
                  obstacles: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Slide object in direction (dr,dc) until hitting wall or obstacle. Returns total (dr, dc)."""
    total_dr, total_dc = 0, 0
    for step in range(1, max(h, w)):
        # Check if all cells can move one more step
        can_move = True
        for r, c in obj.cells:
            nr = r + total_dr + dr
            nc = c + total_dc + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                can_move = False
                break
            if (nr, nc) in obstacles and (nr, nc) not in obj.cells:
                can_move = False
                break
        if not can_move:
            break
        total_dr += dr
        total_dc += dc
    return total_dr, total_dc


# ==================== Movement Strategies ====================

def try_gravity_objects(train_pairs: List[Tuple[Grid, Grid]]) -> List[Grid]:
    """Try sliding all objects in one direction (gravity), with obstacle stopping."""
    results = []
    
    for direction_name, (dr, dc) in [
        ('down', (1, 0)), ('up', (-1, 0)), ('left', (0, -1)), ('right', (0, 1))
    ]:
        ok = True
        test_results = []
        
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            objects = detect_objects(inp, bg)
            
            if not objects:
                ok = False
                break
            
            # Sort objects so we process them in order of direction
            # (furthest in direction first, so they don't block each other)
            if dr > 0:
                objects.sort(key=lambda o: -max(r for r, c in o.cells))
            elif dr < 0:
                objects.sort(key=lambda o: min(r for r, c in o.cells))
            elif dc > 0:
                objects.sort(key=lambda o: -max(c for r, c in o.cells))
            elif dc < 0:
                objects.sort(key=lambda o: min(c for r, c in o.cells))
            
            result = [[bg] * w for _ in range(h)]
            occupied: Set[Tuple[int, int]] = set()
            
            for obj in objects:
                tdr, tdc = _slide_object(obj, dr, dc, h, w, occupied)
                for r, c in obj.cells:
                    nr, nc = r + tdr, c + tdc
                    result[nr][nc] = inp[r][c]
                    occupied.add((nr, nc))
            
            if not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            results.append(('gravity_obj', direction_name, dr, dc))
    
    return results


def try_slide_to_anchor(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try: some objects are static anchors, others slide toward them.
    Direction is determined per-object based on relative position to nearest anchor."""
    results = []
    
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    in_objs = detect_objects(inp0, bg)
    out_objs = detect_objects(out0, bg)
    
    if len(in_objs) < 2:
        return []
    
    # Find which objects are static (same position in input and output)
    static_colors = set()
    mover_colors = set()
    
    for io in in_objs:
        found_static = False
        for oo in out_objs:
            if io.cells == oo.cells and io.color == oo.color:
                found_static = True
                break
        if found_static:
            static_colors.add(io.color)
        else:
            mover_colors.add(io.color)
    
    if not static_colors or not mover_colors:
        return []
    
    # Strategy A: fixed direction (original)
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg_t = most_common_color(inp)
            result = _apply_slide_fixed(inp, h, w, bg_t, static_colors, dr, dc)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            results.append(('slide_to_anchor', static_colors, mover_colors, dr, dc))
    
    # Strategy B: direction toward nearest anchor (per mover)
    ok = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        bg_t = most_common_color(inp)
        result = _apply_slide_toward_anchor(inp, h, w, bg_t, static_colors)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    if ok:
        results.append(('slide_toward_anchor', static_colors, mover_colors))
    
    # Strategy C: align with anchor (move to same row/col as anchor)
    ok = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        bg_t = most_common_color(inp)
        result = _apply_slide_align_anchor(inp, h, w, bg_t, static_colors)
        if result is None or not grid_eq(result, out):
            ok = False
            break
    if ok:
        results.append(('slide_align_anchor', static_colors, mover_colors))
    
    return results


def _apply_slide_fixed(inp, h, w, bg, static_colors, dr, dc):
    objects = detect_objects(inp, bg)
    result = [[bg] * w for _ in range(h)]
    anchors_cells: Set[Tuple[int, int]] = set()
    movers = []
    for obj in objects:
        if obj.color in static_colors:
            for r, c in obj.cells:
                result[r][c] = inp[r][c]
                anchors_cells.add((r, c))
        else:
            movers.append(obj)
    
    if dr > 0:
        movers.sort(key=lambda o: -max(r for r, c in o.cells))
    elif dr < 0:
        movers.sort(key=lambda o: min(r for r, c in o.cells))
    elif dc > 0:
        movers.sort(key=lambda o: -max(c for r, c in o.cells))
    elif dc < 0:
        movers.sort(key=lambda o: min(c for r, c in o.cells))
    
    occupied = set(anchors_cells)
    for obj in movers:
        tdr, tdc = _slide_object(obj, dr, dc, h, w, occupied)
        for r, c in obj.cells:
            nr, nc = r + tdr, c + tdc
            result[nr][nc] = inp[r][c]
            occupied.add((nr, nc))
    return result


def _apply_slide_toward_anchor(inp, h, w, bg, static_colors):
    """Each mover slides toward nearest anchor object."""
    objects = detect_objects(inp, bg)
    result = [[bg] * w for _ in range(h)]
    
    anchors = []
    movers = []
    for obj in objects:
        if obj.color in static_colors:
            anchors.append(obj)
            for r, c in obj.cells:
                result[r][c] = inp[r][c]
        else:
            movers.append(obj)
    
    if not anchors:
        return None
    
    occupied: Set[Tuple[int, int]] = set()
    for a in anchors:
        occupied.update(a.cells)
    
    # Sort movers by distance to nearest anchor (closest first)
    def dist_to_nearest_anchor(obj):
        return min(abs(obj.center[0] - a.center[0]) + abs(obj.center[1] - a.center[1]) for a in anchors)
    movers.sort(key=dist_to_nearest_anchor)
    
    for obj in movers:
        # Find nearest anchor
        nearest = min(anchors, key=lambda a: abs(obj.center[0] - a.center[0]) + abs(obj.center[1] - a.center[1]))
        
        # Determine direction (primary axis toward anchor)
        dy = nearest.center[0] - obj.center[0]
        dx = nearest.center[1] - obj.center[1]
        
        if abs(dy) >= abs(dx):
            dr = 1 if dy > 0 else -1
            dc = 0
        else:
            dr = 0
            dc = 1 if dx > 0 else -1
        
        tdr, tdc = _slide_object(obj, dr, dc, h, w, occupied)
        for r, c in obj.cells:
            nr, nc = r + tdr, c + tdc
            result[nr][nc] = inp[r][c]
            occupied.add((nr, nc))
    
    return result


def _apply_slide_align_anchor(inp, h, w, bg, static_colors):
    """Each mover slides to align with nearest anchor's row/col range."""
    objects = detect_objects(inp, bg)
    result = [[bg] * w for _ in range(h)]
    
    anchors = []
    movers = []
    for obj in objects:
        if obj.color in static_colors:
            anchors.append(obj)
            for r, c in obj.cells:
                result[r][c] = inp[r][c]
        else:
            movers.append(obj)
    
    if not anchors:
        return None
    
    occupied: Set[Tuple[int, int]] = set()
    for a in anchors:
        occupied.update(a.cells)
    
    def dist_to_nearest(obj):
        return min(abs(obj.center[0] - a.center[0]) + abs(obj.center[1] - a.center[1]) for a in anchors)
    movers.sort(key=dist_to_nearest)
    
    for obj in movers:
        nearest = min(anchors, key=lambda a: abs(obj.center[0] - a.center[0]) + abs(obj.center[1] - a.center[1]))
        
        obj_r1, obj_c1, obj_r2, obj_c2 = obj.bbox
        anc_r1, anc_c1, anc_r2, anc_c2 = nearest.bbox
        
        # Compute minimal shift to align row-range with anchor row-range
        # Try: align obj rows to overlap with anchor rows
        if obj_r2 < anc_r1:
            shift_r = anc_r1 - obj_r1  # align top of obj with top of anchor
        elif obj_r1 > anc_r2:
            shift_r = anc_r2 - obj_r2  # align bottom of obj with bottom of anchor
        else:
            shift_r = 0
        
        if obj_c2 < anc_c1:
            shift_c = anc_c1 - obj_c1  # align left of obj with left of anchor
        elif obj_c1 > anc_c2:
            shift_c = anc_c2 - obj_c2  # align right of obj with right of anchor
        else:
            shift_c = 0
        
        # Apply shift on primary axis only, use slide to handle collisions
        if abs(shift_r) >= abs(shift_c):
            shift_c = 0
        else:
            shift_r = 0
        
        # Use slide to handle collisions — slide in the shift direction
        if shift_r != 0:
            dr = 1 if shift_r > 0 else -1
            tdr, _ = _slide_object(obj, dr, 0, h, w, occupied)
            # Cap at target alignment
            if abs(tdr) > abs(shift_r):
                tdr = shift_r
            for r, c in obj.cells:
                result[r + tdr][c] = inp[r][c]
                occupied.add((r + tdr, c))
        elif shift_c != 0:
            dc = 1 if shift_c > 0 else -1
            _, tdc = _slide_object(obj, 0, dc, h, w, occupied)
            if abs(tdc) > abs(shift_c):
                tdc = shift_c
            for r, c in obj.cells:
                result[r][c + tdc] = inp[r][c]
                occupied.add((r, c + tdc))
        else:
            # No movement needed
            for r, c in obj.cells:
                result[r][c] = inp[r][c]
                occupied.add((r, c))
    
    return result


def try_converge_point(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try: all objects move toward a specific point (center, corner, etc)."""
    results = []
    
    for target_name, target_fn in [
        ('center', lambda h, w: (h / 2, w / 2)),
        ('tl', lambda h, w: (0, 0)),
        ('tr', lambda h, w: (0, w - 1)),
        ('bl', lambda h, w: (h - 1, 0)),
        ('br', lambda h, w: (h - 1, w - 1)),
    ]:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            objects = detect_objects(inp, bg)
            if not objects:
                ok = False
                break
            
            ty, tx = target_fn(h, w)
            
            result = [[bg] * w for _ in range(h)]
            occupied: Set[Tuple[int, int]] = set()
            
            # Sort by distance to target (closest first — they arrive first)
            objects.sort(key=lambda o: abs(o.center[0] - ty) + abs(o.center[1] - tx))
            
            for obj in objects:
                cy, cx = obj.center
                # Determine direction toward target
                dy = 0 if abs(cy - ty) < 0.5 else (1 if ty > cy else -1)
                dx = 0 if abs(cx - tx) < 0.5 else (1 if tx > cx else -1)
                
                if dy == 0 and dx == 0:
                    # Already at target
                    for r, c in obj.cells:
                        result[r][c] = inp[r][c]
                        occupied.add((r, c))
                    continue
                
                # Slide toward target
                tdr, tdc = _slide_object(obj, dy, dx, h, w, occupied)
                for r, c in obj.cells:
                    nr, nc = r + tdr, c + tdc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr][nc] = inp[r][c]
                        occupied.add((nr, nc))
                    else:
                        ok = False
                        break
                if not ok:
                    break
            
            if ok and not grid_eq(result, out):
                ok = False
        
        if ok:
            results.append(('converge_point', target_name))
    
    return results


def try_reflect_line(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try: objects reflect across a horizontal or vertical separator line."""
    results = []
    
    inp0 = train_pairs[0][0]
    h, w = grid_shape(inp0)
    bg = most_common_color(inp0)
    
    # Find potential separator lines (full row/col of single non-bg color)
    for r in range(h):
        if len(set(inp0[r])) == 1 and inp0[r][0] != bg:
            sep_color = inp0[r][0]
            # Try reflecting all non-separator objects across this line
            ok = True
            for inp, out in train_pairs:
                ht, wt = grid_shape(inp)
                bg_t = most_common_color(inp)
                
                # Find the separator row in this input
                sep_row = None
                for rt in range(ht):
                    if len(set(inp[rt])) == 1 and inp[rt][0] == sep_color:
                        sep_row = rt
                        break
                if sep_row is None:
                    ok = False
                    break
                
                result = [list(row) for row in inp]
                objects = detect_objects(inp, bg_t)
                # Remove separator from objects
                objects = [o for o in objects if o.color != sep_color]
                
                for obj in objects:
                    for r2, c2 in obj.cells:
                        # Reflect across sep_row
                        nr = 2 * sep_row - r2
                        if 0 <= nr < ht:
                            result[nr][c2] = inp[r2][c2]
                
                if not grid_eq(result, out):
                    ok = False
                    break
            
            if ok:
                results.append(('reflect_h', sep_color))
    
    # Vertical separator
    for c in range(w):
        col = [inp0[r][c] for r in range(h)]
        if len(set(col)) == 1 and col[0] != bg:
            sep_color = col[0]
            ok = True
            for inp, out in train_pairs:
                ht, wt = grid_shape(inp)
                bg_t = most_common_color(inp)
                
                sep_col = None
                for ct in range(wt):
                    col_t = [inp[rt][ct] for rt in range(ht)]
                    if len(set(col_t)) == 1 and col_t[0] == sep_color:
                        sep_col = ct
                        break
                if sep_col is None:
                    ok = False
                    break
                
                result = [list(row) for row in inp]
                objects = detect_objects(inp, bg_t)
                objects = [o for o in objects if o.color != sep_color]
                
                for obj in objects:
                    for r2, c2 in obj.cells:
                        nc = 2 * sep_col - c2
                        if 0 <= nc < wt:
                            result[r2][nc] = inp[r2][c2]
                
                if not grid_eq(result, out):
                    ok = False
                    break
            
            if ok:
                results.append(('reflect_v', sep_color))
    
    return results


def try_move_to_contact(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try: each object slides until it touches another object (any direction per object)."""
    results = []
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        bg = most_common_color(inp)
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        
        if len(in_objs) < 2:
            return []
    
    # Try: for each pair of directions, find which makes movers contact anchors
    # This is handled by slide_to_anchor, skip for now
    return results


def try_per_object_gravity(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try: different objects get different gravity directions based on their position/color."""
    # Detect pattern: objects in top half go down, bottom half go up, etc.
    results = []
    
    for split_axis in ['h', 'v']:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            objects = detect_objects(inp, bg)
            
            result = [[bg] * w for _ in range(h)]
            occupied: Set[Tuple[int, int]] = set()
            
            if split_axis == 'h':
                mid = h / 2
                # Top objects go down, bottom go up
                top = [o for o in objects if o.center[0] < mid]
                bot = [o for o in objects if o.center[0] >= mid]
                # Process bottom-half objects going up first (they're further from midline)
                top.sort(key=lambda o: -o.center[0])  # furthest down first
                bot.sort(key=lambda o: o.center[0])    # furthest up first
                
                for obj in top:
                    tdr, tdc = _slide_object(obj, 1, 0, h, w, occupied)
                    for r, c in obj.cells:
                        nr, nc = r + tdr, c + tdc
                        result[nr][nc] = inp[r][c]
                        occupied.add((nr, nc))
                
                for obj in bot:
                    tdr, tdc = _slide_object(obj, -1, 0, h, w, occupied)
                    for r, c in obj.cells:
                        nr, nc = r + tdr, c + tdc
                        result[nr][nc] = inp[r][c]
                        occupied.add((nr, nc))
            else:
                mid = w / 2
                left = [o for o in objects if o.center[1] < mid]
                right = [o for o in objects if o.center[1] >= mid]
                left.sort(key=lambda o: -o.center[1])
                right.sort(key=lambda o: o.center[1])
                
                for obj in left:
                    tdr, tdc = _slide_object(obj, 0, 1, h, w, occupied)
                    for r, c in obj.cells:
                        nr, nc = r + tdr, c + tdc
                        result[nr][nc] = inp[r][c]
                        occupied.add((nr, nc))
                
                for obj in right:
                    tdr, tdc = _slide_object(obj, 0, -1, h, w, occupied)
                    for r, c in obj.cells:
                        nr, nc = r + tdr, c + tdc
                        result[nr][nc] = inp[r][c]
                        occupied.add((nr, nc))
            
            if not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            results.append(('per_object_gravity', split_axis))
    
    return results


def try_gravity_multicolor(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try gravity with multicolor objects (all non-bg cells as one connected component)."""
    results = []
    
    for direction_name, (dr, dc) in [
        ('down', (1, 0)), ('up', (-1, 0)), ('left', (0, -1)), ('right', (0, 1))
    ]:
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg = most_common_color(inp)
            objects = detect_objects_multicolor(inp, bg)
            
            if not objects:
                ok = False
                break
            
            if dr > 0:
                objects.sort(key=lambda o: -max(r for r, c in o.cells))
            elif dr < 0:
                objects.sort(key=lambda o: min(r for r, c in o.cells))
            elif dc > 0:
                objects.sort(key=lambda o: -max(c for r, c in o.cells))
            elif dc < 0:
                objects.sort(key=lambda o: min(c for r, c in o.cells))
            
            result = [[bg] * w for _ in range(h)]
            occupied: Set[Tuple[int, int]] = set()
            
            for obj in objects:
                tdr, tdc = _slide_object(obj, dr, dc, h, w, occupied)
                for r, c in obj.cells:
                    nr, nc = r + tdr, c + tdc
                    result[nr][nc] = inp[r][c]
                    occupied.add((nr, nc))
            
            if not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            results.append(('gravity_mc', direction_name, dr, dc))
    
    return results


def try_slide_toward_anchor_mc(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try slide toward anchor with multicolor objects."""
    results = []
    
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    in_objs = detect_objects_multicolor(inp0, bg)
    out_objs = detect_objects_multicolor(out0, bg)
    
    if len(in_objs) < 2:
        return []
    
    # Find static objects
    static_shapes = set()
    mover_indices = []
    for i, io in enumerate(in_objs):
        found_static = any(io.cells == oo.cells for oo in out_objs)
        if found_static:
            static_shapes.add(frozenset(io.cells))
        else:
            mover_indices.append(i)
    
    if not static_shapes or not mover_indices:
        return []
    
    # Verify with "slide toward nearest static" strategy
    ok = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        bg_t = most_common_color(inp)
        objects = detect_objects_multicolor(inp, bg_t)
        
        result = [[bg_t] * w for _ in range(h)]
        anchors = []
        movers = []
        
        for obj in objects:
            if any(obj.cells == fs for fs in [frozenset(a.cells) for a in anchors]):
                continue
            # Check if static by seeing if cells match in output
            out_objs_t = detect_objects_multicolor(out, bg_t)
            is_static = any(obj.cells == oo.cells for oo in out_objs_t)
            if is_static and obj.size > 1:  # anchors tend to be larger
                anchors.append(obj)
                for r, c in obj.cells:
                    result[r][c] = inp[r][c]
            else:
                movers.append(obj)
        
        if not anchors:
            ok = False
            break
        
        occupied: Set[Tuple[int, int]] = set()
        for a in anchors:
            occupied.update(a.cells)
        
        movers.sort(key=lambda o: min(
            abs(o.center[0] - a.center[0]) + abs(o.center[1] - a.center[1]) 
            for a in anchors
        ))
        
        for obj in movers:
            nearest = min(anchors, key=lambda a: abs(obj.center[0] - a.center[0]) + abs(obj.center[1] - a.center[1]))
            dy = nearest.center[0] - obj.center[0]
            dx = nearest.center[1] - obj.center[1]
            
            if abs(dy) >= abs(dx):
                dr_d = 1 if dy > 0 else -1
                dc_d = 0
            else:
                dr_d = 0
                dc_d = 1 if dx > 0 else -1
            
            tdr, tdc = _slide_object(obj, dr_d, dc_d, h, w, occupied)
            for r, c in obj.cells:
                nr, nc = r + tdr, c + tdc
                result[nr][nc] = inp[r][c]
                occupied.add((nr, nc))
        
        if not grid_eq(result, out):
            ok = False
            break
    
    if ok:
        results.append(('slide_toward_anchor_mc',))
    
    return results


def try_color_grouped_gravity(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try: each color group gets its own gravity direction."""
    results = []
    
    inp0 = train_pairs[0][0]
    bg = most_common_color(inp0)
    colors = set()
    for row in inp0:
        for c in row:
            if c != bg:
                colors.add(c)
    
    if len(colors) < 2 or len(colors) > 5:
        return []
    
    # Try all direction assignments for each color (up to 4^5 = 1024 combos)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    color_list = sorted(colors)
    
    if len(color_list) > 3:  # Too many combos
        return []
    
    from itertools import product
    for dir_combo in product(range(4), repeat=len(color_list)):
        color_dirs = {color_list[i]: directions[dir_combo[i]] for i in range(len(color_list))}
        
        ok = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            bg_t = most_common_color(inp)
            objects = detect_objects(inp, bg_t)
            
            result = [[bg_t] * w for _ in range(h)]
            occupied: Set[Tuple[int, int]] = set()
            
            # Process colors in order: furthest-in-direction first
            for color, (dr, dc) in color_dirs.items():
                color_objs = [o for o in objects if o.color == color]
                if dr > 0:
                    color_objs.sort(key=lambda o: -max(r for r, c in o.cells))
                elif dr < 0:
                    color_objs.sort(key=lambda o: min(r for r, c in o.cells))
                elif dc > 0:
                    color_objs.sort(key=lambda o: -max(c for r, c in o.cells))
                elif dc < 0:
                    color_objs.sort(key=lambda o: min(c for r, c in o.cells))
                
                for obj in color_objs:
                    tdr, tdc = _slide_object(obj, dr, dc, h, w, occupied)
                    for r, c in obj.cells:
                        nr, nc = r + tdr, c + tdc
                        result[nr][nc] = inp[r][c]
                        occupied.add((nr, nc))
            
            if not grid_eq(result, out):
                ok = False
                break
        
        if ok:
            results.append(('color_grouped_gravity', color_dirs))
            return results  # Return first valid combo
    
    return results


def try_wall_absorb(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """
    Cross mutual monitoring: objects slide toward same-color wall (edge row/col).
    Two 'cross worlds' each absorb their own objects independently.
    """
    results = []
    
    inp0 = train_pairs[0][0]
    h, w = grid_shape(inp0)
    bg = most_common_color(inp0)
    
    # Detect walls: full rows/cols at edges (non-bg)
    walls = {}
    if len(set(inp0[0])) == 1 and inp0[0][0] != bg:
        walls[inp0[0][0]] = 'top'
    if len(set(inp0[h-1])) == 1 and inp0[h-1][0] != bg:
        walls[inp0[h-1][0]] = 'bottom'
    left_col = set(inp0[r][0] for r in range(h))
    if len(left_col) == 1 and inp0[0][0] != bg and inp0[0][0] not in walls:
        walls[inp0[0][0]] = 'left'
    right_col = set(inp0[r][w-1] for r in range(h))
    if len(right_col) == 1 and inp0[0][w-1] != bg and inp0[0][w-1] not in walls:
        walls[inp0[0][w-1]] = 'right'
    
    if len(walls) < 2:
        return []
    
    dir_map = {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    
    # Try different sort orders for sliding
    for sort_mode in ['closest_first', 'farthest_first']:
        ok = True
        for inp, out in train_pairs:
            result = _apply_wall_absorb(inp, walls, dir_map, sort_mode)
            if result is None or not grid_eq(result, out):
                ok = False
                break
        if ok:
            results.append(('wall_absorb', dict(walls), sort_mode))
            return results
    
    return results


def _apply_wall_absorb(inp, walls, dir_map, sort_mode):
    h, w = grid_shape(inp)
    bg = most_common_color(inp)
    
    # Detect walls in this input
    local_walls = {}
    if len(set(inp[0])) == 1 and inp[0][0] != bg:
        local_walls[inp[0][0]] = 'top'
    if len(set(inp[h-1])) == 1 and inp[h-1][0] != bg:
        local_walls[inp[h-1][0]] = 'bottom'
    
    result = [[bg] * w for _ in range(h)]
    wall_cells: Set[Tuple[int, int]] = set()
    
    for color, side in local_walls.items():
        if side == 'top':
            for c in range(w): result[0][c] = color; wall_cells.add((0, c))
        elif side == 'bottom':
            for c in range(w): result[h-1][c] = color; wall_cells.add((h-1, c))
        elif side == 'left':
            for r in range(h): result[r][0] = color; wall_cells.add((r, 0))
        elif side == 'right':
            for r in range(h): result[r][w-1] = color; wall_cells.add((r, w-1))
    
    objects = detect_objects(inp, bg)
    non_wall = [o for o in objects if not any((r, c) in wall_cells for r, c in o.cells)]
    
    # Check all have matching wall color
    for obj in non_wall:
        if obj.color not in local_walls:
            return None
    
    occupied = set(wall_cells)
    
    # Process each wall color separately
    for color, side in local_walls.items():
        dr, dc = dir_map[side]
        color_objs = [o for o in non_wall if o.color == color]
        
        if sort_mode == 'closest_first':
            if side == 'top':
                color_objs.sort(key=lambda o: min(r for r, c in o.cells))
            elif side == 'bottom':
                color_objs.sort(key=lambda o: -max(r for r, c in o.cells))
            elif side == 'left':
                color_objs.sort(key=lambda o: min(c for r, c in o.cells))
            elif side == 'right':
                color_objs.sort(key=lambda o: -max(c for r, c in o.cells))
        else:
            if side == 'top':
                color_objs.sort(key=lambda o: -max(r for r, c in o.cells))
            elif side == 'bottom':
                color_objs.sort(key=lambda o: min(r for r, c in o.cells))
            elif side == 'left':
                color_objs.sort(key=lambda o: -max(c for r, c in o.cells))
            elif side == 'right':
                color_objs.sort(key=lambda o: min(c for r, c in o.cells))
        
        for obj in color_objs:
            tdr, tdc = _slide_object(obj, dr, dc, h, w, occupied)
            for r, c in obj.cells:
                nr, nc = r + tdr, c + tdc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr][nc] = inp[r][c]
                    occupied.add((nr, nc))
    
    return result


def try_sort_objects(train_pairs: List[Tuple[Grid, Grid]]) -> List:
    """Try: objects are sorted (by color, size, etc.) and placed in order."""
    results = []
    
    for sort_key in ['color', 'size', 'size_desc']:
        for direction in ['left_to_right', 'top_to_bottom']:
            ok = True
            for inp, out in train_pairs:
                h, w = grid_shape(inp)
                bg = most_common_color(inp)
                objects = detect_objects(inp, bg)
                
                if not objects:
                    ok = False
                    break
                
                # Sort objects
                if sort_key == 'color':
                    objects.sort(key=lambda o: o.color)
                elif sort_key == 'size':
                    objects.sort(key=lambda o: o.size)
                elif sort_key == 'size_desc':
                    objects.sort(key=lambda o: -o.size)
                
                # Place in order
                result = [[bg] * w for _ in range(h)]
                
                if direction == 'left_to_right':
                    # Sort by original left position to determine slot positions
                    orig_positions = sorted(
                        [(min(c for _, c in o.cells), o) for o in detect_objects(inp, bg)],
                        key=lambda x: x[0]
                    )
                    slot_positions = [p for p, _ in orig_positions]
                    
                    for i, obj in enumerate(objects):
                        if i >= len(slot_positions):
                            break
                        # Get the bounding box of the original slot
                        slot_obj = orig_positions[i][1]
                        dr = min(r for r, c in slot_obj.cells) - min(r for r, c in obj.cells)
                        dc = slot_positions[i] - min(c for r, c in obj.cells)
                        for r, c in obj.cells:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                result[nr][nc] = inp[r][c]
                
                if not grid_eq(result, out):
                    ok = False
                    break
            
            if ok:
                results.append(('sort_objects', sort_key, direction))
    
    return results


# ==================== Main Entry Point ====================

def solve_object_movement(train_pairs: List[Tuple[Grid, Grid]], 
                           test_inputs: List[Grid]) -> Optional[List[Grid]]:
    """Try all object movement strategies. Returns test outputs if any strategy works."""
    
    # Quick check: same size input/output
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    strategies = []
    
    # Try each strategy
    strategies.extend(try_gravity_objects(train_pairs))
    strategies.extend(try_gravity_multicolor(train_pairs))
    strategies.extend(try_slide_to_anchor(train_pairs))
    strategies.extend(try_slide_toward_anchor_mc(train_pairs))
    strategies.extend(try_converge_point(train_pairs))
    strategies.extend(try_reflect_line(train_pairs))
    strategies.extend(try_per_object_gravity(train_pairs))
    strategies.extend(try_color_grouped_gravity(train_pairs))
    strategies.extend(try_wall_absorb(train_pairs))
    strategies.extend(try_sort_objects(train_pairs))
    
    if not strategies:
        return None
    
    # Use first valid strategy to solve test inputs
    strat = strategies[0]
    
    test_outputs = []
    for test_inp in test_inputs:
        h, w = grid_shape(test_inp)
        bg = most_common_color(test_inp)
        
        result = _apply_strategy(strat, test_inp, h, w, bg)
        if result is None:
            return None
        test_outputs.append(result)
    
    return test_outputs


def _apply_strategy(strat, inp: Grid, h: int, w: int, bg: int) -> Optional[Grid]:
    """Apply a learned movement strategy to a single input."""
    name = strat[0]
    
    if name in ('gravity_obj', 'gravity_mc'):
        _, direction_name, dr, dc = strat
        use_mc = (name == 'gravity_mc')
        objects = detect_objects_multicolor(inp, bg) if use_mc else detect_objects(inp, bg)
        if not objects:
            return None
        
        # Sort by direction
        if dr > 0:
            objects.sort(key=lambda o: -max(r for r, c in o.cells))
        elif dr < 0:
            objects.sort(key=lambda o: min(r for r, c in o.cells))
        elif dc > 0:
            objects.sort(key=lambda o: -max(c for r, c in o.cells))
        elif dc < 0:
            objects.sort(key=lambda o: min(c for r, c in o.cells))
        
        result = [[bg] * w for _ in range(h)]
        occupied: Set[Tuple[int, int]] = set()
        
        for obj in objects:
            tdr, tdc = _slide_object(obj, dr, dc, h, w, occupied)
            for r, c in obj.cells:
                nr, nc = r + tdr, c + tdc
                result[nr][nc] = inp[r][c]
                occupied.add((nr, nc))
        return result
    
    elif name == 'slide_to_anchor':
        _, static_colors, mover_colors, dr, dc = strat
        return _apply_slide_fixed(inp, h, w, bg, static_colors, dr, dc)
    
    elif name == 'slide_toward_anchor':
        _, static_colors, mover_colors = strat
        return _apply_slide_toward_anchor(inp, h, w, bg, static_colors)
    
    elif name == 'slide_align_anchor':
        _, static_colors, mover_colors = strat
        return _apply_slide_align_anchor(inp, h, w, bg, static_colors)
    
    elif name == 'converge_point':
        _, target_name = strat
        target_fns = {
            'center': lambda h, w: (h / 2, w / 2),
            'tl': lambda h, w: (0, 0),
            'tr': lambda h, w: (0, w - 1),
            'bl': lambda h, w: (h - 1, 0),
            'br': lambda h, w: (h - 1, w - 1),
        }
        ty, tx = target_fns[target_name](h, w)
        objects = detect_objects(inp, bg)
        objects.sort(key=lambda o: abs(o.center[0] - ty) + abs(o.center[1] - tx))
        
        result = [[bg] * w for _ in range(h)]
        occupied: Set[Tuple[int, int]] = set()
        
        for obj in objects:
            cy, cx = obj.center
            dy = 0 if abs(cy - ty) < 0.5 else (1 if ty > cy else -1)
            dx = 0 if abs(cx - tx) < 0.5 else (1 if tx > cx else -1)
            
            if dy == 0 and dx == 0:
                for r, c in obj.cells:
                    result[r][c] = inp[r][c]
                    occupied.add((r, c))
            else:
                tdr, tdc = _slide_object(obj, dy, dx, h, w, occupied)
                for r, c in obj.cells:
                    nr, nc = r + tdr, c + tdc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr][nc] = inp[r][c]
                        occupied.add((nr, nc))
        return result
    
    elif name == 'reflect_h':
        _, sep_color = strat
        # Find separator row
        sep_row = None
        for r in range(h):
            if len(set(inp[r])) == 1 and inp[r][0] == sep_color:
                sep_row = r
                break
        if sep_row is None:
            return None
        
        result = [list(row) for row in inp]
        objects = detect_objects(inp, bg)
        objects = [o for o in objects if o.color != sep_color]
        for obj in objects:
            for r, c in obj.cells:
                nr = 2 * sep_row - r
                if 0 <= nr < h:
                    result[nr][c] = inp[r][c]
        return result
    
    elif name == 'reflect_v':
        _, sep_color = strat
        sep_col = None
        for c in range(w):
            col = [inp[r][c] for r in range(h)]
            if len(set(col)) == 1 and col[0] == sep_color:
                sep_col = c
                break
        if sep_col is None:
            return None
        
        result = [list(row) for row in inp]
        objects = detect_objects(inp, bg)
        objects = [o for o in objects if o.color != sep_color]
        for obj in objects:
            for r, c in obj.cells:
                nc = 2 * sep_col - c
                if 0 <= nc < w:
                    result[r][nc] = inp[r][c]
        return result
    
    elif name == 'slide_toward_anchor_mc':
        # Re-run the strategy detection to get anchor info, then apply
        # For now, re-detect from this input
        objects = detect_objects_multicolor(inp, bg)
        if len(objects) < 2:
            return None
        
        # Anchors = larger objects, movers = smaller
        objects.sort(key=lambda o: -o.size)
        # Assume largest is anchor
        anchors = [objects[0]]
        movers = objects[1:]
        
        result = [[bg] * w for _ in range(h)]
        for a in anchors:
            for r, c in a.cells:
                result[r][c] = inp[r][c]
        
        occupied: Set[Tuple[int, int]] = set()
        for a in anchors:
            occupied.update(a.cells)
        
        movers.sort(key=lambda o: min(
            abs(o.center[0] - a.center[0]) + abs(o.center[1] - a.center[1])
            for a in anchors
        ))
        
        for obj in movers:
            nearest = min(anchors, key=lambda a: abs(obj.center[0] - a.center[0]) + abs(obj.center[1] - a.center[1]))
            dy = nearest.center[0] - obj.center[0]
            dx = nearest.center[1] - obj.center[1]
            if abs(dy) >= abs(dx):
                dr_d = 1 if dy > 0 else -1
                dc_d = 0
            else:
                dr_d = 0
                dc_d = 1 if dx > 0 else -1
            tdr, tdc = _slide_object(obj, dr_d, dc_d, h, w, occupied)
            for r, c in obj.cells:
                nr, nc = r + tdr, c + tdc
                result[nr][nc] = inp[r][c]
                occupied.add((nr, nc))
        return result
    
    elif name == 'per_object_gravity':
        _, split_axis = strat
        objects = detect_objects(inp, bg)
        result = [[bg] * w for _ in range(h)]
        occupied: Set[Tuple[int, int]] = set()
        
        if split_axis == 'h':
            mid = h / 2
            top = sorted([o for o in objects if o.center[0] < mid], key=lambda o: -o.center[0])
            bot = sorted([o for o in objects if o.center[0] >= mid], key=lambda o: o.center[0])
            
            for obj in top:
                tdr, tdc = _slide_object(obj, 1, 0, h, w, occupied)
                for r, c in obj.cells:
                    result[r + tdr][c] = inp[r][c]
                    occupied.add((r + tdr, c))
            for obj in bot:
                tdr, tdc = _slide_object(obj, -1, 0, h, w, occupied)
                for r, c in obj.cells:
                    result[r + tdr][c] = inp[r][c]
                    occupied.add((r + tdr, c))
        else:
            mid = w / 2
            left = sorted([o for o in objects if o.center[1] < mid], key=lambda o: -o.center[1])
            right = sorted([o for o in objects if o.center[1] >= mid], key=lambda o: o.center[1])
            
            for obj in left:
                tdr, tdc = _slide_object(obj, 0, 1, h, w, occupied)
                for r, c in obj.cells:
                    result[r][c + tdc] = inp[r][c]
                    occupied.add((r, c + tdc))
            for obj in right:
                tdr, tdc = _slide_object(obj, 0, -1, h, w, occupied)
                for r, c in obj.cells:
                    result[r][c + tdc] = inp[r][c]
                    occupied.add((r, c + tdc))
        return result
    
    elif name == 'color_grouped_gravity':
        _, color_dirs = strat
        objects = detect_objects(inp, bg)
        result = [[bg] * w for _ in range(h)]
        occupied: Set[Tuple[int, int]] = set()
        
        for color, (dr_d, dc_d) in color_dirs.items():
            color_objs = [o for o in objects if o.color == color]
            if dr_d > 0:
                color_objs.sort(key=lambda o: -max(r for r, c in o.cells))
            elif dr_d < 0:
                color_objs.sort(key=lambda o: min(r for r, c in o.cells))
            elif dc_d > 0:
                color_objs.sort(key=lambda o: -max(c for r, c in o.cells))
            elif dc_d < 0:
                color_objs.sort(key=lambda o: min(c for r, c in o.cells))
            
            for obj in color_objs:
                tdr, tdc = _slide_object(obj, dr_d, dc_d, h, w, occupied)
                for r, c in obj.cells:
                    nr, nc = r + tdr, c + tdc
                    result[nr][nc] = inp[r][c]
                    occupied.add((nr, nc))
        
        # Handle colors not in color_dirs (leave in place)
        for obj in objects:
            if obj.color not in color_dirs:
                for r, c in obj.cells:
                    result[r][c] = inp[r][c]
        return result
    
    elif name == 'wall_absorb':
        _, wall_map, sort_mode = strat
        dir_map = {'top': (-1, 0), 'bottom': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        return _apply_wall_absorb(inp, wall_map, dir_map, sort_mode)
    
    elif name == 'sort_objects':
        # Complex — for now return None
        return None
    
    return None
