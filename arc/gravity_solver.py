"""
arc/gravity_solver.py — Gravity/Slide Transform Solver

Objects slide in a direction until they hit walls or other objects.
Uses cross structure dimensions for precise collision detection.

Patterns handled:
1. Uniform gravity: all objects slide in one direction (up/down/left/right)
2. Per-object gravity: each object slides in its own direction (toward a wall, center, etc.)
3. Diagonal stacking: objects slide diagonally and stack
4. Slide-to-wall: objects slide until hitting grid boundary
5. Slide-to-object: objects slide until touching another object
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from scipy import ndimage
from collections import Counter

from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.cross_engine import CrossPiece


# ---- Object representation ----

class GravObj:
    """A colored connected component in the grid."""
    __slots__ = ['color', 'pixels', 'mask', 'bbox', 'centroid', 'area']
    
    def __init__(self, color: int, pixels: np.ndarray, grid_shape: Tuple[int, int]):
        self.color = color
        self.pixels = pixels  # Nx2 array of (row, col)
        self.area = len(pixels)
        r0 = pixels[:, 0].min()
        c0 = pixels[:, 1].min()
        r1 = pixels[:, 0].max()
        c1 = pixels[:, 1].max()
        self.bbox = (int(r0), int(c0), int(r1), int(c1))
        self.centroid = (pixels[:, 0].mean(), pixels[:, 1].mean())
        self.mask = np.zeros(grid_shape, dtype=bool)
        self.mask[pixels[:, 0], pixels[:, 1]] = True
    
    def shifted(self, dr: int, dc: int, grid_shape: Tuple[int, int]) -> Optional['GravObj']:
        """Return a new GravObj shifted by (dr, dc), or None if out of bounds."""
        new_pixels = self.pixels.copy()
        new_pixels[:, 0] += dr
        new_pixels[:, 1] += dc
        if (new_pixels[:, 0] < 0).any() or (new_pixels[:, 0] >= grid_shape[0]).any():
            return None
        if (new_pixels[:, 1] < 0).any() or (new_pixels[:, 1] >= grid_shape[1]).any():
            return None
        return GravObj(self.color, new_pixels, grid_shape)
    
    @property
    def shape_sig(self) -> tuple:
        """Normalized shape signature (pixels shifted to origin)."""
        r0, c0 = self.pixels[:, 0].min(), self.pixels[:, 1].min()
        norm = self.pixels - [r0, c0]
        # Sort for consistent comparison
        return tuple(sorted(map(tuple, norm)))


def extract_objects(grid: np.ndarray, bg: int = 0) -> List[GravObj]:
    """Extract all colored connected components as GravObj."""
    h, w = grid.shape
    objects = []
    for color in range(1, 10):
        mask = (grid == color)
        if not mask.any():
            continue
        labeled, n = ndimage.label(mask)
        for lbl in range(1, n + 1):
            pixels = np.argwhere(labeled == lbl)
            objects.append(GravObj(color, pixels, (h, w)))
    return objects


# ---- Direction detection ----

DIRECTIONS = {
    'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1),
    'upleft': (-1, -1), 'upright': (-1, 1), 'downleft': (1, -1), 'downright': (1, 1),
}


def match_objects(objs_in: List[GravObj], objs_out: List[GravObj]) -> List[Tuple[GravObj, GravObj]]:
    """Match input objects to output objects by color + shape."""
    matches = []
    used = set()
    for oi in objs_in:
        best_j = None
        best_dist = float('inf')
        for j, oo in enumerate(objs_out):
            if j in used:
                continue
            if oi.color != oo.color or oi.area != oo.area:
                continue
            if oi.shape_sig != oo.shape_sig:
                continue
            dist = abs(oi.centroid[0] - oo.centroid[0]) + abs(oi.centroid[1] - oo.centroid[1])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            matches.append((oi, objs_out[best_j]))
            used.add(best_j)
    return matches


def detect_movements(matches: List[Tuple[GravObj, GravObj]]) -> List[Tuple[GravObj, int, int]]:
    """For each matched pair, compute (obj, dr, dc)."""
    movements = []
    for oi, oo in matches:
        dr = int(round(oo.centroid[0] - oi.centroid[0]))
        dc = int(round(oo.centroid[1] - oi.centroid[1]))
        movements.append((oi, dr, dc))
    return movements


# ---- Slide simulation ----

def slide_object(obj: GravObj, dr: int, dc: int, obstacles: np.ndarray, 
                 grid_shape: Tuple[int, int]) -> GravObj:
    """
    Slide object in direction (dr, dc) one step at a time until collision.
    dr, dc should be -1, 0, or 1 (unit direction).
    """
    current = obj
    while True:
        candidate = current.shifted(dr, dc, grid_shape)
        if candidate is None:
            break
        # Check collision with obstacles (other objects)
        if (obstacles[candidate.pixels[:, 0], candidate.pixels[:, 1]]).any():
            break
        current = candidate
    return current


def build_obstacle_mask(grid_shape: Tuple[int, int], objects: List[GravObj], 
                        exclude: Optional[GravObj] = None) -> np.ndarray:
    """Build a boolean mask of all object pixels except the excluded one."""
    mask = np.zeros(grid_shape, dtype=bool)
    for obj in objects:
        if exclude is not None and obj is exclude:
            continue
        mask |= obj.mask
    return mask


# ---- Gravity strategies ----

def try_uniform_gravity(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Try: all objects slide in the same direction until hitting wall.
    """
    pieces = []
    
    for dir_name, (dr, dc) in DIRECTIONS.items():
        all_match = True
        for inp, out in train_pairs:
            inp_np = np.array(inp)
            out_np = np.array(out)
            h, w = inp_np.shape
            
            objs = extract_objects(inp_np, bg)
            if not objs:
                all_match = False
                break
            
            # Simulate: slide each object in this direction
            result = np.full_like(inp_np, bg)
            
            # Sort objects so we process those in the slide direction last
            # (they're "first to land")
            if dr > 0:
                objs.sort(key=lambda o: -o.bbox[2])  # bottom first
            elif dr < 0:
                objs.sort(key=lambda o: o.bbox[0])    # top first
            elif dc > 0:
                objs.sort(key=lambda o: -o.bbox[3])   # right first
            elif dc < 0:
                objs.sort(key=lambda o: o.bbox[1])    # left first
            
            placed = []
            for obj in objs:
                obstacles = build_obstacle_mask((h, w), placed)
                slid = slide_object(obj, dr, dc, obstacles, (h, w))
                result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                placed.append(slid)
            
            if not np.array_equal(result, out_np):
                all_match = False
                break
        
        if all_match:
            def _apply(inp, _bg=bg, _dr=dr, _dc=dc):
                inp_np = np.array(inp)
                h, w = inp_np.shape
                objs = extract_objects(inp_np, _bg)
                result = np.full_like(inp_np, _bg)
                if _dr > 0:
                    objs.sort(key=lambda o: -o.bbox[2])
                elif _dr < 0:
                    objs.sort(key=lambda o: o.bbox[0])
                elif _dc > 0:
                    objs.sort(key=lambda o: -o.bbox[3])
                elif _dc < 0:
                    objs.sort(key=lambda o: o.bbox[1])
                placed = []
                for obj in objs:
                    obstacles = build_obstacle_mask((h, w), placed)
                    slid = slide_object(obj, _dr, _dc, obstacles, (h, w))
                    result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                    placed.append(slid)
                return result.tolist()
            
            pieces.append(CrossPiece(f'gravity:uniform_{dir_name}', _apply))
    
    return pieces


def try_diagonal_stack(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Try: objects slide diagonally and stack against each other.
    Test all 4 diagonal directions.
    """
    pieces = []
    diags = [('upleft', -1, -1), ('upright', -1, 1), ('downleft', 1, -1), ('downright', 1, 1)]
    
    for dir_name, dr, dc in diags:
        all_match = True
        for inp, out in train_pairs:
            inp_np = np.array(inp)
            out_np = np.array(out)
            h, w = inp_np.shape
            
            objs = extract_objects(inp_np, bg)
            if not objs:
                all_match = False
                break
            
            result = np.full_like(inp_np, bg)
            
            # Sort: process objects closest to target corner first
            if dr < 0 and dc < 0:  # upleft
                objs.sort(key=lambda o: o.centroid[0] + o.centroid[1])
            elif dr < 0 and dc > 0:  # upright
                objs.sort(key=lambda o: o.centroid[0] - o.centroid[1])
            elif dr > 0 and dc < 0:  # downleft
                objs.sort(key=lambda o: -o.centroid[0] + o.centroid[1])
            else:  # downright
                objs.sort(key=lambda o: -(o.centroid[0] + o.centroid[1]))
            
            placed = []
            for obj in objs:
                obstacles = build_obstacle_mask((h, w), placed)
                slid = slide_object(obj, dr, dc, obstacles, (h, w))
                result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                placed.append(slid)
            
            if not np.array_equal(result, out_np):
                all_match = False
                break
        
        if all_match:
            def _apply(inp, _bg=bg, _dr=dr, _dc=dc, _dn=dir_name):
                inp_np = np.array(inp)
                h, w = inp_np.shape
                objs = extract_objects(inp_np, _bg)
                result = np.full_like(inp_np, _bg)
                if _dr < 0 and _dc < 0:
                    objs.sort(key=lambda o: o.centroid[0] + o.centroid[1])
                elif _dr < 0 and _dc > 0:
                    objs.sort(key=lambda o: o.centroid[0] - o.centroid[1])
                elif _dr > 0 and _dc < 0:
                    objs.sort(key=lambda o: -o.centroid[0] + o.centroid[1])
                else:
                    objs.sort(key=lambda o: -(o.centroid[0] + o.centroid[1]))
                placed = []
                for obj in objs:
                    obstacles = build_obstacle_mask((h, w), placed)
                    slid = slide_object(obj, _dr, _dc, obstacles, (h, w))
                    result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                    placed.append(slid)
                return result.tolist()
            
            pieces.append(CrossPiece(f'gravity:diag_stack_{dir_name}', _apply))
    
    return pieces


def try_static_anchor_gravity(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Try: some objects are static (walls/anchors), others slide toward them.
    Detect static objects (same position in input and output) and moving objects.
    Try each direction for moving objects.
    """
    pieces = []
    
    # First, identify which objects are static across all training pairs
    for inp, out in train_pairs[:1]:  # analyze first pair
        inp_np = np.array(inp)
        out_np = np.array(out)
        h, w = inp_np.shape
        
        objs_in = extract_objects(inp_np, bg)
        objs_out = extract_objects(out_np, bg)
        matches = match_objects(objs_in, objs_out)
        
        static_colors = set()
        moving_colors = set()
        for oi, oo in matches:
            dr = abs(oi.centroid[0] - oo.centroid[0])
            dc = abs(oi.centroid[1] - oo.centroid[1])
            if dr < 0.5 and dc < 0.5:
                static_colors.add(oi.color)
            else:
                moving_colors.add(oi.color)
        
        if not moving_colors or not static_colors:
            continue
        
        # Try each direction for moving objects
        for dir_name, (dr, dc) in DIRECTIONS.items():
            all_match = True
            for inp2, out2 in train_pairs:
                inp2_np = np.array(inp2)
                out2_np = np.array(out2)
                h2, w2 = inp2_np.shape
                
                objs = extract_objects(inp2_np, bg)
                result = np.full_like(inp2_np, bg)
                
                # Place static objects first
                static_objs = []
                moving_objs = []
                for obj in objs:
                    if obj.color in static_colors:
                        static_objs.append(obj)
                        result[obj.pixels[:, 0], obj.pixels[:, 1]] = obj.color
                    else:
                        moving_objs.append(obj)
                
                # Sort moving objects by position along slide direction
                if dr > 0:
                    moving_objs.sort(key=lambda o: -o.bbox[2])
                elif dr < 0:
                    moving_objs.sort(key=lambda o: o.bbox[0])
                elif dc > 0:
                    moving_objs.sort(key=lambda o: -o.bbox[3])
                elif dc < 0:
                    moving_objs.sort(key=lambda o: o.bbox[1])
                else:
                    # diagonal
                    if dr != 0 and dc != 0:
                        if dr < 0 and dc < 0:
                            moving_objs.sort(key=lambda o: o.centroid[0] + o.centroid[1])
                        elif dr < 0 and dc > 0:
                            moving_objs.sort(key=lambda o: o.centroid[0] - o.centroid[1])
                        elif dr > 0 and dc < 0:
                            moving_objs.sort(key=lambda o: -o.centroid[0] + o.centroid[1])
                        else:
                            moving_objs.sort(key=lambda o: -(o.centroid[0] + o.centroid[1]))
                
                placed = list(static_objs)
                for obj in moving_objs:
                    obstacles = build_obstacle_mask((h2, w2), placed, exclude=obj)
                    slid = slide_object(obj, dr, dc, obstacles, (h2, w2))
                    result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                    placed.append(slid)
                
                if not np.array_equal(result, out2_np):
                    all_match = False
                    break
            
            if all_match:
                def _apply(inp, _bg=bg, _dr=dr, _dc=dc, _sc=frozenset(static_colors)):
                    inp_np = np.array(inp)
                    h, w = inp_np.shape
                    objs = extract_objects(inp_np, _bg)
                    result = np.full_like(inp_np, _bg)
                    static_objs = []
                    moving_objs = []
                    for obj in objs:
                        if obj.color in _sc:
                            static_objs.append(obj)
                            result[obj.pixels[:, 0], obj.pixels[:, 1]] = obj.color
                        else:
                            moving_objs.append(obj)
                    if _dr > 0:
                        moving_objs.sort(key=lambda o: -o.bbox[2])
                    elif _dr < 0:
                        moving_objs.sort(key=lambda o: o.bbox[0])
                    elif _dc > 0:
                        moving_objs.sort(key=lambda o: -o.bbox[3])
                    elif _dc < 0:
                        moving_objs.sort(key=lambda o: o.bbox[1])
                    placed = list(static_objs)
                    for obj in moving_objs:
                        obstacles = build_obstacle_mask((h, w), placed, exclude=obj)
                        slid = slide_object(obj, _dr, _dc, obstacles, (h, w))
                        result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                        placed.append(slid)
                    return result.tolist()
                
                pieces.append(CrossPiece(f'gravity:anchor_{dir_name}', _apply))
    
    return pieces


def try_per_color_gravity(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Try: each color slides in a different direction.
    Learn per-color direction from first training pair, verify on all.
    """
    pieces = []
    
    inp0, out0 = train_pairs[0]
    inp_np = np.array(inp0)
    out_np = np.array(out0)
    h, w = inp_np.shape
    
    objs_in = extract_objects(inp_np, bg)
    objs_out = extract_objects(out_np, bg)
    matches = match_objects(objs_in, objs_out)
    movements = detect_movements(matches)
    
    if not movements:
        return pieces
    
    # Group movements by color and find dominant direction
    color_dirs: Dict[int, Tuple[int, int]] = {}
    from collections import defaultdict
    color_moves = defaultdict(list)
    for obj, dr, dc in movements:
        color_moves[obj.color].append((dr, dc))
    
    for color, moves in color_moves.items():
        # All objects of this color should move in same direction
        if len(set(moves)) > 1:
            # Try to find a consistent unit direction
            avg_dr = sum(m[0] for m in moves) / len(moves)
            avg_dc = sum(m[1] for m in moves) / len(moves)
            if abs(avg_dr) < 0.5 and abs(avg_dc) < 0.5:
                continue  # static
            dr_unit = 0 if abs(avg_dr) < 0.5 else (1 if avg_dr > 0 else -1)
            dc_unit = 0 if abs(avg_dc) < 0.5 else (1 if avg_dc > 0 else -1)
        else:
            dr, dc = moves[0]
            if dr == 0 and dc == 0:
                continue  # static
            dr_unit = 0 if dr == 0 else (1 if dr > 0 else -1)
            dc_unit = 0 if dc == 0 else (1 if dc > 0 else -1)
        
        color_dirs[color] = (dr_unit, dc_unit)
    
    if not color_dirs:
        return pieces
    
    # Verify on all training pairs
    all_match = True
    for inp, out in train_pairs:
        inp_np = np.array(inp)
        out_np = np.array(out)
        h, w = inp_np.shape
        
        objs = extract_objects(inp_np, bg)
        result = np.full_like(inp_np, bg)
        
        # Place static objects first
        static_objs = []
        moving_groups: Dict[Tuple[int,int], List[GravObj]] = defaultdict(list)
        for obj in objs:
            if obj.color in color_dirs:
                d = color_dirs[obj.color]
                moving_groups[d].append(obj)
            else:
                static_objs.append(obj)
                result[obj.pixels[:, 0], obj.pixels[:, 1]] = obj.color
        
        placed = list(static_objs)
        
        # Process each direction group
        for (dr, dc), group in moving_groups.items():
            # Sort within group
            if dr > 0:
                group.sort(key=lambda o: -o.bbox[2])
            elif dr < 0:
                group.sort(key=lambda o: o.bbox[0])
            elif dc > 0:
                group.sort(key=lambda o: -o.bbox[3])
            elif dc < 0:
                group.sort(key=lambda o: o.bbox[1])
            
            for obj in group:
                obstacles = build_obstacle_mask((h, w), placed, exclude=obj)
                slid = slide_object(obj, dr, dc, obstacles, (h, w))
                result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                placed.append(slid)
        
        if not np.array_equal(result, out_np):
            all_match = False
            break
    
    if all_match and color_dirs:
        def _apply(inp, _bg=bg, _cd=dict(color_dirs)):
            inp_np = np.array(inp)
            h, w = inp_np.shape
            objs = extract_objects(inp_np, _bg)
            result = np.full_like(inp_np, _bg)
            static_objs = []
            moving_groups = defaultdict(list)
            for obj in objs:
                if obj.color in _cd:
                    moving_groups[_cd[obj.color]].append(obj)
                else:
                    static_objs.append(obj)
                    result[obj.pixels[:, 0], obj.pixels[:, 1]] = obj.color
            placed = list(static_objs)
            for (dr, dc), group in moving_groups.items():
                if dr > 0:
                    group.sort(key=lambda o: -o.bbox[2])
                elif dr < 0:
                    group.sort(key=lambda o: o.bbox[0])
                elif dc > 0:
                    group.sort(key=lambda o: -o.bbox[3])
                elif dc < 0:
                    group.sort(key=lambda o: o.bbox[1])
                for obj in group:
                    obstacles = build_obstacle_mask((h, w), placed, exclude=obj)
                    slid = slide_object(obj, dr, dc, obstacles, (h, w))
                    result[slid.pixels[:, 0], slid.pixels[:, 1]] = slid.color
                    placed.append(slid)
            return result.tolist()
        
        dir_desc = '_'.join(f'c{c}{"udlr"[((d[0]+1)//2)*2+(d[1]+1)//2] if d[0] or d[1] else "s"}' 
                           for c, d in sorted(color_dirs.items()))
        pieces.append(CrossPiece(f'gravity:per_color', _apply))
    
    return pieces


# ---- Main entry ----

def generate_gravity_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate gravity/slide transform pieces."""
    pieces = []
    
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    
    # Quick check: same grid size?
    if grid_shape(inp0) != grid_shape(out0):
        return pieces
    
    # Quick check: same colors?
    inp_colors = set(np.array(inp0).flatten()) - {bg}
    out_colors = set(np.array(out0).flatten()) - {bg}
    if inp_colors != out_colors:
        return pieces
    
    # Try strategies in order of simplicity
    pieces.extend(try_uniform_gravity(train_pairs, bg))
    if pieces:
        return pieces
    
    pieces.extend(try_diagonal_stack(train_pairs, bg))
    if pieces:
        return pieces
    
    pieces.extend(try_static_anchor_gravity(train_pairs, bg))
    if pieces:
        return pieces
    
    pieces.extend(try_per_color_gravity(train_pairs, bg))
    
    return pieces
