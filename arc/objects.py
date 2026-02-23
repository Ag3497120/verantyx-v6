"""
arc/objects.py — Object-level detection and operations for ARC-AGI-2

Addresses Wall 1: Object recognition (453 unsolved tasks, 49%)

Objects are connected regions of non-background cells.
Each object has: color, cells, bbox, size, shape (relative coords).
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
from dataclasses import dataclass, field
from arc.grid import Grid, grid_shape, most_common_color


@dataclass
class ArcObject:
    """A connected region in an ARC grid"""
    color: int
    cells: List[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]  # r1, c1, r2, c2
    
    @property
    def size(self) -> int:
        return len(self.cells)
    
    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1
    
    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
    
    @property
    def shape(self) -> Tuple[Tuple[int, int], ...]:
        """Relative coordinates (translation-invariant)"""
        r0, c0 = self.bbox[0], self.bbox[1]
        return tuple(sorted((r - r0, c - c0) for r, c in self.cells))
    
    def as_grid(self, bg: int = 0) -> Grid:
        """Extract object as a small grid"""
        r1, c1, r2, c2 = self.bbox
        h, w = r2 - r1 + 1, c2 - c1 + 1
        g = [[bg] * w for _ in range(h)]
        for r, c in self.cells:
            g[r - r1][c - c1] = self.color
        return g
    
    def as_multicolor_grid(self, source: Grid, bg: int = 0) -> Grid:
        """Extract object bbox from source grid"""
        r1, c1, r2, c2 = self.bbox
        return [source[r][c1:c2+1] for r in range(r1, r2+1)]


def detect_objects(g: Grid, bg: Optional[int] = None, 
                   multicolor: bool = False) -> List[ArcObject]:
    """Detect connected objects in grid via flood fill"""
    h, w = grid_shape(g)
    if h == 0:
        return []
    if bg is None:
        bg = most_common_color(g)
    
    visited = [[False] * w for _ in range(h)]
    objects = []
    
    for r in range(h):
        for c in range(w):
            if visited[r][c] or g[r][c] == bg:
                continue
            # BFS flood fill
            if multicolor:
                # Multi-color object: any non-bg connected cells
                cells = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                color = g[r][c]  # primary color
            else:
                # Single-color object
                color = g[r][c]
                cells = []
                queue = [(r, c)]
                visited[r][c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and g[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
            
            r1 = min(r for r, c in cells)
            c1 = min(c for r, c in cells)
            r2 = max(r for r, c in cells)
            c2 = max(c for r, c in cells)
            objects.append(ArcObject(color=color, cells=cells, bbox=(r1, c1, r2, c2)))
    
    return sorted(objects, key=lambda o: o.size, reverse=True)


def detect_objects_multicolor(g: Grid, bg: Optional[int] = None) -> List[ArcObject]:
    """Detect multi-color connected objects"""
    return detect_objects(g, bg, multicolor=True)


def find_matching_objects(objs_in: List[ArcObject], objs_out: List[ArcObject]) -> List[Tuple[ArcObject, ArcObject]]:
    """Match input objects to output objects by shape or position"""
    matches = []
    used_out = set()
    
    # Match by same position (bbox overlap)
    for oi in objs_in:
        best = None
        best_overlap = 0
        for j, oo in enumerate(objs_out):
            if j in used_out:
                continue
            # Compute overlap
            cells_i = set(oi.cells)
            cells_o = set(oo.cells)
            overlap = len(cells_i & cells_o)
            if overlap > best_overlap:
                best_overlap = overlap
                best = j
        if best is not None and best_overlap > 0:
            matches.append((oi, objs_out[best]))
            used_out.add(best)
    
    return matches


def object_transform_type(obj_in: ArcObject, obj_out: ArcObject, 
                          inp: Grid, out: Grid, bg: int) -> Optional[str]:
    """Detect what transform was applied to an object"""
    # Same position, different color
    if obj_in.shape == obj_out.shape and obj_in.bbox == obj_out.bbox:
        if obj_in.color != obj_out.color:
            return 'recolor'
        return 'identity'
    
    # Same shape, moved
    if obj_in.shape == obj_out.shape and obj_in.bbox != obj_out.bbox:
        return 'move'
    
    # Scaled
    if (obj_out.height == obj_in.height * 2 and obj_out.width == obj_in.width * 2):
        return 'scale_2x'
    
    # Removed (output object is empty or doesn't exist)
    if obj_out.size == 0:
        return 'remove'
    
    return None


# === Object-level grid operations ===

def move_object(g: Grid, obj: ArcObject, dr: int, dc: int, bg: int) -> Grid:
    """Move an object by (dr, dc)"""
    h, w = grid_shape(g)
    result = [row[:] for row in g]
    # Clear old position
    for r, c in obj.cells:
        result[r][c] = bg
    # Place at new position
    for r, c in obj.cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result[nr][nc] = obj.color
    return result


def copy_object(g: Grid, obj: ArcObject, dr: int, dc: int) -> Grid:
    """Copy an object to (dr, dc) offset"""
    h, w = grid_shape(g)
    result = [row[:] for row in g]
    for r, c in obj.cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            result[nr][nc] = obj.color
    return result


def remove_object(g: Grid, obj: ArcObject, bg: int) -> Grid:
    """Remove an object (fill with bg)"""
    result = [row[:] for row in g]
    for r, c in obj.cells:
        result[r][c] = bg
    return result


def recolor_object(g: Grid, obj: ArcObject, new_color: int) -> Grid:
    """Change object color"""
    result = [row[:] for row in g]
    for r, c in obj.cells:
        result[r][c] = new_color
    return result


def fill_object_bbox(g: Grid, obj: ArcObject, fill_color: int) -> Grid:
    """Fill object's bounding box with a color"""
    result = [row[:] for row in g]
    r1, c1, r2, c2 = obj.bbox
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            result[r][c] = fill_color
    return result


def scale_object(g: Grid, obj: ArcObject, factor: int, bg: int) -> Grid:
    """Scale an object by factor, keeping position"""
    h, w = grid_shape(g)
    result = [row[:] for row in g]
    r0, c0 = obj.bbox[0], obj.bbox[1]
    for r, c in obj.cells:
        rr, cc = r - r0, c - c0
        for dr in range(factor):
            for dc in range(factor):
                nr, nc = r0 + rr * factor + dr, c0 + cc * factor + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr][nc] = obj.color
    return result


def stamp_pattern(g: Grid, pattern: Grid, r0: int, c0: int, bg: int = 0) -> Grid:
    """Stamp a pattern grid onto g at position (r0, c0)"""
    h, w = grid_shape(g)
    ph, pw = grid_shape(pattern)
    result = [row[:] for row in g]
    for r in range(ph):
        for c in range(pw):
            if pattern[r][c] != bg:
                nr, nc = r0 + r, c0 + c
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr][nc] = pattern[r][c]
    return result


def extract_pattern_at_dots(g: Grid, dot_color: int, pattern: Grid, bg: int) -> Grid:
    """Place pattern at every occurrence of dot_color"""
    h, w = grid_shape(g)
    result = [row[:] for row in g]
    ph, pw = grid_shape(pattern)
    pr, pc = ph // 2, pw // 2  # center
    for r in range(h):
        for c in range(w):
            if g[r][c] == dot_color:
                result = stamp_pattern(result, pattern, r - pr, c - pc, bg)
    return result
