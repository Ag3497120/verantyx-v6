"""
arc/object_ir.py — Object Intermediate Representation for ARC grids

Provides:
- Connected component detection (4-conn and 8-conn)
- Object attribute table (area, bbox, centroid, shape class, etc.)
- Cell role signatures (is_border, is_interior, distance_to_centroid, etc.)
- Region/topology info (enclosed regions, touching border, etc.)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from collections import Counter, defaultdict


class ArcObject:
    """Represents a connected component in an ARC grid."""
    __slots__ = ['obj_id', 'color', 'cells', 'area',
                 'r_min', 'r_max', 'c_min', 'c_max',
                 'centroid_r', 'centroid_c',
                 'bbox_h', 'bbox_w',
                 'is_rectangular', 'touches_border',
                 'n_holes', 'compactness', 'obj_rank_by_area']
    
    def __init__(self, obj_id: int, color: int, cells: List[Tuple[int, int]],
                 grid_h: int, grid_w: int):
        self.obj_id = obj_id
        self.color = color
        self.cells = cells
        self.area = len(cells)
        
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        self.r_min = min(rs)
        self.r_max = max(rs)
        self.c_min = min(cs)
        self.c_max = max(cs)
        self.bbox_h = self.r_max - self.r_min + 1
        self.bbox_w = self.c_max - self.c_min + 1
        self.centroid_r = sum(rs) / len(rs)
        self.centroid_c = sum(cs) / len(cs)
        
        self.is_rectangular = (self.area == self.bbox_h * self.bbox_w)
        self.touches_border = any(
            r == 0 or r == grid_h - 1 or c == 0 or c == grid_w - 1
            for r, c in cells
        )
        
        # Holes: bg cells inside bbox that aren't in this object
        cell_set = set(cells)
        bbox_cells = self.bbox_h * self.bbox_w
        self.n_holes = bbox_cells - self.area if not self.is_rectangular else 0
        
        # Compactness: area / bbox_area
        self.compactness = self.area / bbox_cells if bbox_cells > 0 else 0
        self.obj_rank_by_area = -1  # set later


class CellRole:
    """Role signature for a single cell in the grid."""
    __slots__ = ['obj_id', 'obj_color', 'obj_area', 'obj_rank_by_area',
                 'is_bg', 'is_border_of_obj', 'is_interior_of_obj',
                 'is_corner_of_obj', 'dist_to_centroid',
                 'dist_to_nearest_other_obj', 'dist_to_grid_border',
                 'n_adjacent_objects', 'enclosed_region_id',
                 'rel_r', 'rel_c']  # position relative to object bbox
    
    def __init__(self):
        self.obj_id = -1
        self.obj_color = 0
        self.obj_area = 0
        self.obj_rank_by_area = -1
        self.is_bg = True
        self.is_border_of_obj = False
        self.is_interior_of_obj = False
        self.is_corner_of_obj = False
        self.dist_to_centroid = 0.0
        self.dist_to_nearest_other_obj = 99
        self.dist_to_grid_border = 0
        self.n_adjacent_objects = 0
        self.enclosed_region_id = -1
        self.rel_r = 0.0
        self.rel_c = 0.0


class ObjectIR:
    """Complete object-level intermediate representation of an ARC grid."""
    
    def __init__(self, grid, bg: int = 0, connectivity: int = 4):
        if isinstance(grid, np.ndarray):
            self.grid = grid
        else:
            self.grid = np.array(grid)
        self.bg = bg
        self.h, self.w = self.grid.shape
        self.connectivity = connectivity
        
        # Detect objects
        self.objects: List[ArcObject] = []
        self.cell_to_obj: np.ndarray = np.full((self.h, self.w), -1, dtype=int)
        self._detect_objects()
        
        # Compute cell roles
        self.cell_roles: List[List[CellRole]] = [[CellRole() for _ in range(self.w)] for _ in range(self.h)]
        self._compute_cell_roles()
        
        # Detect enclosed regions
        self.enclosed_regions: List[List[Tuple[int, int]]] = []
        self._detect_enclosed_regions()
    
    def _detect_objects(self):
        """Detect connected components of non-bg cells."""
        visited = np.zeros((self.h, self.w), dtype=bool)
        obj_id = 0
        
        if self.connectivity == 4:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for sr in range(self.h):
            for sc in range(self.w):
                if visited[sr, sc] or self.grid[sr, sc] == self.bg:
                    continue
                
                color = int(self.grid[sr, sc])
                queue = [(sr, sc)]
                visited[sr, sc] = True
                cells = []
                
                while queue:
                    r, c = queue.pop(0)
                    cells.append((r, c))
                    self.cell_to_obj[r, c] = obj_id
                    
                    for dr, dc in deltas:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.h and 0 <= nc < self.w 
                            and not visited[nr, nc]
                            and self.grid[nr, nc] == color):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                self.objects.append(ArcObject(obj_id, color, cells, self.h, self.w))
                obj_id += 1
        
        # Sort objects by area (descending) and assign ranks
        sorted_by_area = sorted(range(len(self.objects)), 
                                key=lambda i: self.objects[i].area, reverse=True)
        for rank, idx in enumerate(sorted_by_area):
            self.objects[idx].obj_rank_by_area = rank
    
    def _compute_cell_roles(self):
        """Compute role signatures for every cell."""
        # Precompute object border cells
        obj_border_cells = {}
        for obj in self.objects:
            cell_set = set(obj.cells)
            borders = set()
            for r, c in obj.cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.h and 0 <= nc < self.w) or (nr, nc) not in cell_set:
                        borders.add((r, c))
                        break
            obj_border_cells[obj.obj_id] = borders
        
        # Precompute corner cells (border cells at bbox corners)
        obj_corner_cells = {}
        for obj in self.objects:
            corners = set()
            for r, c in obj.cells:
                if ((r == obj.r_min or r == obj.r_max) and 
                    (c == obj.c_min or c == obj.c_max)):
                    corners.add((r, c))
            obj_corner_cells[obj.obj_id] = corners
        
        for r in range(self.h):
            for c in range(self.w):
                role = self.cell_roles[r][c]
                oid = self.cell_to_obj[r, c]
                
                # Distance to grid border
                role.dist_to_grid_border = min(r, self.h - 1 - r, c, self.w - 1 - c)
                
                if oid == -1:
                    # Background cell
                    role.is_bg = True
                    role.obj_id = -1
                    role.obj_color = self.bg
                    
                    # Distance to nearest object
                    min_dist = 99
                    n_adj = 0
                    adj_objs = set()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.h and 0 <= nc < self.w:
                            noid = self.cell_to_obj[nr, nc]
                            if noid >= 0:
                                adj_objs.add(noid)
                                min_dist = 1
                    role.dist_to_nearest_other_obj = min_dist if adj_objs else 99
                    role.n_adjacent_objects = len(adj_objs)
                else:
                    obj = self.objects[oid]
                    role.is_bg = False
                    role.obj_id = oid
                    role.obj_color = obj.color
                    role.obj_area = obj.area
                    role.obj_rank_by_area = obj.obj_rank_by_area
                    role.is_border_of_obj = (r, c) in obj_border_cells[oid]
                    role.is_interior_of_obj = not role.is_border_of_obj
                    role.is_corner_of_obj = (r, c) in obj_corner_cells[oid]
                    
                    # Distance to centroid (normalized by bbox size)
                    dr = abs(r - obj.centroid_r)
                    dc = abs(c - obj.centroid_c)
                    role.dist_to_centroid = (dr + dc) / max(obj.bbox_h + obj.bbox_w, 1)
                    
                    # Relative position in bbox (0.0 to 1.0)
                    role.rel_r = (r - obj.r_min) / max(obj.bbox_h - 1, 1)
                    role.rel_c = (c - obj.c_min) / max(obj.bbox_w - 1, 1)
                    
                    # Adjacent different objects
                    adj_objs = set()
                    for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr2, c + dc2
                        if 0 <= nr < self.h and 0 <= nc < self.w:
                            noid = self.cell_to_obj[nr, nc]
                            if noid >= 0 and noid != oid:
                                adj_objs.add(noid)
                    role.n_adjacent_objects = len(adj_objs)
                    role.dist_to_nearest_other_obj = 0 if adj_objs else 99
    
    def _detect_enclosed_regions(self):
        """Detect bg regions fully enclosed by non-bg cells (not touching grid border)."""
        visited = np.zeros((self.h, self.w), dtype=bool)
        region_id = 0
        
        for sr in range(self.h):
            for sc in range(self.w):
                if visited[sr, sc] or self.grid[sr, sc] != self.bg:
                    continue
                
                queue = [(sr, sc)]
                visited[sr, sc] = True
                cells = []
                touches_border = False
                
                while queue:
                    r, c = queue.pop(0)
                    cells.append((r, c))
                    if r == 0 or r == self.h - 1 or c == 0 or c == self.w - 1:
                        touches_border = True
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.h and 0 <= nc < self.w
                            and not visited[nr, nc]
                            and self.grid[nr, nc] == self.bg):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                
                if not touches_border:
                    self.enclosed_regions.append(cells)
                    for r, c in cells:
                        self.cell_roles[r][c].enclosed_region_id = region_id
                    region_id += 1
    
    def get_role_signature(self, r: int, c: int) -> tuple:
        """Get a hashable role signature for a cell.
        
        Returns tuple of discretized role features suitable for use as dict key.
        """
        role = self.cell_roles[r][c]
        
        # Discretize continuous values
        dist_centroid_bin = min(int(role.dist_to_centroid * 4), 3)  # 0-3
        rel_r_bin = min(int(role.rel_r * 3), 2)  # 0-2
        rel_c_bin = min(int(role.rel_c * 3), 2)  # 0-2
        
        return (
            role.is_bg,
            role.is_border_of_obj,
            role.is_interior_of_obj,
            role.is_corner_of_obj,
            dist_centroid_bin,
            min(role.dist_to_grid_border, 3),  # cap at 3
            min(role.n_adjacent_objects, 3),    # cap at 3
            role.enclosed_region_id >= 0,       # is enclosed
            min(role.obj_area, 100) if not role.is_bg else 0,  # capped area
            role.obj_rank_by_area if not role.is_bg else -1,
        )
    
    def get_compact_role(self, r: int, c: int) -> tuple:
        """Get a more compact (coarser) role signature — better for generalization."""
        role = self.cell_roles[r][c]
        
        # Area class: tiny(1-3), small(4-10), medium(11-50), large(50+)
        if role.is_bg:
            area_class = 0
        elif role.obj_area <= 3:
            area_class = 1
        elif role.obj_area <= 10:
            area_class = 2
        elif role.obj_area <= 50:
            area_class = 3
        else:
            area_class = 4
        
        return (
            role.is_bg,
            role.is_border_of_obj,
            role.is_interior_of_obj,
            min(role.n_adjacent_objects, 2),
            role.enclosed_region_id >= 0,
            area_class,
        )
    
    def get_nb_plus_role(self, r: int, c: int, inp_grid=None) -> tuple:
        """Get combined NB pattern + role signature.
        
        This is the key innovation: structural NB pattern PLUS object-level role.
        """
        if inp_grid is None:
            inp_grid = self.grid
        
        h, w = self.h, self.w
        center = int(inp_grid[r, c]) if isinstance(inp_grid, np.ndarray) else inp_grid[r][c]
        
        # Structural NB (3x3)
        nb = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(inp_grid[nr, nc]) if isinstance(inp_grid, np.ndarray) else inp_grid[nr][nc]
                    if v == self.bg:
                        nb.append(0)
                    elif v == center:
                        nb.append(1)
                    else:
                        nb.append(2)
                else:
                    nb.append(-1)
        
        role = self.get_compact_role(r, c)
        return (tuple(nb), role)


def build_object_ir(grid, bg: int = 0, connectivity: int = 4) -> ObjectIR:
    """Build ObjectIR from a grid."""
    return ObjectIR(grid, bg, connectivity)
