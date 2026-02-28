"""
arc/topology_solver.py — Topology-based solvers using multiple enclosure hypotheses

Tries multiple ways to detect "enclosed" regions:
H1: 4-connected flood fill (strict)
H2: 8-connected flood fill (diagonal walls count)
H3: Wall-distance based (cells far from grid border)
H4: Ignore narrow openings (1-cell gaps)

For each hypothesis, tries common fill rules:
- Fill all enclosed regions with new color
- Fill smallest/largest enclosed region
- Fill based on surrounding object properties
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from collections import Counter
from arc.grid import Grid, grid_shape, most_common_color, grid_eq


def _flood_regions(grid, bg=0, connectivity=4):
    """Find connected bg regions, return (regions, touches_border_flags)."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    regions = []
    
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for sr in range(h):
        for sc in range(w):
            if visited[sr, sc] or grid[sr, sc] != bg:
                continue
            
            queue = [(sr, sc)]
            visited[sr, sc] = True
            cells = []
            touches = False
            
            while queue:
                r, c = queue.pop(0)
                cells.append((r, c))
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches = True
                for dr, dc in deltas:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and grid[nr, nc] == bg:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            
            regions.append({'cells': cells, 'touches_border': touches, 'size': len(cells)})
    
    return regions


def _flood_regions_ignore_gaps(grid, bg=0, max_gap=1):
    """Find enclosed regions, treating narrow gaps (<=max_gap cells) as walls."""
    h, w = grid.shape
    
    # Create "sealed" grid: fill 1-cell-wide openings
    sealed = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg:
                # Check if this is a narrow gap between non-bg cells
                # Horizontal gap: non-bg left and right
                h_sealed = (c > 0 and c < w-1 and grid[r, c-1] != bg and grid[r, c+1] != bg)
                # Vertical gap
                v_sealed = (r > 0 and r < h-1 and grid[r-1, c] != bg and grid[r+1, c] != bg)
                if h_sealed or v_sealed:
                    sealed[r, c] = 99  # temporary wall marker
    
    # Now flood fill on sealed grid, treating 99 as wall
    return _flood_regions_custom(sealed, bg, [99])


def _flood_regions_custom(grid, bg, extra_walls):
    """Flood fill treating bg cells as fillable, extra_walls as barriers."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    wall_set = set(extra_walls)
    regions = []
    
    for sr in range(h):
        for sc in range(w):
            if visited[sr, sc] or grid[sr, sc] != bg:
                continue
            
            queue = [(sr, sc)]
            visited[sr, sc] = True
            cells = []
            touches = False
            
            while queue:
                r, c = queue.pop(0)
                cells.append((r, c))
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    touches = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        v = grid[nr, nc]
                        if v == bg:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            
            regions.append({'cells': cells, 'touches_border': touches, 'size': len(cells)})
    
    return regions


def _try_fill_rule(train_pairs, regions_fn, fill_rule, bg=0):
    """Try a specific (region detection, fill rule) combination on all train pairs.
    
    fill_rule: 'all', 'smallest', 'largest', 'non_largest'
    Returns fill_color if consistent, else None.
    """
    fill_color = None
    
    for inp_grid, out_grid in train_pairs:
        inp = np.array(inp_grid)
        out = np.array(out_grid)
        
        if inp.shape != out.shape:
            return None
        
        diff = inp != out
        if not diff.any():
            continue  # no change this pair
        
        # Check only bg→single_color changes
        cf = set(inp[diff].tolist())
        ct = set(out[diff].tolist())
        if len(ct) != 1:
            return None
        fc = ct.pop()
        if fill_color is None:
            fill_color = fc
        elif fill_color != fc:
            return None
        
        # Get regions
        regions = regions_fn(inp)
        enclosed = [r for r in regions if not r['touches_border']]
        
        if not enclosed:
            return None
        
        # Apply fill rule
        if fill_rule == 'all':
            to_fill = enclosed
        elif fill_rule == 'smallest':
            to_fill = [min(enclosed, key=lambda r: r['size'])]
        elif fill_rule == 'largest':
            to_fill = [max(enclosed, key=lambda r: r['size'])]
        elif fill_rule == 'non_largest':
            if len(enclosed) <= 1:
                return None
            largest = max(enclosed, key=lambda r: r['size'])
            to_fill = [r for r in enclosed if r is not largest]
        else:
            return None
        
        # Build expected output
        result = inp.copy()
        for reg in to_fill:
            for r, c in reg['cells']:
                result[r, c] = fill_color
        
        if not np.array_equal(result, out):
            return None
    
    return fill_color


def generate_topology_pieces(train_pairs, bg=None):
    """Generate CrossPiece candidates using topology-based fill rules."""
    from arc.cross_engine import CrossPiece
    
    if bg is None:
        bg = most_common_color(train_pairs[0][0])
    
    # Check same size
    for inp, out in train_pairs:
        if grid_shape(inp) != grid_shape(out):
            return []
    
    pieces = []
    
    # Define region detection hypotheses
    region_fns = {
        '4conn': lambda g: _flood_regions(g, bg, 4),
        '8conn': lambda g: _flood_regions(g, bg, 8),
        'sealed': lambda g: _flood_regions_ignore_gaps(g, bg, 1),
    }
    
    fill_rules = ['all', 'smallest', 'largest', 'non_largest']
    
    for rname, rfn in region_fns.items():
        for frule in fill_rules:
            fc = _try_fill_rule(train_pairs, rfn, frule, bg)
            if fc is not None:
                # Build apply function
                _rfn = rfn
                _frule = frule
                _fc = fc
                _bg = bg
                
                def make_apply(rfn_, frule_, fc_, bg_):
                    def apply_fn(inp_grid):
                        inp = np.array(inp_grid)
                        regions = rfn_(inp)
                        enclosed = [r for r in regions if not r['touches_border']]
                        
                        if not enclosed:
                            return inp.tolist()
                        
                        if frule_ == 'all':
                            to_fill = enclosed
                        elif frule_ == 'smallest':
                            to_fill = [min(enclosed, key=lambda r: r['size'])]
                        elif frule_ == 'largest':
                            to_fill = [max(enclosed, key=lambda r: r['size'])]
                        elif frule_ == 'non_largest':
                            largest = max(enclosed, key=lambda r: r['size'])
                            to_fill = [r for r in enclosed if r is not largest]
                        else:
                            to_fill = enclosed
                        
                        out = inp.copy()
                        for reg in to_fill:
                            for r, c in reg['cells']:
                                out[r, c] = fc_
                        return out.tolist()
                    return apply_fn
                
                apply_fn = make_apply(_rfn, _frule, _fc, _bg)
                
                # Final verify
                ok = True
                for inp_grid, out_grid in train_pairs:
                    if apply_fn(inp_grid) != out_grid:
                        ok = False
                        break
                
                if ok:
                    pieces.append(CrossPiece(
                        name=f"topo_fill:{rname}_{frule}_c{fc}",
                        apply_fn=apply_fn,
                        version=1
                    ))
    
    return pieces
