"""
arc/cross_solver.py — Verantyx Cross-Structure ARC Solver

No hardcoded patterns. Instead:
1. Decompose input→output relationship into cell-level rules (ピース分解)
2. Synthesize candidate rules from a DSL of grid operations
3. Verify candidates against ALL training pairs (CEGIS)
4. Apply verified rules to test inputs

DSL (Domain Specific Language) for cell rules:
  - copy(dr, dc)           → output[r][c] = input[r+dr][c+dc]
  - const(v)               → output[r][c] = v
  - colormap(mapping)      → output[r][c] = mapping[input[r][c]]
  - cond(color, then, else) → if input[r][c]==color then X else Y
  - neighbor_count(color)   → count of 'color' in 4-neighbors
  - mirror_h()             → output[r][c] = input[r][W-1-c]
  - mirror_v()             → output[r][c] = input[H-1-r][c]
  - mod_copy(mh, mw)       → output[r][c] = input[r%mh][c%mw]  (tiling)
  - region_color()         → color of the region containing (r,c)
"""

from __future__ import annotations
import json
from typing import List, Optional, Tuple, Dict, Callable
from collections import Counter
from dataclasses import dataclass, field
from arc.grid import (
    Grid, grid_shape, grid_eq, grid_colors,
    most_common_color, flood_fill_regions,
    flip_h, flip_v,
    rotate_90, rotate_180, rotate_270, transpose,
)


@dataclass
class CellRule:
    """A rule that computes output[r][c] from input grid"""
    name: str
    params: dict = field(default_factory=dict)
    
    def apply(self, inp: Grid, r: int, c: int, out_h: int, out_w: int) -> Optional[int]:
        """Compute output value for cell (r,c). Returns None if rule doesn't apply."""
        ih, iw = grid_shape(inp)
        
        if self.name == 'copy':
            dr, dc = self.params.get('dr', 0), self.params.get('dc', 0)
            sr, sc = r + dr, c + dc
            if 0 <= sr < ih and 0 <= sc < iw:
                return inp[sr][sc]
                
        elif self.name == 'const':
            return self.params['v']
            
        elif self.name == 'colormap':
            mapping = self.params['map']
            if 0 <= r < ih and 0 <= c < iw:
                return mapping.get(inp[r][c], inp[r][c])
                
        elif self.name == 'mirror_h':
            sc = out_w - 1 - c
            if 0 <= r < ih and 0 <= sc < iw:
                return inp[r][sc]
                
        elif self.name == 'mirror_v':
            sr = out_h - 1 - r
            if 0 <= sr < ih and 0 <= c < iw:
                return inp[sr][c]
                
        elif self.name == 'mirror_hv':
            sr, sc = out_h - 1 - r, out_w - 1 - c
            if 0 <= sr < ih and 0 <= sc < iw:
                return inp[sr][sc]
        
        elif self.name == 'transpose':
            if 0 <= c < ih and 0 <= r < iw:
                return inp[c][r]
                
        elif self.name == 'mod_copy':
            mh, mw = self.params['mh'], self.params['mw']
            sr, sc = r % mh, c % mw
            if 0 <= sr < ih and 0 <= sc < iw:
                return inp[sr][sc]
        
        elif self.name == 'mod_copy_flip':
            # Tiling with alternating row flips
            mh, mw = self.params['mh'], self.params['mw']
            block_r = r // mh
            sr, sc = r % mh, c % mw
            if block_r % 2 == 1:
                sc = mw - 1 - sc  # flip horizontally in odd blocks
            if 0 <= sr < ih and 0 <= sc < iw:
                return inp[sr][sc]
        
        elif self.name == 'scale':
            factor = self.params['factor']
            sr, sc = r // factor, c // factor
            if 0 <= sr < ih and 0 <= sc < iw:
                return inp[sr][sc]
                
        elif self.name == 'cond_replace':
            old_c, new_c = self.params['old'], self.params['new']
            if 0 <= r < ih and 0 <= c < iw:
                return new_c if inp[r][c] == old_c else inp[r][c]
        
        elif self.name == 'cond_neighbor':
            # If cell has specific neighbor, change color
            old_c = self.params['old']
            new_c = self.params['new']
            check_c = self.params['neighbor']
            if 0 <= r < ih and 0 <= c < iw:
                if inp[r][c] != old_c:
                    return inp[r][c]
                has_nb = False
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < ih and 0 <= nc < iw and inp[nr][nc] == check_c:
                        has_nb = True
                        break
                return new_c if has_nb else inp[r][c]
        
        elif self.name == 'gravity':
            # For gravity, need whole-column context
            color = self.params['color']
            bg = self.params['bg']
            if 0 <= c < iw:
                col_count = sum(1 for rr in range(ih) if inp[rr][c] == color)
                threshold = ih - col_count
                if r >= threshold:
                    return color
                else:
                    return bg if (0 <= r < ih and inp[r][c] == color) else (inp[r][c] if 0 <= r < ih else bg)
        
        elif self.name == 'symmetry_fill':
            # Fill background to achieve symmetry
            axis = self.params.get('axis', 'h')  # h or v
            bg = self.params['bg']
            if 0 <= r < ih and 0 <= c < iw:
                if inp[r][c] != bg:
                    return inp[r][c]
                if axis == 'h':
                    mc = iw - 1 - c
                    if 0 <= mc < iw and inp[r][mc] != bg:
                        return inp[r][mc]
                elif axis == 'v':
                    mr = ih - 1 - r
                    if 0 <= mr < ih and inp[mr][c] != bg:
                        return inp[mr][c]
                elif axis == 'hv':
                    mr, mc = ih - 1 - r, iw - 1 - c
                    if 0 <= mr < ih and 0 <= mc < iw and inp[mr][mc] != bg:
                        return inp[mr][mc]
                return inp[r][c]
        
        elif self.name == 'extract_at':
            sr, sc = self.params['sr'], self.params['sc']
            ir, ic = sr + r, sc + c
            if 0 <= ir < ih and 0 <= ic < iw:
                return inp[ir][ic]
        
        elif self.name == 'self_tile':
            # Input as its own tiling mask: out[r][c] = non_bg if BOTH input[r%h][c%w] and input[r//h][c//w] are non_bg
            bg = self.params['bg']
            non_bg = self.params.get('non_bg', None)
            sr, sc = r % ih, c % iw     # position within tile
            br, bc = r // ih, c // iw   # which block
            if 0 <= br < ih and 0 <= bc < iw:
                tile_val = inp[sr][sc]
                mask_val = inp[br][bc]
                if tile_val != bg and mask_val != bg:
                    return tile_val if non_bg is None else non_bg
                else:
                    return bg
            return bg
        
        elif self.name == 'fill_enclosed':
            # Fill bg cells enclosed by wall color — only cells surrounded by wall on all 4-connected paths
            wall = self.params['wall']
            fill_c = self.params['fill']
            bg = self.params['bg']
            if 0 <= r < ih and 0 <= c < iw:
                if inp[r][c] != bg:
                    return inp[r][c]
                # Use cached enclosed regions per input grid (keyed by id)
                cache_key = '_enc_' + str(id(inp))
                enclosed = self.params.get(cache_key)
                if enclosed is None:
                    enclosed = _compute_enclosed_by_wall(inp, bg, wall)
                    self.params[cache_key] = enclosed
                if (r, c) in enclosed:
                    return fill_c
                return inp[r][c]
        
        elif self.name == 'periodic_extend':
            # Extend input pattern periodically to larger output
            period_h = self.params.get('period_h', ih)
            period_w = self.params.get('period_w', iw)
            sr, sc = r % period_h, c % period_w
            if 0 <= sr < ih and 0 <= sc < iw:
                return inp[sr][sc]
            return 0
        
        elif self.name == 'periodic_extend_recolor':
            # Extend + recolor
            period_h = self.params.get('period_h', ih)
            period_w = self.params.get('period_w', iw)
            cmap = self.params['map']
            sr, sc = r % period_h, c % period_w
            if 0 <= sr < ih and 0 <= sc < iw:
                return cmap.get(inp[sr][sc], inp[sr][sc])
            return 0
        
        elif self.name == 'diagonal_tile':
            # Fill grid with repeating diagonal pattern extracted from input's non-bg cells
            bg = self.params.get('bg', 0)
            n = self.params.get('n', 3)
            # Extract color mapping: for each (r+c)%n, find the non-bg color in input
            cache_key = '_dtcache_' + str(id(inp))
            colors = self.params.get(cache_key)
            if colors is None:
                pattern = {}
                for rr in range(ih):
                    for cc in range(iw):
                        if inp[rr][cc] != bg:
                            key = (rr + cc) % n
                            if key not in pattern:
                                pattern[key] = inp[rr][cc]
                if len(pattern) == n:
                    colors = [pattern[i] for i in range(n)]
                    self.params[cache_key] = colors
                else:
                    self.params[cache_key] = []
                    return None
            if colors:
                return colors[(r + c) % n]
            return None
        
        elif self.name == 'move_object':
            # Move object of move_color toward anchor_color
            move_c = self.params['move_color']
            anchor_c = self.params['anchor_color']
            bg = self.params['bg']
            dx, dy = self.params['dx'], self.params['dy']
            if 0 <= r < ih and 0 <= c < iw:
                # If this cell is anchor, keep
                if inp[r][c] == anchor_c:
                    return anchor_c
                # If source cell (before move) had move_color
                sr, sc = r - dy, c - dx
                if 0 <= sr < ih and 0 <= sc < iw and inp[sr][sc] == move_c:
                    return move_c
                # If original position had move_color but object moved away
                if inp[r][c] == move_c:
                    return bg
                return inp[r][c]
        
        elif self.name == 'border_color':
            # Color cells on the border of regions
            bg = self.params['bg']
            border_c = self.params['color']
            if 0 <= r < ih and 0 <= c < iw:
                if inp[r][c] == bg:
                    return bg
                # Check if any neighbor is bg
                for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr2, c+dc2
                    if nr < 0 or nr >= ih or nc < 0 or nc >= iw or inp[nr][nc] == bg:
                        return border_c
                return inp[r][c]
        
        elif self.name == 'flood_color':
            # Change all cells of one connected region to a new color
            old_c = self.params['old']
            new_c = self.params['new']
            if 0 <= r < ih and 0 <= c < iw:
                return new_c if inp[r][c] == old_c else inp[r][c]
        
        return None
    
    def __repr__(self):
        return f"CellRule({self.name}, {self.params})"


@dataclass
class SynthesizedProgram:
    """A complete program = output shape + cell rule"""
    out_h: int
    out_w: int
    rule: CellRule
    confidence: float = 0.0
    
    def apply(self, inp: Grid) -> Optional[Grid]:
        """Apply program to generate output grid"""
        ih, iw = grid_shape(inp)
        
        # Dynamic output size
        oh, ow = self._resolve_size(inp)
        if oh <= 0 or ow <= 0:
            return None
        
        result = []
        for r in range(oh):
            row = []
            for c in range(ow):
                val = self.rule.apply(inp, r, c, oh, ow)
                if val is None:
                    return None  # Rule failed
                row.append(val)
            result.append(row)
        return result
    
    def _resolve_size(self, inp: Grid) -> Tuple[int, int]:
        ih, iw = grid_shape(inp)
        oh = self.out_h if self.out_h > 0 else ih
        ow = self.out_w if self.out_w > 0 else iw
        # Symbolic sizes
        if self.out_h == -1:  # same as input
            oh = ih
        if self.out_w == -1:
            ow = iw
        if self.out_h == -2:  # swap (transpose)
            oh = iw
        if self.out_w == -2:
            ow = ih
        return oh, ow


def _compute_enclosed_by_wall(g: Grid, bg: int, wall: int) -> set:
    """Find bg cells that are completely enclosed by wall-colored cells (not just unreachable from border)"""
    h, w = grid_shape(g)
    # BFS from all border bg cells, but ONLY through bg cells (wall blocks passage)
    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and g[r][c] == bg:
                reachable.add((r, c))
                queue.append((r, c))
    
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable:
                if g[nr][nc] == bg:  # can only pass through bg, wall blocks
                    reachable.add((nr, nc))
                    queue.append((nr, nc))
    
    # Enclosed = ALL bg cells NOT reachable from border
    enclosed = set()
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg and (r, c) not in reachable:
                enclosed.add((r, c))
    return enclosed


def _compute_enclosed(g: Grid, bg: int, wall: int) -> set:
    """Find bg cells enclosed by wall color (can't reach grid border via bg)"""
    h, w = grid_shape(g)
    # BFS from all border bg cells
    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h-1 or c == 0 or c == w-1) and g[r][c] == bg:
                reachable.add((r, c))
                queue.append((r, c))
    
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and g[nr][nc] == bg:
                reachable.add((nr, nc))
                queue.append((nr, nc))
    
    # Enclosed = bg cells NOT reachable from border
    enclosed = set()
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg and (r, c) not in reachable:
                enclosed.add((r, c))
    return enclosed


def generate_candidates(train_pairs: List[Tuple[Grid, Grid]]) -> List[SynthesizedProgram]:
    """
    Generate candidate programs from training pairs.
    This is the "hypothesis generation" step of CEGIS.
    """
    candidates = []
    
    if not train_pairs:
        return candidates
    
    inp0, out0 = train_pairs[0]
    ih, iw = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    
    # Check if all pairs have same size relationship
    size_consistent = all(
        grid_shape(o) == (oh, ow) and grid_shape(i) == (ih, iw)
        for i, o in train_pairs
    )
    
    if not size_consistent:
        # Try relative size relationships
        ratios = set()
        for i, o in train_pairs:
            ish, isw = grid_shape(i)
            osh, osw = grid_shape(o)
            if ish > 0 and isw > 0:
                ratios.add((osh / ish, osw / isw))
        
        if len(ratios) == 1:
            rh, rw = ratios.pop()
            if rh == int(rh) and rw == int(rw):
                oh_rel, ow_rel = int(rh), int(rw)
                if oh_rel == 1 and ow_rel == 1:
                    # Same size but variable input dimensions
                    _add_same_size_candidates(candidates, train_pairs, ih, iw)
                    _add_fill_enclosed_candidates(candidates, train_pairs, ih, iw)
                    _add_diagonal_candidates(candidates, train_pairs, ih, iw)
                    _add_move_candidates(candidates, train_pairs, ih, iw)
                elif oh_rel > 1 and ow_rel > 1:
                    _add_tiling_candidates(candidates, train_pairs, oh_rel, ow_rel)
                    _add_scaling_candidates(candidates, train_pairs, oh_rel)
            size_consistent = True  # We handle it via relative sizing
    
    if not size_consistent:
        return candidates
    
    # ── Same size transforms ──
    if (ih, iw) == (oh, ow):
        _add_same_size_candidates(candidates, train_pairs, ih, iw)
    
    # ── Transposed size ──
    if (ih, iw) == (ow, oh) and ih != iw:
        candidates.append(SynthesizedProgram(-2, -2, CellRule('transpose')))
    
    # ── Smaller output (extraction) ──
    if oh <= ih and ow <= iw and (oh < ih or ow < iw):
        _add_extraction_candidates(candidates, train_pairs, ih, iw, oh, ow)
    
    # ── Larger output (tiling/scaling) ──
    if oh > ih or ow > iw:
        if ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0:
            rh, rw = oh // ih, ow // iw
            _add_tiling_candidates(candidates, train_pairs, rh, rw)
            if rh == rw and rh > 1:
                _add_scaling_candidates(candidates, train_pairs, rh)
    
    # ── Self-tile (input as mask for its own tiling) ──
    if ih > 0 and iw > 0 and oh % ih == 0 and ow % iw == 0:
        rh, rw = oh // ih, ow // iw
        if rh == ih and rw == iw:  # output is input_h * input_h
            for bg_c in grid_colors(inp0):
                candidates.append(SynthesizedProgram(oh, ow, CellRule('self_tile', {'bg': bg_c})))
    
    # ── Fill enclosed regions ──
    if (ih, iw) == (oh, ow):
        _add_fill_enclosed_candidates(candidates, train_pairs, ih, iw)
    
    # ── Diagonal tile ──
    if (ih, iw) == (oh, ow):
        _add_diagonal_candidates(candidates, train_pairs, ih, iw)
    
    # ── Move object ──
    if (ih, iw) == (oh, ow):
        _add_move_candidates(candidates, train_pairs, ih, iw)
    
    # ── Periodic extend (different sizes) ──
    if oh != ih or ow != iw:
        _add_periodic_extend_candidates(candidates, train_pairs, ih, iw, oh, ow)
    
    # ── Border color ──
    if (ih, iw) == (oh, ow):
        bg = most_common_color(inp0)
        for color in range(10):
            if color != bg:
                candidates.append(SynthesizedProgram(-1, -1,
                    CellRule('border_color', {'bg': bg, 'color': color})))
    
    return candidates


def _add_same_size_candidates(candidates, train_pairs, h, w):
    """Generate candidates for same-size input→output"""
    inp0, out0 = train_pairs[0]
    
    # Identity
    candidates.append(SynthesizedProgram(-1, -1, CellRule('copy', {'dr': 0, 'dc': 0})))
    
    # Mirror
    candidates.append(SynthesizedProgram(-1, -1, CellRule('mirror_h')))
    candidates.append(SynthesizedProgram(-1, -1, CellRule('mirror_v')))
    candidates.append(SynthesizedProgram(-1, -1, CellRule('mirror_hv')))
    
    # Color mapping: infer from ALL train pairs
    cmap = {}
    consistent = True
    for inp_i, out_i in train_pairs:
        hi, wi = grid_shape(inp_i)
        for r in range(hi):
            for c in range(wi):
                ic = inp_i[r][c]
                oc = out_i[r][c]
                if ic in cmap:
                    if cmap[ic] != oc:
                        consistent = False
                        break
                else:
                    cmap[ic] = oc
            if not consistent:
                break
        if not consistent:
            break
    
    if consistent and any(k != v for k, v in cmap.items()):
        candidates.append(SynthesizedProgram(-1, -1, CellRule('colormap', {'map': cmap})))
    
    # Color replacement: find changed cells
    bg = most_common_color(inp0)
    changed = {}
    for r in range(h):
        for c in range(w):
            if inp0[r][c] != out0[r][c]:
                key = (inp0[r][c], out0[r][c])
                changed[key] = changed.get(key, 0) + 1
    
    for (old_c, new_c), count in changed.items():
        candidates.append(SynthesizedProgram(-1, -1, CellRule('cond_replace', {'old': old_c, 'new': new_c})))
    
    # Neighbor-conditional
    for (old_c, new_c), _ in changed.items():
        for nb_color in range(10):
            candidates.append(SynthesizedProgram(-1, -1, 
                CellRule('cond_neighbor', {'old': old_c, 'new': new_c, 'neighbor': nb_color})))
    
    # Gravity (per color)
    colors = grid_colors(inp0) - {bg}
    for color in colors:
        candidates.append(SynthesizedProgram(-1, -1,
            CellRule('gravity', {'color': color, 'bg': bg})))
    
    # Symmetry fill
    for axis in ['h', 'v', 'hv']:
        candidates.append(SynthesizedProgram(-1, -1,
            CellRule('symmetry_fill', {'bg': bg, 'axis': axis})))


def _add_tiling_candidates(candidates, train_pairs, rh, rw):
    """Generate tiling candidates"""
    inp0, _ = train_pairs[0]
    ih, iw = grid_shape(inp0)
    
    candidates.append(SynthesizedProgram(
        ih * rh, iw * rw,
        CellRule('mod_copy', {'mh': ih, 'mw': iw})
    ))
    candidates.append(SynthesizedProgram(
        ih * rh, iw * rw,
        CellRule('mod_copy_flip', {'mh': ih, 'mw': iw})
    ))


def _add_scaling_candidates(candidates, train_pairs, factor):
    """Generate scaling candidates"""
    inp0, _ = train_pairs[0]
    ih, iw = grid_shape(inp0)
    candidates.append(SynthesizedProgram(
        ih * factor, iw * factor,
        CellRule('scale', {'factor': factor})
    ))


def _add_extraction_candidates(candidates, train_pairs, ih, iw, oh, ow):
    """Generate extraction candidates"""
    # Try all offsets
    for sr in range(ih - oh + 1):
        for sc in range(iw - ow + 1):
            candidates.append(SynthesizedProgram(
                oh, ow,
                CellRule('extract_at', {'sr': sr, 'sc': sc})
            ))
    
    # Region-based extraction: extract bounding box of specific color
    inp0, out0 = train_pairs[0]
    regions = flood_fill_regions(inp0)
    for reg in regions[:10]:
        r1, c1, r2, c2 = reg['bbox']
        bh, bw = r2 - r1 + 1, c2 - c1 + 1
        if (bh, bw) == (oh, ow):
            candidates.append(SynthesizedProgram(
                oh, ow,
                CellRule('extract_at', {'sr': r1, 'sc': c1})
            ))


def _add_fill_enclosed_candidates(candidates, train_pairs, h, w):
    """Fill bg cells enclosed by walls"""
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    
    # Find what color was filled in
    for r in range(h):
        for c in range(w):
            if inp0[r][c] == bg and out0[r][c] != bg:
                fill_c = out0[r][c]
                # Find the wall color (neighbors of changed cells)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and inp0[nr][nc] != bg:
                        wall_c = inp0[nr][nc]
                        candidates.append(SynthesizedProgram(-1, -1,
                            CellRule('fill_enclosed', {'wall': wall_c, 'fill': fill_c, 'bg': bg})))
                        return  # One candidate per fill color


def _add_diagonal_candidates(candidates, train_pairs, h, w):
    """Detect diagonal repeating patterns — colors extracted from input at runtime"""
    inp0, out0 = train_pairs[0]
    
    # Check if output has diagonal pattern: out[r][c] depends on (r+c) % n
    for bg_c in grid_colors(inp0):
        for n in range(2, 6):
            pattern = {}
            consistent = True
            for r in range(h):
                for c in range(w):
                    key = (r + c) % n
                    if key in pattern:
                        if pattern[key] != out0[r][c]:
                            consistent = False
                            break
                    else:
                        pattern[key] = out0[r][c]
                if not consistent:
                    break
            
            if consistent and pattern:
                candidates.append(SynthesizedProgram(-1, -1,
                    CellRule('diagonal_tile', {'bg': bg_c, 'n': n})))


def _add_move_candidates(candidates, train_pairs, h, w):
    """Detect object movement"""
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    colors = grid_colors(inp0) - {bg}
    
    if len(colors) != 2:
        return
    
    colors = list(colors)
    for move_c, anchor_c in [(colors[0], colors[1]), (colors[1], colors[0])]:
        # Find center of mass of move_color in input and output
        in_cells = [(r, c) for r in range(h) for c in range(w) if inp0[r][c] == move_c]
        out_cells = [(r, c) for r in range(h) for c in range(w) if out0[r][c] == move_c]
        
        if in_cells and out_cells and len(in_cells) == len(out_cells):
            # Compute displacement
            in_cr = sum(r for r, c in in_cells) / len(in_cells)
            in_cc = sum(c for r, c in in_cells) / len(in_cells)
            out_cr = sum(r for r, c in out_cells) / len(out_cells)
            out_cc = sum(c for r, c in out_cells) / len(out_cells)
            
            dy = round(out_cr - in_cr)
            dx = round(out_cc - in_cc)
            
            if dx != 0 or dy != 0:
                candidates.append(SynthesizedProgram(-1, -1,
                    CellRule('move_object', {
                        'move_color': move_c, 'anchor_color': anchor_c,
                        'bg': bg, 'dx': dx, 'dy': dy
                    })))


def _add_periodic_extend_candidates(candidates, train_pairs, ih, iw, oh, ow):
    """Periodic extension to different sizes"""
    inp0, out0 = train_pairs[0]
    
    # Try various period sizes
    for ph in range(1, ih + 1):
        if oh % ph == 0 or ph == ih:
            for pw in range(1, iw + 1):
                if ow % pw == 0 or pw == iw:
                    candidates.append(SynthesizedProgram(oh, ow,
                        CellRule('periodic_extend', {'period_h': ph, 'period_w': pw})))
    
    # Periodic extend with recolor
    bg = most_common_color(inp0)
    in_colors = grid_colors(inp0) - {bg}
    out_colors = grid_colors(out0) - {bg if bg in grid_colors(out0) else -1}
    
    if len(in_colors) == len(out_colors) and in_colors != out_colors:
        # Try to infer color mapping
        cmap = {}
        for r in range(min(ih, oh)):
            for c in range(min(iw, ow)):
                if inp0[r][c] != bg:
                    ic = inp0[r][c]
                    oc = out0[r][c]
                    if ic in cmap:
                        if cmap[ic] != oc:
                            cmap = None
                            break
                    else:
                        cmap[ic] = oc
            if cmap is None:
                break
        
        if cmap:
            cmap[bg] = bg
            for ph in [ih]:
                for pw in [iw]:
                    candidates.append(SynthesizedProgram(oh, ow,
                        CellRule('periodic_extend_recolor', {
                            'period_h': ph, 'period_w': pw, 'map': cmap
                        })))


@dataclass
class CompositeProgram:
    """Two-step pipeline: apply step1, then step2 to the result"""
    step1: SynthesizedProgram
    step2: SynthesizedProgram
    
    def apply(self, inp: Grid) -> Optional[Grid]:
        mid = self.step1.apply(inp)
        if mid is None:
            return None
        return self.step2.apply(mid)


@dataclass
class WholeGridProgram:
    """Program that operates on the whole grid (not cell-by-cell)"""
    name: str
    params: dict = field(default_factory=dict)
    
    def apply(self, inp: Grid) -> Optional[Grid]:
        h, w = grid_shape(inp)
        
        if self.name == 'row_sort':
            key = self.params.get('key', 'color_count')
            bg = self.params.get('bg', 0)
            rows = list(inp)
            if key == 'color_count':
                rows.sort(key=lambda row: sum(1 for x in row if x != bg))
            elif key == 'color_count_desc':
                rows.sort(key=lambda row: sum(1 for x in row if x != bg), reverse=True)
            elif key == 'first_nonbg':
                rows.sort(key=lambda row: next((c for c in row if c != bg), 0))
            elif key == 'sum':
                rows.sort(key=sum)
            elif key == 'sum_desc':
                rows.sort(key=sum, reverse=True)
            return rows
        
        elif self.name == 'col_sort':
            key = self.params.get('key', 'color_count')
            bg = self.params.get('bg', 0)
            cols = [[inp[r][c] for r in range(h)] for c in range(w)]
            if key == 'color_count':
                cols.sort(key=lambda col: sum(1 for x in col if x != bg))
            elif key == 'color_count_desc':
                cols.sort(key=lambda col: sum(1 for x in col if x != bg), reverse=True)
            elif key == 'sum':
                cols.sort(key=sum)
            elif key == 'sum_desc':
                cols.sort(key=sum, reverse=True)
            result = [[cols[c][r] for c in range(w)] for r in range(h)]
            return result
        
        elif self.name == 'rotate_90':
            return rotate_90(inp)
        elif self.name == 'rotate_180':
            return rotate_180(inp)
        elif self.name == 'rotate_270':
            return rotate_270(inp)
        elif self.name == 'transpose':
            return transpose(inp)
        
        elif self.name == 'colormap':
            mapping = self.params['map']
            return [[mapping.get(c, c) for c in row] for row in inp]
        
        elif self.name == 'fill_enclosed_auto':
            bg = most_common_color(inp)
            # Find enclosed bg cells and fill with the most common non-bg neighbor color
            enclosed = set()
            reachable = set()
            queue = []
            for r in range(h):
                for c in range(w):
                    if (r == 0 or r == h-1 or c == 0 or c == w-1) and inp[r][c] == bg:
                        reachable.add((r, c))
                        queue.append((r, c))
            while queue:
                cr, cc = queue.pop(0)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = cr+dr, cc+dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and inp[nr][nc] == bg:
                        reachable.add((nr, nc))
                        queue.append((nr, nc))
            
            enclosed = {(r,c) for r in range(h) for c in range(w) if inp[r][c] == bg and (r,c) not in reachable}
            if not enclosed:
                return None
            
            # Group enclosed cells into connected components
            visited = set()
            result = [list(row) for row in inp]
            for er, ec in enclosed:
                if (er, ec) in visited:
                    continue
                # BFS to find component
                comp = set()
                q = [(er, ec)]
                while q:
                    cr, cc = q.pop(0)
                    if (cr, cc) in comp:
                        continue
                    comp.add((cr, cc))
                    visited.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if (nr, nc) in enclosed and (nr, nc) not in comp:
                            q.append((nr, nc))
                
                # Find surrounding non-bg colors
                neighbor_colors = Counter()
                for cr, cc in comp:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] != bg:
                            neighbor_colors[inp[nr][nc]] += 1
                
                # Fill with... we need to figure out fill color from training
                # For now, use a fixed fill approach
                fill_c = neighbor_colors.most_common(1)[0][0] + 1 if neighbor_colors else bg
                # This is too heuristic, skip
            return None
        
        elif self.name == 'gravity_down':
            color = self.params['color']
            bg = self.params['bg']
            result = [list(row) for row in inp]
            for c in range(w):
                # Collect non-bg cells of this color
                cells = [r for r in range(h) if inp[r][c] == color]
                if not cells:
                    continue
                # Clear old positions
                for r in cells:
                    result[r][c] = bg
                # Place at bottom
                for i, r in enumerate(range(h - 1, h - 1 - len(cells), -1)):
                    result[r][c] = color
            return result
        
        elif self.name == 'symmetry_fill':
            bg = self.params['bg']
            axis = self.params['axis']
            result = [list(row) for row in inp]
            for r in range(h):
                for c in range(w):
                    if result[r][c] == bg:
                        if axis == 'h' or axis == 'hv':
                            mc = w - 1 - c
                            if 0 <= mc < w and inp[r][mc] != bg:
                                result[r][c] = inp[r][mc]
                        if result[r][c] == bg and (axis == 'v' or axis == 'hv'):
                            mr = h - 1 - r
                            if 0 <= mr < h and inp[mr][c] != bg:
                                result[r][c] = inp[mr][c]
                        if result[r][c] == bg and axis == 'hv':
                            mr, mc = h - 1 - r, w - 1 - c
                            if 0 <= mr < h and 0 <= mc < w and inp[mr][mc] != bg:
                                result[r][c] = inp[mr][mc]
            return result
        
        elif self.name == 'crop_bbox':
            bg = self.params.get('bg', most_common_color(inp))
            r1, r2, c1, c2 = h, 0, w, 0
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != bg:
                        r1=min(r1,r); r2=max(r2,r); c1=min(c1,c); c2=max(c2,c)
            if r1 > r2:
                return None
            return [row[c1:c2+1] for row in inp[r1:r2+1]]
        
        elif self.name == 'subgrid_select':
            # Split grid by separator lines, select the subgrid matching criteria
            sep_c = self.params['sep_color']
            select = self.params['select']  # 'largest_nonbg', 'smallest_nonbg', 'most_colors', 'least_colors', 'unique', index
            subgrids = _split_by_separators(inp, sep_c)
            if not subgrids:
                return None
            bg = self.params.get('bg', 0)
            
            if select == 'largest_nonbg':
                subgrids.sort(key=lambda g: sum(1 for row in g for c in row if c != bg), reverse=True)
                return subgrids[0]
            elif select == 'smallest_nonbg':
                nonbg = [(sum(1 for row in g for c in row if c != bg), g) for g in subgrids]
                nonbg = [(n, g) for n, g in nonbg if n > 0]
                if nonbg:
                    nonbg.sort()
                    return nonbg[0][1]
            elif select == 'most_colors':
                subgrids.sort(key=lambda g: len(set(c for row in g for c in row) - {bg}), reverse=True)
                return subgrids[0]
            elif select == 'unique':
                # Find the subgrid that appears only once
                for i, g in enumerate(subgrids):
                    count = sum(1 for g2 in subgrids if grid_eq(g, g2))
                    if count == 1:
                        return g
            elif select == 'majority':
                # Find the most common subgrid
                from collections import Counter as Cnt
                for i, g in enumerate(subgrids):
                    count = sum(1 for g2 in subgrids if grid_eq(g, g2))
                    if count > len(subgrids) // 2:
                        return g
            elif isinstance(select, int) and 0 <= select < len(subgrids):
                return subgrids[select]
            return None
        
        elif self.name == 'subgrid_overlay':
            # Split by separators, overlay subgrids (OR/AND/XOR)
            sep_c = self.params['sep_color']
            mode = self.params.get('mode', 'or')  # or, and, xor
            subgrids = _split_by_separators(inp, sep_c)
            if len(subgrids) < 2:
                return None
            bg = self.params.get('bg', 0)
            sh, sw = grid_shape(subgrids[0])
            if not all(grid_shape(g) == (sh, sw) for g in subgrids):
                return None
            result = [[bg]*sw for _ in range(sh)]
            if mode == 'or':
                for g in subgrids:
                    for r in range(sh):
                        for c in range(sw):
                            if g[r][c] != bg:
                                result[r][c] = g[r][c]
            elif mode == 'and':
                for r in range(sh):
                    for c in range(sw):
                        if all(g[r][c] != bg for g in subgrids):
                            result[r][c] = subgrids[0][r][c]
            elif mode == 'xor':
                for r in range(sh):
                    for c in range(sw):
                        nonbg = [g[r][c] for g in subgrids if g[r][c] != bg]
                        if len(nonbg) == 1:
                            result[r][c] = nonbg[0]
            return result
        
        elif self.name == 'subgrid_diff':
            # Find diff between two subgrids
            sep_c = self.params['sep_color']
            subgrids = _split_by_separators(inp, sep_c)
            if len(subgrids) != 2:
                return None
            bg = self.params.get('bg', 0)
            g1, g2 = subgrids
            sh, sw = grid_shape(g1)
            if grid_shape(g2) != (sh, sw):
                return None
            result = [[bg]*sw for _ in range(sh)]
            for r in range(sh):
                for c in range(sw):
                    if g1[r][c] != g2[r][c]:
                        result[r][c] = g1[r][c] if g1[r][c] != bg else g2[r][c]
            return result
        
        elif self.name == 'remove_color':
            color = self.params['color']
            bg = self.params.get('bg', 0)
            return [[bg if c == color else c for c in row] for row in inp]
        
        elif self.name == 'extract_largest_region':
            bg = self.params.get('bg', 0)
            regs = flood_fill_regions(inp)
            regs = [r for r in regs if r['color'] != bg]
            if not regs:
                return None
            largest = max(regs, key=lambda r: r['size'])
            r1, c1, r2, c2 = largest['bbox']
            return [row[c1:c2+1] for row in inp[r1:r2+1]]
        
        elif self.name == 'downscale':
            fh, fw = self.params['fh'], self.params['fw']
            oh2, ow2 = h // fh, w // fw
            result = []
            for r in range(oh2):
                row = []
                for c in range(ow2):
                    block = [inp[r*fh+dr][c*fw+dc] for dr in range(fh) for dc in range(fw)]
                    row.append(max(set(block), key=block.count))
                result.append(row)
            return result
        
        elif self.name == 'upscale':
            fh, fw = self.params['fh'], self.params['fw']
            return [[inp[r//fh][c//fw] for c in range(w*fw)] for r in range(h*fh)]
        
        elif self.name == 'keep_one_color':
            color = self.params['color']
            bg = self.params.get('bg', 0)
            return [[c if c == color else bg for c in row] for row in inp]
        
        elif self.name == 'flip_h':
            return flip_h(inp)
        elif self.name == 'flip_v':
            return flip_v(inp)
        
        elif self.name == 'extract_smallest_region':
            bg = self.params.get('bg', most_common_color(inp))
            regs = flood_fill_regions(inp)
            regs = [r for r in regs if r['color'] != bg]
            if not regs: return None
            smallest = min(regs, key=lambda r: r['size'])
            r1, c1, r2, c2 = smallest['bbox']
            return [row[c1:c2+1] for row in inp[r1:r2+1]]
        
        elif self.name == 'stack_v':
            return inp + inp
        elif self.name == 'stack_v_flip':
            return inp + flip_v(inp)
        elif self.name == 'stack_h':
            return [r + r for r in inp]
        elif self.name == 'stack_h_flip':
            return [r + list(reversed(r)) for r in inp]
        
        elif self.name == 'remove_empty_rows':
            bg = self.params.get('bg', most_common_color(inp))
            result = [row for row in inp if any(c != bg for c in row)]
            return result if result else None
        elif self.name == 'remove_empty_cols':
            bg = self.params.get('bg', most_common_color(inp))
            keep = [c for c in range(w) if any(inp[r][c] != bg for r in range(h))]
            return [[inp[r][c] for c in keep] for r in range(h)] if keep else None
        elif self.name == 'remove_empty_both':
            bg = self.params.get('bg', most_common_color(inp))
            rows = [row for row in inp if any(c != bg for c in row)]
            if not rows: return None
            h2, w2 = len(rows), len(rows[0])
            keep = [c for c in range(w2) if any(rows[r][c] != bg for r in range(h2))]
            return [[rows[r][c] for c in keep] for r in range(h2)] if keep else None
        
        elif self.name == 'gravity_all':
            bg = self.params.get('bg', most_common_color(inp))
            result = [[bg]*w for _ in range(h)]
            for c in range(w):
                vals = [inp[r][c] for r in range(h) if inp[r][c] != bg]
                for i, v in enumerate(reversed(vals)):
                    result[h-1-i][c] = v
            return result
        
        elif self.name == 'gravity_up':
            bg = self.params.get('bg', most_common_color(inp))
            result = [[bg]*w for _ in range(h)]
            for c in range(w):
                vals = [inp[r][c] for r in range(h) if inp[r][c] != bg]
                for i, v in enumerate(vals):
                    result[i][c] = v
            return result
        
        elif self.name == 'gravity_left':
            bg = self.params.get('bg', most_common_color(inp))
            result = [[bg]*w for _ in range(h)]
            for r in range(h):
                vals = [inp[r][c] for c in range(w) if inp[r][c] != bg]
                for i, v in enumerate(vals):
                    result[r][i] = v
            return result
        
        elif self.name == 'gravity_right':
            bg = self.params.get('bg', most_common_color(inp))
            result = [[bg]*w for _ in range(h)]
            for r in range(h):
                vals = [inp[r][c] for c in range(w) if inp[r][c] != bg]
                for i, v in enumerate(reversed(vals)):
                    result[r][w-1-i] = v
            return result
        
        elif self.name == 'connect_h':
            bg = self.params.get('bg', most_common_color(inp))
            result = [list(row) for row in inp]
            for r in range(h):
                for color in set(inp[r]) - {bg}:
                    positions = [c for c in range(w) if inp[r][c] == color]
                    if len(positions) >= 2:
                        for c in range(min(positions), max(positions)+1):
                            if result[r][c] == bg:
                                result[r][c] = color
            return result
        
        elif self.name == 'connect_v':
            bg = self.params.get('bg', most_common_color(inp))
            result = [list(row) for row in inp]
            for c in range(w):
                col_colors = set(inp[r][c] for r in range(h)) - {bg}
                for color in col_colors:
                    positions = [r for r in range(h) if inp[r][c] == color]
                    if len(positions) >= 2:
                        for r in range(min(positions), max(positions)+1):
                            if result[r][c] == bg:
                                result[r][c] = color
            return result
        
        elif self.name == 'connect_hv':
            bg = self.params.get('bg', most_common_color(inp))
            # First connect_h, then connect_v
            mid = [list(row) for row in inp]
            for r in range(h):
                for color in set(inp[r]) - {bg}:
                    positions = [c for c in range(w) if inp[r][c] == color]
                    if len(positions) >= 2:
                        for c in range(min(positions), max(positions)+1):
                            if mid[r][c] == bg:
                                mid[r][c] = color
            result = [list(row) for row in mid]
            for c in range(w):
                col_colors = set(mid[r][c] for r in range(h)) - {bg}
                for color in col_colors:
                    positions = [r for r in range(h) if mid[r][c] == color]
                    if len(positions) >= 2:
                        for r in range(min(positions), max(positions)+1):
                            if result[r][c] == bg:
                                result[r][c] = color
            return result
        
        elif self.name == 'outline':
            bg = self.params.get('bg', most_common_color(inp))
            result = [[bg]*w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != bg:
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if nr<0 or nr>=h or nc<0 or nc>=w or inp[nr][nc] == bg:
                                result[r][c] = inp[r][c]
                                break
            return result
        
        return None


def _split_by_separators(g: Grid, sep_color: int) -> List[Grid]:
    """Split grid into subgrids using separator rows/columns"""
    h, w = grid_shape(g)
    
    sep_rows = [r for r in range(h) if all(g[r][c] == sep_color for c in range(w))]
    sep_cols = [c for c in range(w) if all(g[r][c] == sep_color for r in range(h))]
    
    # Build row and column boundaries
    row_bounds = [-1] + sep_rows + [h]
    col_bounds = [-1] + sep_cols + [w]
    
    subgrids = []
    for i in range(len(row_bounds) - 1):
        for j in range(len(col_bounds) - 1):
            r1 = row_bounds[i] + 1
            r2 = row_bounds[i + 1]
            c1 = col_bounds[j] + 1
            c2 = col_bounds[j + 1]
            if r1 < r2 and c1 < c2:
                sub = [g[r][c1:c2] for r in range(r1, r2)]
                subgrids.append(sub)
    
    return subgrids


def verify_whole_grid(prog: WholeGridProgram, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    for inp, expected in train_pairs:
        result = prog.apply(inp)
        if result is None or not grid_eq(result, expected):
            return False
    return True


def verify_program(prog: SynthesizedProgram, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    """
    CEGIS verification: does program produce correct output for ALL training pairs?
    """
    for inp, expected in train_pairs:
        result = prog.apply(inp)
        if result is None or not grid_eq(result, expected):
            return False
    return True


def _generate_whole_grid_candidates(train_pairs: List[Tuple[Grid, Grid]]) -> List[WholeGridProgram]:
    """Generate WholeGridProgram candidates"""
    candidates = []
    
    if not train_pairs:
        return candidates
    
    inp0, out0 = train_pairs[0]
    ih, iw = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    
    # Rotations (for square grids or transposed sizes)
    if ih == iw and oh == ow and ih == oh:
        candidates.append(WholeGridProgram('rotate_90'))
        candidates.append(WholeGridProgram('rotate_180'))
        candidates.append(WholeGridProgram('rotate_270'))
    
    # Transpose
    if (oh, ow) == (iw, ih):
        candidates.append(WholeGridProgram('transpose'))
    
    # Flips
    if (ih, iw) == (oh, ow):
        candidates.append(WholeGridProgram('flip_h'))
        candidates.append(WholeGridProgram('flip_v'))
    
    # Row/col sort
    bg = most_common_color(inp0)
    for key in ['color_count', 'color_count_desc', 'sum', 'sum_desc', 'first_nonbg']:
        candidates.append(WholeGridProgram('row_sort', {'key': key, 'bg': bg}))
    for key in ['color_count', 'color_count_desc', 'sum', 'sum_desc']:
        candidates.append(WholeGridProgram('col_sort', {'key': key, 'bg': bg}))
    
    # Fill enclosed (whole-grid version, works with variable sizes)
    candidates.append(WholeGridProgram('fill_enclosed_auto'))
    
    # Colormap from all pairs (only when same size)
    all_same = all(grid_shape(i) == grid_shape(o) for i, o in train_pairs)
    if all_same:
        cmap = {}
        consistent = True
        for inp_i, out_i in train_pairs:
            hi, wi = grid_shape(inp_i)
            for r in range(hi):
                for c in range(wi):
                    ic, oc = inp_i[r][c], out_i[r][c]
                    if ic in cmap:
                        if cmap[ic] != oc: consistent = False; break
                    else: cmap[ic] = oc
                if not consistent: break
            if not consistent: break
        if consistent and any(k!=v for k,v in cmap.items()):
            candidates.append(WholeGridProgram('colormap', {'map': cmap}))
    
    # Gravity down (per color)
    for color in range(10):
        candidates.append(WholeGridProgram('gravity_down', {'color': color, 'bg': bg}))
    
    # Symmetry fill (whole grid)
    for axis in ['h', 'v', 'hv']:
        candidates.append(WholeGridProgram('symmetry_fill', {'bg': bg, 'axis': axis}))
    
    # Crop to bounding box
    candidates.append(WholeGridProgram('crop_bbox', {'bg': bg}))
    for c in grid_colors(inp0) - {bg}:
        candidates.append(WholeGridProgram('crop_bbox', {'bg': c}))
    
    # Extract largest region
    candidates.append(WholeGridProgram('extract_largest_region', {'bg': bg}))
    
    # Downscale
    if oh < ih and ow < iw and ih > 0 and iw > 0 and oh > 0 and ow > 0:
        if ih % oh == 0 and iw % ow == 0:
            candidates.append(WholeGridProgram('downscale', {'fh': ih//oh, 'fw': iw//ow}))
    
    # Upscale 
    if oh > ih and ow > iw and ih > 0 and iw > 0:
        if oh % ih == 0 and ow % iw == 0:
            candidates.append(WholeGridProgram('upscale', {'fh': oh//ih, 'fw': ow//iw}))
    
    # Keep one color
    for c in grid_colors(inp0) - {bg}:
        candidates.append(WholeGridProgram('keep_one_color', {'color': c, 'bg': bg}))
    
    # Extract smallest region
    candidates.append(WholeGridProgram('extract_smallest_region', {'bg': bg}))
    
    # Stack operations
    if oh == 2 * ih and ow == iw:
        candidates.append(WholeGridProgram('stack_v'))
        candidates.append(WholeGridProgram('stack_v_flip'))
    if oh == ih and ow == 2 * iw:
        candidates.append(WholeGridProgram('stack_h'))
        candidates.append(WholeGridProgram('stack_h_flip'))
    
    # Remove empty rows/cols
    candidates.append(WholeGridProgram('remove_empty_rows', {'bg': bg}))
    candidates.append(WholeGridProgram('remove_empty_cols', {'bg': bg}))
    candidates.append(WholeGridProgram('remove_empty_both', {'bg': bg}))
    
    # Gravity (all directions)
    candidates.append(WholeGridProgram('gravity_all', {'bg': bg}))
    candidates.append(WholeGridProgram('gravity_up', {'bg': bg}))
    candidates.append(WholeGridProgram('gravity_left', {'bg': bg}))
    candidates.append(WholeGridProgram('gravity_right', {'bg': bg}))
    
    # Connect same-color cells
    candidates.append(WholeGridProgram('connect_h', {'bg': bg}))
    candidates.append(WholeGridProgram('connect_v', {'bg': bg}))
    candidates.append(WholeGridProgram('connect_hv', {'bg': bg}))
    
    # Outline
    candidates.append(WholeGridProgram('outline', {'bg': bg}))
    
    # Remove color
    for c in grid_colors(inp0) - {bg}:
        candidates.append(WholeGridProgram('remove_color', {'color': c, 'bg': bg}))
    
    # Subgrid operations (detect separators)
    for sep_c in grid_colors(inp0) - {bg}:
        subs = _split_by_separators(inp0, sep_c)
        if len(subs) >= 2:
            for select in ['largest_nonbg', 'smallest_nonbg', 'most_colors', 'unique', 'majority']:
                candidates.append(WholeGridProgram('subgrid_select', {
                    'sep_color': sep_c, 'select': select, 'bg': bg
                }))
            for idx in range(min(len(subs), 6)):
                candidates.append(WholeGridProgram('subgrid_select', {
                    'sep_color': sep_c, 'select': idx, 'bg': bg
                }))
            for mode in ['or', 'and', 'xor']:
                candidates.append(WholeGridProgram('subgrid_overlay', {
                    'sep_color': sep_c, 'mode': mode, 'bg': bg
                }))
            candidates.append(WholeGridProgram('subgrid_diff', {
                'sep_color': sep_c, 'bg': bg
            }))
    
    return candidates


def solve_cross(train_pairs: List[Tuple[Grid, Grid]], test_inputs: List[Grid]) -> List[List[Grid]]:
    """
    Solve ARC task using Cross-structure synthesis + CEGIS verification.
    
    1. Generate candidate programs (hypothesis space)
    2. Verify each against ALL training pairs (CEGIS)
    3. Apply verified programs to test inputs
    4. Return up to 2 attempts per test input
    """
    # Step 1: Generate cell-rule candidates
    candidates = generate_candidates(train_pairs)
    
    # Step 2: CEGIS verification (cell rules)
    verified = []
    for prog in candidates:
        if verify_program(prog, train_pairs):
            verified.append(('cell', prog))
            if len(verified) >= 2:
                break
    
    # Step 3: Whole-grid candidates
    if len(verified) < 2:
        wg_cands = _generate_whole_grid_candidates(train_pairs)
        for wg in wg_cands:
            if verify_whole_grid(wg, train_pairs):
                verified.append(('whole', wg))
                if len(verified) >= 2:
                    break
    
    # Step 4: 2-step composition (if nothing found yet)
    if len(verified) < 2:
        wg_all = _generate_whole_grid_candidates(train_pairs)
        # Fast pre-filter: only try compositions where step1 changes something
        # and step2 applied to step1's output gives expected
        for p1 in wg_all:
            if len(verified) >= 2:
                break
            # Quick test: does p1 produce valid intermediate grids?
            mid0 = p1.apply(train_pairs[0][0])
            if mid0 is None:
                continue
            for p2 in wg_all:
                if p1.name == p2.name and p1.params == p2.params:
                    continue  # skip identity composition
                comp = CompositeProgram(p1, p2)
                ok = True
                for inp, exp in train_pairs:
                    res = comp.apply(inp)
                    if res is None or not grid_eq(res, exp):
                        ok = False
                        break
                if ok:
                    verified.append(('composite', comp))
                    if len(verified) >= 2:
                        break
    
    # Step 5: Apply to test inputs
    predictions = []
    for test_inp in test_inputs:
        attempts = []
        for kind, prog in verified[:2]:
            result = prog.apply(test_inp)
            if result is not None:
                attempts.append(result)
        predictions.append(attempts)
    
    return predictions, verified


# ── Evaluation entry point ──

def solve_task_cross(task_path: str) -> dict:
    """Solve a single ARC task file using Cross solver"""
    with open(task_path) as f:
        data = json.load(f)
    
    train = [(ex['input'], ex['output']) for ex in data['train']]
    test_inputs = [ex['input'] for ex in data['test']]
    test_outputs = [ex.get('output') for ex in data['test']]
    
    predictions, verified = solve_cross(train, test_inputs)
    
    # Check correctness
    correct = True
    attempted = bool(any(p for p in predictions))
    
    for ti, (preds, expected) in enumerate(zip(predictions, test_outputs)):
        if expected is None or not preds:
            correct = False
            continue
        if not any(grid_eq(p, expected) for p in preds):
            correct = False
    
    if verified:
        kind, prog = verified[0]
        if kind == 'cell':
            method = prog.rule.name
            rule_str = str(prog.rule)
        elif kind == 'composite':
            method = f"composite({prog.step1.name}+{prog.step2.name})"
            rule_str = method
        else:
            method = prog.name
            rule_str = f"{prog.name}({prog.params})"
    else:
        method = 'none'
        rule_str = 'none'
    
    return {
        'correct': correct and attempted,
        'attempted': attempted,
        'method': method,
        'n_candidates': len(generate_candidates(train)),
        'n_verified': len(verified),
        'rule': rule_str,
    }
