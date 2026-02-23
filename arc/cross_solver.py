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
from dataclasses import dataclass, field
from arc.grid import (
    Grid, grid_shape, grid_eq, grid_colors,
    most_common_color, flood_fill_regions,
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
            # Extract subgrid starting at (sr, sc)
            sr, sc = self.params['sr'], self.params['sc']
            ir, ic = sr + r, sc + c
            if 0 <= ir < ih and 0 <= ic < iw:
                return inp[ir][ic]
        
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
                # Scaling or tiling
                if oh_rel > 1 and ow_rel > 1:
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
    
    # Color mapping: infer from first pair
    cmap = {}
    consistent = True
    for r in range(h):
        for c in range(w):
            ic = inp0[r][c]
            oc = out0[r][c]
            if ic in cmap:
                if cmap[ic] != oc:
                    consistent = False
                    break
            else:
                cmap[ic] = oc
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


def verify_program(prog: SynthesizedProgram, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    """
    CEGIS verification: does program produce correct output for ALL training pairs?
    """
    for inp, expected in train_pairs:
        result = prog.apply(inp)
        if result is None or not grid_eq(result, expected):
            return False
    return True


def solve_cross(train_pairs: List[Tuple[Grid, Grid]], test_inputs: List[Grid]) -> List[List[Grid]]:
    """
    Solve ARC task using Cross-structure synthesis + CEGIS verification.
    
    1. Generate candidate programs (hypothesis space)
    2. Verify each against ALL training pairs (CEGIS)
    3. Apply verified programs to test inputs
    4. Return up to 2 attempts per test input
    """
    # Step 1: Generate candidates
    candidates = generate_candidates(train_pairs)
    
    # Step 2: CEGIS verification
    verified = []
    for prog in candidates:
        if verify_program(prog, train_pairs):
            verified.append(prog)
            if len(verified) >= 2:  # ARC allows pass@2
                break
    
    # Step 3: Apply to test inputs
    predictions = []
    for test_inp in test_inputs:
        attempts = []
        for prog in verified[:2]:
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
    
    method = verified[0].rule.name if verified else 'none'
    
    return {
        'correct': correct and attempted,
        'attempted': attempted,
        'method': method,
        'n_candidates': len(generate_candidates(train)),
        'n_verified': len(verified),
        'rule': str(verified[0].rule) if verified else 'none',
    }
