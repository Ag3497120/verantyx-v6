"""
arc/pattern_atoms.py — Transform pattern detection as Atoms

ARC-AGI tasks encode grid transformations. We detect these as "TransformAtoms"
and synthesize rules by composing them.

TransformAtom categories:
  - geometric: rotate, flip, translate, scale, tile
  - color: recolor, fill, swap, gradient
  - structural: crop, extract, overlay, merge, split
  - logical: mask, conditional, count-based
  - topological: flood_fill, boundary, connectivity
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from arc.grid import (
    Grid, grid_shape, grid_eq, grid_colors,
    rotate_90, rotate_180, rotate_270,
    flip_h, flip_v, transpose,
    tile, tile_with_flip, tile_checkerboard,
    recolor, most_common_color,
    extract_subgrid, flood_fill_regions, analyze,
)


@dataclass
class TransformAtom:
    """An atomic transformation detected between input→output"""
    category: str       # geometric, color, structural, logical, topological
    operation: str       # specific operation name
    params: dict         # operation parameters
    confidence: float    # how well this explains the transform (0-1)
    explanation: str     # human-readable description


def detect_transforms(inp: Grid, out: Grid) -> List[TransformAtom]:
    """Detect all possible transform atoms between input and output grids."""
    atoms = []
    
    atoms.extend(_detect_geometric(inp, out))
    atoms.extend(_detect_tiling(inp, out))
    atoms.extend(_detect_color_mapping(inp, out))
    atoms.extend(_detect_scaling(inp, out))
    atoms.extend(_detect_crop_or_extract(inp, out))
    atoms.extend(_detect_overlay(inp, out))
    
    # Sort by confidence
    atoms.sort(key=lambda a: a.confidence, reverse=True)
    return atoms


# ── Geometric transforms ──

def _detect_geometric(inp: Grid, out: Grid) -> List[TransformAtom]:
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    # Identity
    if grid_eq(inp, out):
        atoms.append(TransformAtom('geometric', 'identity', {}, 1.0, 'Output is identical to input'))
    
    # Same size transforms
    if (ih, iw) == (oh, ow):
        if grid_eq(flip_h(inp), out):
            atoms.append(TransformAtom('geometric', 'flip_h', {}, 1.0, 'Horizontal flip'))
        if grid_eq(flip_v(inp), out):
            atoms.append(TransformAtom('geometric', 'flip_v', {}, 1.0, 'Vertical flip'))
        if grid_eq(rotate_180(inp), out):
            atoms.append(TransformAtom('geometric', 'rotate_180', {}, 1.0, '180° rotation'))
    
    # Rotated (dimensions swap)
    if (ih, iw) == (ow, oh):
        if grid_eq(rotate_90(inp), out):
            atoms.append(TransformAtom('geometric', 'rotate_90', {}, 1.0, '90° clockwise rotation'))
        if grid_eq(rotate_270(inp), out):
            atoms.append(TransformAtom('geometric', 'rotate_270', {}, 1.0, '270° clockwise rotation'))
        if grid_eq(transpose(inp), out):
            atoms.append(TransformAtom('geometric', 'transpose', {}, 1.0, 'Transpose'))
    
    return atoms


# ── Tiling ──

def _detect_tiling(inp: Grid, out: Grid) -> List[TransformAtom]:
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    if ih == 0 or iw == 0:
        return atoms
    
    # Check if output is exact tiling of input
    if oh % ih == 0 and ow % iw == 0:
        rh = oh // ih
        rw = ow // iw
        if rh >= 1 and rw >= 1 and (rh > 1 or rw > 1):
            # Simple tile
            if grid_eq(tile(inp, rh, rw), out):
                atoms.append(TransformAtom(
                    'geometric', 'tile', {'repeat_h': rh, 'repeat_w': rw},
                    1.0, f'Tile {rh}×{rw}'
                ))
            # Tile with alternating row flips
            if grid_eq(tile_with_flip(inp, rh, rw), out):
                atoms.append(TransformAtom(
                    'geometric', 'tile_flip', {'repeat_h': rh, 'repeat_w': rw},
                    1.0, f'Tile with row flip {rh}×{rw}'
                ))
            # Tile with checkerboard flips
            if grid_eq(tile_checkerboard(inp, rh, rw), out):
                atoms.append(TransformAtom(
                    'geometric', 'tile_checkerboard', {'repeat_h': rh, 'repeat_w': rw},
                    1.0, f'Tile checkerboard {rh}×{rw}'
                ))
    
    return atoms


# ── Color mapping ──

def _detect_color_mapping(inp: Grid, out: Grid) -> List[TransformAtom]:
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    if (ih, iw) != (oh, ow):
        return atoms
    
    # Try to find a consistent color mapping
    cmap = {}
    consistent = True
    for r in range(ih):
        for c in range(iw):
            ic = inp[r][c]
            oc = out[r][c]
            if ic in cmap:
                if cmap[ic] != oc:
                    consistent = False
                    break
            else:
                cmap[ic] = oc
        if not consistent:
            break
    
    if consistent and cmap:
        # Check it's not identity
        non_identity = any(k != v for k, v in cmap.items())
        if non_identity:
            atoms.append(TransformAtom(
                'color', 'recolor', {'map': cmap},
                1.0, f'Color mapping: {cmap}'
            ))
        
        # Check for color swap (A↔B)
        swaps = [(k, v) for k, v in cmap.items() if cmap.get(v) == k and k != v]
        if swaps:
            atoms.append(TransformAtom(
                'color', 'color_swap', {'swaps': swaps},
                0.95, f'Color swap: {swaps}'
            ))
    
    return atoms


# ── Scaling ──

def _detect_scaling(inp: Grid, out: Grid) -> List[TransformAtom]:
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    if ih == 0 or iw == 0:
        return atoms
    
    # Integer scaling
    if oh % ih == 0 and ow % iw == 0:
        sh = oh // ih
        sw = ow // iw
        if sh == sw and sh > 1:
            # Check pixel scaling
            scaled = True
            for r in range(ih):
                for c in range(iw):
                    for dr in range(sh):
                        for dc in range(sw):
                            if out[r * sh + dr][c * sw + dc] != inp[r][c]:
                                scaled = False
                                break
                        if not scaled:
                            break
                    if not scaled:
                        break
                if not scaled:
                    break
            
            if scaled:
                atoms.append(TransformAtom(
                    'geometric', 'scale', {'factor': sh},
                    1.0, f'Scale {sh}x'
                ))
    
    return atoms


# ── Crop / Extract ──

def _detect_crop_or_extract(inp: Grid, out: Grid) -> List[TransformAtom]:
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    # Output smaller → could be crop/extract
    if oh <= ih and ow <= iw and (oh < ih or ow < iw):
        # Try all positions
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                sub = extract_subgrid(inp, r, c, r + oh - 1, c + ow - 1)
                if grid_eq(sub, out):
                    atoms.append(TransformAtom(
                        'structural', 'extract', {'r': r, 'c': c, 'h': oh, 'w': ow},
                        0.9, f'Extract subgrid at ({r},{c}) size {oh}×{ow}'
                    ))
        
        # Extract by region (non-bg bounding box)
        regions = flood_fill_regions(inp)
        if regions:
            for reg in regions[:5]:
                r1, c1, r2, c2 = reg['bbox']
                sub = extract_subgrid(inp, r1, c1, r2, c2)
                if grid_eq(sub, out):
                    atoms.append(TransformAtom(
                        'structural', 'extract_region',
                        {'color': reg['color'], 'bbox': reg['bbox']},
                        0.95, f'Extract region color={reg["color"]} bbox={reg["bbox"]}'
                    ))
    
    return atoms


# ── Overlay / Merge ──

def _detect_overlay(inp: Grid, out: Grid) -> List[TransformAtom]:
    atoms = []
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    
    if (ih, iw) != (oh, ow):
        return atoms
    
    # Check which cells changed
    changed = []
    unchanged = []
    for r in range(ih):
        for c in range(iw):
            if inp[r][c] != out[r][c]:
                changed.append((r, c, inp[r][c], out[r][c]))
            else:
                unchanged.append((r, c))
    
    if not changed:
        return atoms
    
    # Check if all changes are to the same color (fill)
    new_colors = set(nc for _, _, _, nc in changed)
    if len(new_colors) == 1:
        fill_color = new_colors.pop()
        old_colors = set(oc for _, _, oc, _ in changed)
        atoms.append(TransformAtom(
            'color', 'fill_cells',
            {'count': len(changed), 'new_color': fill_color, 'old_colors': list(old_colors)},
            0.7, f'Fill {len(changed)} cells → color {fill_color}'
        ))
    
    # Check if changes only affect background cells
    bg = most_common_color(inp)
    if all(oc == bg for _, _, oc, _ in changed):
        atoms.append(TransformAtom(
            'structural', 'fill_background',
            {'bg_color': bg, 'new_colors': list(new_colors)},
            0.75, f'Fill background (color {bg}) with new patterns'
        ))
    
    return atoms


# ── Cross-matching transforms across train examples ──

def find_common_transforms(train_pairs: List[Tuple[Grid, Grid]]) -> List[TransformAtom]:
    """Find transforms that appear in ALL training pairs"""
    if not train_pairs:
        return []
    
    # Get transforms for each pair
    per_pair = [detect_transforms(inp, out) for inp, out in train_pairs]
    
    if not per_pair:
        return []
    
    # Find operations that appear in all pairs
    # Match by (category, operation)
    first_ops = {(a.category, a.operation) for a in per_pair[0]}
    
    common_ops = first_ops
    for pair_atoms in per_pair[1:]:
        pair_ops = {(a.category, a.operation) for a in pair_atoms}
        common_ops = common_ops & pair_ops
    
    # Collect common atoms with averaged confidence
    result = []
    for cat, op in common_ops:
        atoms_per_pair = []
        for pair_atoms in per_pair:
            matching = [a for a in pair_atoms if a.category == cat and a.operation == op]
            if matching:
                atoms_per_pair.append(matching[0])
        
        if len(atoms_per_pair) == len(train_pairs):
            # Check if params are consistent
            avg_conf = sum(a.confidence for a in atoms_per_pair) / len(atoms_per_pair)
            result.append(TransformAtom(
                cat, op,
                atoms_per_pair[0].params,  # use first pair's params
                avg_conf,
                f'{atoms_per_pair[0].explanation} (consistent across {len(train_pairs)} examples)'
            ))
    
    result.sort(key=lambda a: a.confidence, reverse=True)
    return result
