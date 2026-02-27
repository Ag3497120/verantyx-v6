"""
arc/symmetry_solver.py — Symmetry Detection & Repair Solver

Detects broken symmetry in input grids and repairs them.
Handles: horizontal mirror, vertical mirror, 90° rotation, 180° rotation,
diagonal mirror, and combinations thereof.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter

from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.cross_engine import CrossPiece


def _apply_symmetry(grid: np.ndarray, sym_type: str) -> np.ndarray:
    """Apply a symmetry transform to get the mirrored version."""
    if sym_type == 'h_mirror':
        return grid[:, ::-1]
    elif sym_type == 'v_mirror':
        return grid[::-1, :]
    elif sym_type == 'rot90':
        return np.rot90(grid, 1)
    elif sym_type == 'rot180':
        return np.rot90(grid, 2)
    elif sym_type == 'rot270':
        return np.rot90(grid, 3)
    elif sym_type == 'diag_main':  # transpose
        return grid.T
    elif sym_type == 'diag_anti':  # anti-diagonal transpose
        return np.rot90(grid, 1)[::-1, :]
    return grid


def _repair_by_symmetry(grid: np.ndarray, sym_type: str, bg: int) -> np.ndarray:
    """
    Repair a grid by enforcing symmetry.
    For each pair of symmetric positions:
    - If one is bg and the other isn't, copy the non-bg value
    - If both are non-bg but different, keep the original (don't change non-bg)
    """
    h, w = grid.shape
    result = grid.copy()
    mirrored = _apply_symmetry(grid, sym_type)
    
    if mirrored.shape != grid.shape:
        return result
    
    for r in range(h):
        for c in range(w):
            if result[r, c] == bg and mirrored[r, c] != bg:
                result[r, c] = mirrored[r, c]
    
    return result


def _repair_by_symmetry_overwrite(grid: np.ndarray, sym_type: str, bg: int) -> np.ndarray:
    """
    Repair by enforcing symmetry, overwriting conflicts.
    For conflicts: use the non-bg value, or the value from the "preferred" half.
    """
    h, w = grid.shape
    result = grid.copy()
    
    mirrored = _apply_symmetry(grid, sym_type)
    if mirrored.shape != grid.shape:
        return result
    
    for r in range(h):
        for c in range(w):
            if result[r, c] == bg and mirrored[r, c] != bg:
                result[r, c] = mirrored[r, c]
    
    return result


def _find_symmetry_axis(grid: np.ndarray, bg: int) -> List[str]:
    """Detect which symmetries the grid approximately satisfies."""
    h, w = grid.shape
    symmetries = []
    
    for sym_type in ['h_mirror', 'v_mirror', 'rot180', 'diag_main']:
        mirrored = _apply_symmetry(grid, sym_type)
        if mirrored.shape != grid.shape:
            continue
        
        # Count how many non-bg cells match
        non_bg = (grid != bg) | (mirrored != bg)
        if not non_bg.any():
            continue
        
        matching = (grid == mirrored) & non_bg
        match_ratio = matching.sum() / non_bg.sum() if non_bg.sum() > 0 else 0
        
        if match_ratio > 0.7:  # At least 70% of non-bg cells match
            symmetries.append((sym_type, match_ratio))
    
    # Sort by match ratio
    symmetries.sort(key=lambda x: -x[1])
    return [s[0] for s in symmetries]


def try_symmetry_repair(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Try: output is input with symmetry enforced/repaired.
    """
    pieces = []
    
    for sym_type in ['h_mirror', 'v_mirror', 'rot180', 'diag_main', 'diag_anti']:
        # Try simple repair (only fill bg cells)
        all_match = True
        for inp, out in train_pairs:
            inp_np = np.array(inp)
            out_np = np.array(out)
            if inp_np.shape != out_np.shape:
                all_match = False
                break
            repaired = _repair_by_symmetry(inp_np, sym_type, bg)
            if not np.array_equal(repaired, out_np):
                all_match = False
                break
        
        if all_match:
            def _apply(inp, _bg=bg, _st=sym_type):
                return _repair_by_symmetry(np.array(inp), _st, _bg).tolist()
            pieces.append(CrossPiece(f'symmetry:repair_{sym_type}', _apply))
        
        # Try overwrite repair
        all_match2 = True
        for inp, out in train_pairs:
            inp_np = np.array(inp)
            out_np = np.array(out)
            if inp_np.shape != out_np.shape:
                all_match2 = False
                break
            repaired = _repair_by_symmetry_overwrite(inp_np, sym_type, bg)
            if not np.array_equal(repaired, out_np):
                all_match2 = False
                break
        
        if all_match2 and not all_match:  # don't duplicate
            def _apply2(inp, _bg=bg, _st=sym_type):
                return _repair_by_symmetry_overwrite(np.array(inp), _st, _bg).tolist()
            pieces.append(CrossPiece(f'symmetry:repair_overwrite_{sym_type}', _apply2))
    
    return pieces


def try_make_symmetric(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Try: output is a fully symmetric version derived from input.
    Detect which half is the "source" and mirror it.
    """
    pieces = []
    
    for sym_type in ['h_mirror', 'v_mirror', 'rot180']:
        # Try using left/top half as source
        for use_first_half in [True, False]:
            all_match = True
            for inp, out in train_pairs:
                inp_np = np.array(inp)
                out_np = np.array(out)
                h, w = inp_np.shape
                
                if inp_np.shape != out_np.shape:
                    all_match = False
                    break
                
                result = inp_np.copy()
                
                if sym_type == 'h_mirror':
                    mid = w // 2
                    if use_first_half:
                        # Copy left half to right
                        result[:, mid + (w % 2):] = result[:, :mid][:, ::-1]
                    else:
                        # Copy right half to left
                        result[:, :mid] = result[:, mid + (w % 2):][:, ::-1]
                elif sym_type == 'v_mirror':
                    mid = h // 2
                    if use_first_half:
                        result[mid + (h % 2):, :] = result[:mid, :][::-1, :]
                    else:
                        result[:mid, :] = result[mid + (h % 2):, :][::-1, :]
                elif sym_type == 'rot180':
                    if use_first_half:
                        # Use top-left quadrant
                        result = inp_np.copy()
                        rotated = np.rot90(result, 2)
                        # Overwrite second half with rotated first half
                        mid_r = h // 2
                        result[mid_r:, :] = rotated[mid_r:, :]
                    else:
                        result = inp_np.copy()
                        rotated = np.rot90(result, 2)
                        mid_r = h // 2
                        result[:mid_r, :] = rotated[:mid_r, :]
                
                if not np.array_equal(result, out_np):
                    all_match = False
                    break
            
            if all_match:
                def _apply(inp, _bg=bg, _st=sym_type, _fh=use_first_half):
                    inp_np = np.array(inp)
                    h, w = inp_np.shape
                    result = inp_np.copy()
                    if _st == 'h_mirror':
                        mid = w // 2
                        if _fh:
                            result[:, mid + (w % 2):] = result[:, :mid][:, ::-1]
                        else:
                            result[:, :mid] = result[:, mid + (w % 2):][:, ::-1]
                    elif _st == 'v_mirror':
                        mid = h // 2
                        if _fh:
                            result[mid + (h % 2):, :] = result[:mid, :][::-1, :]
                        else:
                            result[:mid, :] = result[mid + (h % 2):, :][::-1, :]
                    return result.tolist()
                
                half = 'first' if use_first_half else 'second'
                pieces.append(CrossPiece(f'symmetry:make_{sym_type}_{half}', _apply))
    
    return pieces


def try_complete_pattern(train_pairs: List[Tuple[Grid, Grid]], bg: int) -> List[CrossPiece]:
    """
    Try: input has a partial pattern, output completes it.
    Detect the pattern from non-bg cells and fill in the gaps.
    
    Strategy: find the smallest rectangular tile that explains all non-bg cells,
    then tile it to fill the grid.
    """
    pieces = []
    
    inp0_np = np.array(train_pairs[0][0])
    out0_np = np.array(train_pairs[0][1])
    
    if inp0_np.shape != out0_np.shape:
        return pieces
    
    h, w = inp0_np.shape
    
    # Check if output has translational symmetry
    for ph in range(1, min(h // 2 + 1, 16)):
        for pw in range(1, min(w // 2 + 1, 16)):
            # Extract tile from output
            tile = out0_np[:ph, :pw]
            
            # Check if output is perfectly tiled
            is_tiled = True
            for r in range(h):
                for c in range(w):
                    if out0_np[r, c] != tile[r % ph, c % pw]:
                        is_tiled = False
                        break
                if not is_tiled:
                    break
            
            if not is_tiled:
                continue
            
            # Check if input's non-bg cells are consistent with this tile
            for r in range(h):
                for c in range(w):
                    if inp0_np[r, c] != bg and inp0_np[r, c] != tile[r % ph, c % pw]:
                        is_tiled = False
                        break
                if not is_tiled:
                    break
            
            if not is_tiled:
                continue
            
            # Verify on all training pairs
            all_ok = True
            for inp, out in train_pairs:
                inp_np = np.array(inp)
                out_np = np.array(out)
                h2, w2 = inp_np.shape
                
                if inp_np.shape != out_np.shape:
                    all_ok = False
                    break
                
                # Reconstruct tile from input's non-bg cells
                tile2 = np.full((ph, pw), bg, dtype=inp_np.dtype)
                tile_set = np.zeros((ph, pw), dtype=bool)
                conflict = False
                
                for r in range(h2):
                    for c in range(w2):
                        if inp_np[r, c] != bg:
                            tr, tc = r % ph, c % pw
                            if tile_set[tr, tc] and tile2[tr, tc] != inp_np[r, c]:
                                conflict = True
                                break
                            tile2[tr, tc] = inp_np[r, c]
                            tile_set[tr, tc] = True
                    if conflict:
                        break
                
                if conflict or not tile_set.all():
                    all_ok = False
                    break
                
                # Apply tile
                result = np.zeros_like(inp_np)
                for r in range(h2):
                    for c in range(w2):
                        result[r, c] = tile2[r % ph, c % pw]
                
                if not np.array_equal(result, out_np):
                    all_ok = False
                    break
            
            if all_ok:
                def _apply(inp, _bg=bg, _ph=ph, _pw=pw):
                    inp_np = np.array(inp)
                    h, w = inp_np.shape
                    tile = np.full((_ph, _pw), _bg, dtype=inp_np.dtype)
                    for r in range(h):
                        for c in range(w):
                            if inp_np[r, c] != _bg:
                                tile[r % _ph, c % _pw] = inp_np[r, c]
                    result = np.zeros_like(inp_np)
                    for r in range(h):
                        for c in range(w):
                            result[r, c] = tile[r % _ph, c % _pw]
                    return result.tolist()
                
                pieces.append(CrossPiece(f'symmetry:tile_complete_{ph}x{pw}', _apply))
                return pieces  # Found one
    
    return pieces


def generate_symmetry_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate symmetry-based transform pieces."""
    pieces = []
    
    inp0, out0 = train_pairs[0]
    bg = most_common_color(inp0)
    
    pieces.extend(try_symmetry_repair(train_pairs, bg))
    pieces.extend(try_make_symmetric(train_pairs, bg))
    pieces.extend(try_complete_pattern(train_pairs, bg))
    
    return pieces
