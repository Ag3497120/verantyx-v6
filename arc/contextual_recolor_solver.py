"""
arc/contextual_recolor_solver.py — Contextual Recoloring Solver

Finds rules like:
- "Color A next to color B → change A to C"
- "Color A with N neighbors of color B → change to C"
- "Color A in a row/column with color B → change to C"

Works for ver=0 same-size tasks where output differs from input only in color changes.
"""

import numpy as np
from typing import List, Tuple
from collections import Counter
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _bg(g):
    return int(Counter(g.flatten()).most_common(1)[0][0])

def _neighbor_colors(g, r, c, radius=1):
    """Get set of colors in the neighborhood"""
    H, W = g.shape
    colors = set()
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            if dr == 0 and dc == 0: continue
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                colors.add(int(g[nr, nc]))
    return colors

def _4nb_colors(g, r, c):
    """4-connected neighbor colors"""
    H, W = g.shape
    colors = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < H and 0 <= nc < W:
            colors.append(int(g[nr, nc]))
    return colors

def generate_contextual_recolor_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate pieces that recolor based on local context"""
    pieces = []
    
    if not train_pairs:
        return pieces
    
    inp0 = np.array(train_pairs[0][0])
    out0 = np.array(train_pairs[0][1])
    
    # Only for same-size tasks
    if inp0.shape != out0.shape:
        return pieces
    
    bg = _bg(inp0)
    
    # Collect all (old_color, new_color, neighbor_has_color) triples
    # Strategy 1: "If cell is color A and has 4-neighbor of color B, change to C"
    rules_4nb = {}  # (old_color, nb_color) -> new_color
    rules_4nb_consistent = True
    
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return pieces
        
        H, W = inp.shape
        for r in range(H):
            for c in range(W):
                old_c = int(inp[r, c])
                new_c = int(out[r, c])
                if old_c == new_c:
                    continue
                
                nbs = _4nb_colors(inp, r, c)
                nb_set = set(nbs)
                
                for nb_c in nb_set:
                    key = (old_c, nb_c)
                    if key in rules_4nb:
                        if rules_4nb[key] != new_c:
                            rules_4nb_consistent = False
                    else:
                        rules_4nb[key] = new_c
    
    if rules_4nb_consistent and rules_4nb:
        # Verify: also check that unchanged cells DON'T match the rule
        verify_ok = True
        for inp_raw, out_raw in train_pairs:
            inp = np.array(inp_raw)
            out = np.array(out_raw)
            H, W = inp.shape
            pred = inp.copy()
            for r in range(H):
                for c in range(W):
                    old_c = int(inp[r, c])
                    nbs = set(_4nb_colors(inp, r, c))
                    for nb_c in nbs:
                        if (old_c, nb_c) in rules_4nb:
                            pred[r, c] = rules_4nb[(old_c, nb_c)]
                            break
            if not np.array_equal(pred, out):
                verify_ok = False
                break
        
        if verify_ok:
            _rules = dict(rules_4nb)
            def apply_4nb_recolor(inp_raw, rules=_rules):
                inp = np.array(inp_raw)
                H, W = inp.shape
                out = inp.copy()
                for r in range(H):
                    for c in range(W):
                        old_c = int(inp[r, c])
                        nbs = set(_4nb_colors(inp, r, c))
                        for nb_c in nbs:
                            if (old_c, nb_c) in rules:
                                out[r, c] = rules[(old_c, nb_c)]
                                break
                return out.tolist()
            pieces.append(CrossPiece('ctx_recolor_4nb', apply_4nb_recolor))
    
    # Strategy 2: "If cell is color A and has 8-neighbor of color B, change to C"
    rules_8nb = {}
    rules_8nb_consistent = True
    
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return pieces
        
        H, W = inp.shape
        for r in range(H):
            for c in range(W):
                old_c = int(inp[r, c])
                new_c = int(out[r, c])
                if old_c == new_c:
                    continue
                
                nb_set = _neighbor_colors(inp, r, c, radius=1)
                
                for nb_c in nb_set:
                    key = (old_c, nb_c)
                    if key in rules_8nb:
                        if rules_8nb[key] != new_c:
                            rules_8nb_consistent = False
                    else:
                        rules_8nb[key] = new_c
    
    if rules_8nb_consistent and rules_8nb and not (rules_4nb_consistent and rules_4nb):
        verify_ok = True
        for inp_raw, out_raw in train_pairs:
            inp = np.array(inp_raw)
            out = np.array(out_raw)
            H, W = inp.shape
            pred = inp.copy()
            for r in range(H):
                for c in range(W):
                    old_c = int(inp[r, c])
                    nb_set = _neighbor_colors(inp, r, c, radius=1)
                    for nb_c in nb_set:
                        if (old_c, nb_c) in rules_8nb:
                            pred[r, c] = rules_8nb[(old_c, nb_c)]
                            break
            if not np.array_equal(pred, out):
                verify_ok = False
                break
        
        if verify_ok:
            _rules8 = dict(rules_8nb)
            def apply_8nb_recolor(inp_raw, rules=_rules8):
                inp = np.array(inp_raw)
                H, W = inp.shape
                out = inp.copy()
                for r in range(H):
                    for c in range(W):
                        old_c = int(inp[r, c])
                        nb_set = _neighbor_colors(inp, r, c, radius=1)
                        for nb_c in nb_set:
                            if (old_c, nb_c) in rules:
                                out[r, c] = rules[(old_c, nb_c)]
                                break
                return out.tolist()
            pieces.append(CrossPiece('ctx_recolor_8nb', apply_8nb_recolor))
    
    # Strategy 3: Row-based context — "Color A in same row as color B → C"
    rules_row = {}
    rules_row_consistent = True
    
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        H, W = inp.shape
        for r in range(H):
            row_colors = set(int(inp[r, c]) for c in range(W))
            for c in range(W):
                old_c = int(inp[r, c])
                new_c = int(out[r, c])
                if old_c == new_c:
                    continue
                for rc in row_colors:
                    if rc == old_c: continue
                    key = (old_c, rc)
                    if key in rules_row:
                        if rules_row[key] != new_c:
                            rules_row_consistent = False
                    else:
                        rules_row[key] = new_c
    
    if rules_row_consistent and rules_row:
        verify_ok = True
        for inp_raw, out_raw in train_pairs:
            inp = np.array(inp_raw)
            out = np.array(out_raw)
            H, W = inp.shape
            pred = inp.copy()
            for r in range(H):
                row_colors = set(int(inp[r, c]) for c in range(W))
                for c in range(W):
                    old_c = int(inp[r, c])
                    for rc in row_colors:
                        if rc == old_c: continue
                        if (old_c, rc) in rules_row:
                            pred[r, c] = rules_row[(old_c, rc)]
                            break
            if not np.array_equal(pred, out):
                verify_ok = False
                break
        
        if verify_ok:
            _rules_r = dict(rules_row)
            def apply_row_recolor(inp_raw, rules=_rules_r):
                inp = np.array(inp_raw)
                H, W = inp.shape
                out = inp.copy()
                for r in range(H):
                    row_colors = set(int(inp[r, c]) for c in range(W))
                    for c in range(W):
                        old_c = int(inp[r, c])
                        for rc in row_colors:
                            if rc == old_c: continue
                            if (old_c, rc) in rules:
                                out[r, c] = rules[(old_c, rc)]
                                break
                return out.tolist()
            pieces.append(CrossPiece('ctx_recolor_row', apply_row_recolor))
    
    # Strategy 4: Column-based context
    rules_col = {}
    rules_col_consistent = True
    
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        H, W = inp.shape
        for c in range(W):
            col_colors = set(int(inp[r, c]) for r in range(H))
            for r in range(H):
                old_c = int(inp[r, c])
                new_c = int(out[r, c])
                if old_c == new_c:
                    continue
                for cc in col_colors:
                    if cc == old_c: continue
                    key = (old_c, cc)
                    if key in rules_col:
                        if rules_col[key] != new_c:
                            rules_col_consistent = False
                    else:
                        rules_col[key] = new_c
    
    if rules_col_consistent and rules_col:
        verify_ok = True
        for inp_raw, out_raw in train_pairs:
            inp = np.array(inp_raw)
            out = np.array(out_raw)
            H, W = inp.shape
            pred = inp.copy()
            for c in range(W):
                col_colors = set(int(inp[r, c]) for r in range(H))
                for r in range(H):
                    old_c = int(inp[r, c])
                    for cc in col_colors:
                        if cc == old_c: continue
                        if (old_c, cc) in rules_col:
                            pred[r, c] = rules_col[(old_c, cc)]
                            break
            if not np.array_equal(pred, out):
                verify_ok = False
                break
        
        if verify_ok:
            _rules_c = dict(rules_col)
            def apply_col_recolor(inp_raw, rules=_rules_c):
                inp = np.array(inp_raw)
                H, W = inp.shape
                out = inp.copy()
                for c in range(W):
                    col_colors = set(int(inp[r, c]) for r in range(H))
                    for r in range(H):
                        old_c = int(inp[r, c])
                        for cc in col_colors:
                            if cc == old_c: continue
                            if (old_c, cc) in rules:
                                out[r, c] = rules[(old_c, cc)]
                                break
                return out.tolist()
            pieces.append(CrossPiece('ctx_recolor_col', apply_col_recolor))
    
    return pieces
