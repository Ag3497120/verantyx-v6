"""
arc/cross_engine.py — Cross-Structure Engine for ARC-AGI-2

Full Verantyx Cross architecture:
1. Decompose task into structural pieces
2. Cross-Simulator: verify piece combinations against constraints
3. Puzzle reasoning: CEGIS with backtracking

Integrates: cross_solver (DSL), objects, nb_abstract, conditional
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.cross_solver import (
    solve_cross, WholeGridProgram, 
    _generate_whole_grid_candidates, verify_whole_grid,
    generate_candidates, verify_program, CompositeProgram,
)
from arc.nb_abstract import (
    learn_abstract_nb_rule, apply_abstract_nb_rule,
    learn_count_based_rule, apply_count_based_rule,
    learn_structural_nb_rule, apply_structural_nb_rule,
    learn_cross_nb_rule, apply_cross_nb_rule,
)
from arc.conditional import (
    learn_conditional_color_rule, apply_conditional_color_rule,
    learn_region_property_rule, apply_region_property_rule,
)
from arc.objects import detect_objects, find_matching_objects, object_transform_type


class CrossPiece:
    """A piece in the Cross structure — an atomic transformation"""
    def __init__(self, name: str, apply_fn, params: dict = None):
        self.name = name
        self.apply_fn = apply_fn
        self.params = params or {}
    
    def apply(self, inp: Grid) -> Optional[Grid]:
        try:
            return self.apply_fn(inp, **self.params)
        except Exception:
            return None
    
    def __repr__(self):
        return f"CrossPiece({self.name})"


class CrossSimulator:
    """Verify pieces against training constraints (small-world verification)"""
    
    @staticmethod
    def verify(piece: CrossPiece, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
        """CEGIS: verify piece against ALL training pairs"""
        for inp, expected in train_pairs:
            result = piece.apply(inp)
            if result is None or not grid_eq(result, expected):
                return False
        return True
    
    @staticmethod
    def partial_verify(piece: CrossPiece, train_pairs: List[Tuple[Grid, Grid]]) -> float:
        """Partial match score (0.0 to 1.0) — for puzzle reasoning"""
        if not train_pairs:
            return 0.0
        
        total_cells = 0
        matching_cells = 0
        
        for inp, expected in train_pairs:
            result = piece.apply(inp)
            if result is None:
                return 0.0
            
            h, w = grid_shape(expected)
            rh, rw = grid_shape(result)
            if (h, w) != (rh, rw):
                return 0.0
            
            for r in range(h):
                for c in range(w):
                    total_cells += 1
                    if result[r][c] == expected[r][c]:
                        matching_cells += 1
        
        return matching_cells / total_cells if total_cells > 0 else 0.0


def _generate_cross_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate all candidate pieces from all modules"""
    pieces = []
    
    # === Module 1: Neighborhood Rules (Wall 2) ===
    # Priority 1: Cross NB (structural pattern + const colors + role-based output)
    for radius in [1, 2]:
        rule = learn_cross_nb_rule(train_pairs, radius)
        if rule is not None:
            r = rule
            pieces.insert(0, CrossPiece(
                f'cross_nb_r{radius}',
                lambda inp, _r=r: apply_cross_nb_rule(inp, _r)
            ))
            break
    
    # Priority 2: Structural NB (exact + structural fallback)
    for radius in [1, 2]:
        rule = learn_structural_nb_rule(train_pairs, radius)
        if rule is not None:
            r = rule
            pieces.append(CrossPiece(
                f'structural_nb_r{radius}',
                lambda inp, _r=r: apply_structural_nb_rule(inp, _r)
            ))
            break
    
    # Priority 3: Abstract NB
    for radius in [1, 2]:
        rule = learn_abstract_nb_rule(train_pairs, radius)
        if rule is not None:
            r = rule
            pieces.append(CrossPiece(
                f'abstract_nb_r{radius}',
                lambda inp, _r=r: apply_abstract_nb_rule(inp, _r)
            ))
            break
    
    # Count-based rule
    rule = learn_count_based_rule(train_pairs)
    if rule is not None:
        r = rule
        pieces.append(CrossPiece(
            'count_based_nb',
            lambda inp, _r=r: apply_count_based_rule(inp, _r)
        ))
    
    # === Module 2: Conditional Rules (Wall 3) ===
    rule = learn_conditional_color_rule(train_pairs)
    if rule is not None:
        r = rule
        pieces.append(CrossPiece(
            'conditional_color',
            lambda inp, _r=r: apply_conditional_color_rule(inp, _r)
        ))
    
    rule = learn_region_property_rule(train_pairs)
    if rule is not None:
        r = rule
        if 'shape_map' in r:
            pieces.append(CrossPiece(
                'region_shape_recolor',
                lambda inp, _r=r: apply_region_property_rule(inp, _r)
            ))
        if 'size_map' in r:
            pieces.append(CrossPiece(
                'region_size_recolor',
                lambda inp, _r=r: apply_region_property_rule(inp, _r)
            ))
    
    # === Module 3: Object-level operations (Wall 1) ===
    bg = most_common_color(train_pairs[0][0])
    all_same = all(grid_shape(i) == grid_shape(o) for i, o in train_pairs)
    
    if all_same:
        # Try: move all objects by learned offset
        _add_object_move_pieces(pieces, train_pairs, bg)
        
        # Try: recolor objects by some property
        _add_object_recolor_pieces(pieces, train_pairs, bg)
        
        # Try: stamp pattern at marker positions
        _add_stamp_pieces(pieces, train_pairs, bg)
        
        # Try: object-level transforms (Selector × Transformer × Composer)
        _add_object_transform_pieces(pieces, train_pairs, bg)
    
    # Try: extract specific object
    _add_extract_pieces(pieces, train_pairs, bg)
    
    return pieces


def _add_object_move_pieces(pieces: List[CrossPiece], 
                            train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Try to learn object movement patterns"""
    # Check if all objects move by same delta
    for color in range(10):
        if color == bg:
            continue
        
        deltas = []
        consistent = True
        
        for inp, out in train_pairs:
            objs_in = [o for o in detect_objects(inp, bg) if o.color == color]
            objs_out = [o for o in detect_objects(out, bg) if o.color == color]
            
            if len(objs_in) != len(objs_out):
                consistent = False
                break
            
            for oi, oo in zip(
                sorted(objs_in, key=lambda o: o.bbox),
                sorted(objs_out, key=lambda o: o.bbox)
            ):
                if oi.shape != oo.shape:
                    consistent = False
                    break
                dr = oo.bbox[0] - oi.bbox[0]
                dc = oo.bbox[1] - oi.bbox[1]
                deltas.append((dr, dc))
            if not consistent:
                break
        
        if consistent and deltas and len(set(deltas)) == 1:
            dr, dc = deltas[0]
            if dr != 0 or dc != 0:
                _color = color
                _dr, _dc = dr, dc
                _bg = bg
                pieces.append(CrossPiece(
                    f'move_color_{color}_by_{dr}_{dc}',
                    lambda inp, c=_color, ddr=_dr, ddc=_dc, b=_bg: _apply_move_color(inp, c, ddr, ddc, b)
                ))


def _apply_move_color(inp: Grid, color: int, dr: int, dc: int, bg: int) -> Grid:
    """Move all objects of a specific color by (dr, dc)"""
    from arc.objects import detect_objects, move_object
    result = [row[:] for row in inp]
    objs = [o for o in detect_objects(inp, bg) if o.color == color]
    # Clear old positions
    for obj in objs:
        for r, c in obj.cells:
            result[r][c] = bg
    # Place at new positions
    h, w = grid_shape(inp)
    for obj in objs:
        for r, c in obj.cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = color
    return result


def _add_object_recolor_pieces(pieces: List[CrossPiece],
                               train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Try to learn object recoloring patterns"""
    from arc.objects import detect_objects
    
    # Learn: smallest object gets color X, largest gets color Y
    for sort_key in ['size', 'height', 'width']:
        rank_colors = {}
        consistent = True
        
        for inp, out in train_pairs:
            objs = detect_objects(inp, bg)
            if not objs:
                consistent = False
                break
            
            if sort_key == 'size':
                sorted_objs = sorted(objs, key=lambda o: o.size)
            elif sort_key == 'height':
                sorted_objs = sorted(objs, key=lambda o: o.height)
            else:
                sorted_objs = sorted(objs, key=lambda o: o.width)
            
            for rank, obj in enumerate(sorted_objs):
                out_cs = set(out[r][c] for r, c in obj.cells if out[r][c] != bg)
                if len(out_cs) == 1:
                    oc = out_cs.pop()
                    if rank in rank_colors:
                        if rank_colors[rank] != oc:
                            consistent = False
                            break
                    else:
                        rank_colors[rank] = oc
            if not consistent:
                break
        
        if consistent and rank_colors:
            _rc = rank_colors
            _sk = sort_key
            _bg = bg
            pieces.append(CrossPiece(
                f'recolor_by_{sort_key}_rank',
                lambda inp, rc=_rc, sk=_sk, b=_bg: _apply_rank_recolor(inp, rc, sk, b)
            ))


def _apply_rank_recolor(inp: Grid, rank_colors: Dict, sort_key: str, bg: int) -> Grid:
    from arc.objects import detect_objects
    objs = detect_objects(inp, bg)
    if sort_key == 'size':
        sorted_objs = sorted(objs, key=lambda o: o.size)
    elif sort_key == 'height':
        sorted_objs = sorted(objs, key=lambda o: o.height)
    else:
        sorted_objs = sorted(objs, key=lambda o: o.width)
    
    result = [row[:] for row in inp]
    for rank, obj in enumerate(sorted_objs):
        if rank in rank_colors:
            for r, c in obj.cells:
                result[r][c] = rank_colors[rank]
    return result


def _add_stamp_pieces(pieces: List[CrossPiece],
                      train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Try: find marker dots in input, stamp pattern at each"""
    from arc.objects import detect_objects
    
    inp0, out0 = train_pairs[0]
    h, w = grid_shape(inp0)
    
    # Find single-cell objects (dots/markers)
    objs = detect_objects(inp0, bg)
    dot_objs = [o for o in objs if o.size == 1]
    
    if not dot_objs or len(dot_objs) < 2:
        return
    
    # Check if all dots are same color
    dot_colors = set(o.color for o in dot_objs)
    if len(dot_colors) != 1:
        return
    
    dot_color = dot_colors.pop()
    
    # Find what pattern appears around dots in output
    # Use first dot to extract pattern
    dr0, dc0 = dot_objs[0].cells[0]
    
    # Try different pattern sizes
    for ps in [3, 5, 7]:
        pr = ps // 2
        r1, c1 = dr0 - pr, dc0 - pr
        if r1 < 0 or c1 < 0 or r1 + ps > h or c1 + ps > w:
            continue
        
        pattern = [out0[r][c1:c1+ps] for r in range(r1, r1+ps)]
        
        # Verify all dots have same pattern in output
        ok = True
        for dot in dot_objs[1:]:
            dr, dc = dot.cells[0]
            rr, cc = dr - pr, dc - pr
            if rr < 0 or cc < 0 or rr + ps > h or cc + ps > w:
                ok = False
                break
            actual = [out0[r][cc:cc+ps] for r in range(rr, rr+ps)]
            if actual != pattern:
                ok = False
                break
        
        if ok:
            _pattern = pattern
            _dot_color = dot_color
            _bg = bg
            _ps = ps
            pieces.append(CrossPiece(
                f'stamp_at_dots_{dot_color}_size{ps}',
                lambda inp, p=_pattern, dc=_dot_color, b=_bg, s=_ps: _apply_stamp(inp, p, dc, b, s)
            ))
            break


def _apply_stamp(inp: Grid, pattern: Grid, dot_color: int, bg: int, ps: int) -> Grid:
    h, w = grid_shape(inp)
    pr = ps // 2
    result = [row[:] for row in inp]
    for r in range(h):
        for c in range(w):
            if inp[r][c] == dot_color:
                for dr in range(ps):
                    for dc in range(ps):
                        nr, nc = r - pr + dr, c - pr + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if pattern[dr][dc] != bg:
                                result[nr][nc] = pattern[dr][dc]
    return result


def _add_object_transform_pieces(pieces: List[CrossPiece],
                                 train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Object-level DSL: Selector × Transformer × Composer
    
    Learn per-object transforms from training pairs using correspondence.
    """
    from arc.objects import detect_objects, find_matching_objects
    
    # Only for same-size grids
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return
    
    inp0, out0 = train_pairs[0]
    ih, iw = grid_shape(inp0)
    
    in_objs0 = detect_objects(inp0, bg)
    out_objs0 = detect_objects(out0, bg)
    
    if not in_objs0 or not out_objs0:
        return
    
    # === Strategy 1: Object-color mapping (most common pattern) ===
    # Each object keeps its shape/position but changes color based on a property
    
    # Build correspondence via position
    matches0 = find_matching_objects(in_objs0, out_objs0)
    if len(matches0) < 2:
        return
    
    # Check: recolor by neighbor count
    _try_recolor_by_neighbor_count(pieces, train_pairs, bg)
    
    # Check: recolor by enclosed/border property  
    _try_recolor_by_enclosure(pieces, train_pairs, bg)
    
    # Check: selective deletion (remove objects matching a condition)
    _try_selective_deletion(pieces, train_pairs, bg)
    
    # Check: fill object interiors
    _try_fill_object_interior(pieces, train_pairs, bg)


def _try_recolor_by_neighbor_count(pieces: List[CrossPiece],
                                    train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Recolor objects based on how many other objects they touch"""
    from arc.objects import detect_objects, find_matching_objects
    
    for pair_idx, (inp, out) in enumerate(train_pairs):
        h, w = grid_shape(inp)
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        
        if len(matches) < 2:
            return
        
        # Count adjacent objects for each object
        adj_count_to_color = {}
        for obj_in, obj_out in matches:
            # Count neighbors: objects whose bbox is within 2 cells
            adj = 0
            for other in in_objs:
                if other is obj_in:
                    continue
                # Check if adjacent (bboxes within 1 cell)
                r1, c1, r2, c2 = obj_in.bbox
                or1, oc1, or2, oc2 = other.bbox
                if (r1 - 1 <= or2 and or1 <= r2 + 1 and 
                    c1 - 1 <= oc2 and oc1 <= c2 + 1):
                    adj += 1
            
            new_color = obj_out.color
            if adj in adj_count_to_color:
                if adj_count_to_color[adj] != new_color:
                    return  # inconsistent
            else:
                adj_count_to_color[adj] = new_color
        
        if pair_idx == 0:
            first_map = adj_count_to_color.copy()
        else:
            if adj_count_to_color != first_map:
                return
    
    if first_map:
        _map = first_map
        _bg = bg
        pieces.append(CrossPiece(
            'recolor_by_adj_count',
            lambda inp, m=_map, b=_bg: _apply_recolor_by_adj_count(inp, m, b)
        ))


def _apply_recolor_by_adj_count(inp: Grid, adj_map: Dict, bg: int) -> Grid:
    from arc.objects import detect_objects, recolor_object
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    for obj in objs:
        adj = 0
        for other in objs:
            if other is obj:
                continue
            r1, c1, r2, c2 = obj.bbox
            or1, oc1, or2, oc2 = other.bbox
            if (r1 - 1 <= or2 and or1 <= r2 + 1 and 
                c1 - 1 <= oc2 and oc1 <= c2 + 1):
                adj += 1
        if adj in adj_map:
            for r, c in obj.cells:
                result[r][c] = adj_map[adj]
    return result


def _try_recolor_by_enclosure(pieces: List[CrossPiece],
                               train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Recolor objects based on whether they are enclosed by another object"""
    from arc.objects import detect_objects, find_matching_objects
    
    enclosed_color = None
    free_color = None
    
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        
        for obj_in, obj_out in matches:
            if obj_in.color == obj_out.color:
                continue
            
            # Check if obj_in is enclosed (surrounded by another object's cells)
            r1, c1, r2, c2 = obj_in.bbox
            is_enclosed = True
            
            # Check if there's a larger object surrounding this one
            for border_r in range(max(0, r1-1), min(h, r2+2)):
                for border_c in range(max(0, c1-1), min(w, c2+2)):
                    if (border_r, border_c) not in set(obj_in.cells):
                        if border_r in (r1-1, r2+1) or border_c in (c1-1, c2+1):
                            if inp[border_r][border_c] == bg:
                                is_enclosed = False
                                break
                if not is_enclosed:
                    break
            
            if is_enclosed:
                if enclosed_color is not None and enclosed_color != obj_out.color:
                    return
                enclosed_color = obj_out.color
            else:
                if free_color is not None and free_color != obj_out.color:
                    return
                free_color = obj_out.color
    
    if enclosed_color is not None or free_color is not None:
        _ec = enclosed_color
        _fc = free_color
        _bg = bg
        pieces.append(CrossPiece(
            'recolor_by_enclosure',
            lambda inp, ec=_ec, fc=_fc, b=_bg: _apply_recolor_by_enclosure(inp, ec, fc, b)
        ))


def _apply_recolor_by_enclosure(inp: Grid, enclosed_color, free_color, bg: int) -> Grid:
    from arc.objects import detect_objects
    h, w = grid_shape(inp)
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    for obj in objs:
        r1, c1, r2, c2 = obj.bbox
        is_enclosed = True
        for border_r in range(max(0, r1-1), min(h, r2+2)):
            for border_c in range(max(0, c1-1), min(w, c2+2)):
                if (border_r, border_c) not in set(obj.cells):
                    if border_r in (r1-1, r2+1) or border_c in (c1-1, c2+1):
                        if inp[border_r][border_c] == bg:
                            is_enclosed = False
                            break
            if not is_enclosed:
                break
        
        new_color = enclosed_color if is_enclosed else free_color
        if new_color is not None:
            for r, c in obj.cells:
                result[r][c] = new_color
    return result


def _try_selective_deletion(pieces: List[CrossPiece],
                            train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Delete objects matching a condition (size, color, shape)"""
    from arc.objects import detect_objects
    
    # Check: delete objects of specific size
    for selector in ['smallest', 'largest', 'by_color']:
        consistent = True
        delete_param = None
        
        for inp, out in train_pairs:
            in_objs = detect_objects(inp, bg)
            out_objs = detect_objects(out, bg)
            
            deleted_shapes = set()
            for obj in in_objs:
                # Check if this object is gone in output
                cells = set(obj.cells)
                still_there = False
                for oo in out_objs:
                    if set(oo.cells) & cells:
                        still_there = True
                        break
                if not still_there:
                    deleted_shapes.add(obj.size)
            
            if not deleted_shapes:
                consistent = False
                break
            
            if selector == 'smallest':
                min_size = min(o.size for o in in_objs)
                if deleted_shapes != {min_size}:
                    consistent = False; break
            elif selector == 'largest':
                max_size = max(o.size for o in in_objs)
                if deleted_shapes != {max_size}:
                    consistent = False; break
        
        if consistent and selector in ['smallest', 'largest']:
            _sel = selector
            _bg = bg
            pieces.append(CrossPiece(
                f'delete_{selector}_objects',
                lambda inp, sel=_sel, b=_bg: _apply_delete_by_selector(inp, sel, b)
            ))


def _apply_delete_by_selector(inp: Grid, selector: str, bg: int) -> Grid:
    from arc.objects import detect_objects
    objs = detect_objects(inp, bg)
    if not objs:
        return inp
    
    result = [row[:] for row in inp]
    if selector == 'smallest':
        target_size = min(o.size for o in objs)
    elif selector == 'largest':
        target_size = max(o.size for o in objs)
    else:
        return inp
    
    for obj in objs:
        if obj.size == target_size:
            for r, c in obj.cells:
                result[r][c] = bg
    return result


def _try_fill_object_interior(pieces: List[CrossPiece],
                               train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Fill hollow objects' interiors with their color"""
    from arc.objects import detect_objects
    
    consistent = True
    for inp, out in train_pairs:
        h, w = grid_shape(inp)
        in_objs = detect_objects(inp, bg)
        
        test_result = [row[:] for row in inp]
        for obj in in_objs:
            r1, c1, r2, c2 = obj.bbox
            # Fill bbox interior with object color where bg exists
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if inp[r][c] == bg:
                        # Check if surrounded by object cells (simple: within bbox)
                        test_result[r][c] = obj.color
        
        if not grid_eq(test_result, out):
            consistent = False
            break
    
    if consistent:
        _bg = bg
        pieces.append(CrossPiece(
            'fill_object_bbox_interior',
            lambda inp, b=_bg: _apply_fill_interior(inp, b)
        ))


def _apply_fill_interior(inp: Grid, bg: int) -> Grid:
    from arc.objects import detect_objects
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    for obj in objs:
        r1, c1, r2, c2 = obj.bbox
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                if result[r][c] == bg:
                    result[r][c] = obj.color
    return result


def _add_extract_pieces(pieces: List[CrossPiece],
                        train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Try: extract a specific object as output"""
    from arc.objects import detect_objects
    
    inp0, out0 = train_pairs[0]
    oh, ow = grid_shape(out0)
    ih, iw = grid_shape(inp0)
    
    if oh >= ih and ow >= iw:
        return  # output is not smaller, skip extraction
    
    objs = detect_objects(inp0, bg)
    
    # Try extracting each object that matches output size
    for obj in objs:
        if obj.height == oh and obj.width == ow:
            grid = obj.as_multicolor_grid(inp0)
            if grid_eq(grid, out0):
                _color = obj.color
                _bg = bg
                pieces.append(CrossPiece(
                    f'extract_object_color_{obj.color}',
                    lambda inp, col=_color, b=_bg: _apply_extract_by_color(inp, col, b)
                ))
    
    # Try extracting by rank within same color (e.g., 3rd largest object of color 8)
    by_color = {}
    for obj in objs:
        by_color.setdefault(obj.color, []).append(obj)
    
    for color, color_objs in by_color.items():
        sorted_objs = sorted(color_objs, key=lambda o: o.size, reverse=True)
        for rank, obj in enumerate(sorted_objs):
            if obj.height == oh and obj.width == ow:
                grid = obj.as_multicolor_grid(inp0)
                if grid_eq(grid, out0):
                    _color = color
                    _rank = rank
                    _bg = bg
                    pieces.append(CrossPiece(
                        f'extract_color{color}_rank{rank}',
                        lambda inp, col=_color, rnk=_rank, b=_bg: _apply_extract_by_rank(inp, col, rnk, b)
                    ))
    
    # Try extracting object that contains a unique color
    for obj in objs:
        grid = obj.as_multicolor_grid(inp0)
        gh, gw = grid_shape(grid)
        if gh == oh and gw == ow and grid_eq(grid, out0):
            # Find what makes this object unique
            obj_colors = set()
            for r, c in obj.cells:
                obj_colors.add(inp0[r][c])
            
            for uc in obj_colors:
                # Check if this color only appears in this object's bbox
                unique_to_obj = True
                for r in range(ih):
                    for c in range(iw):
                        if inp0[r][c] == uc:
                            if not (obj.bbox[0] <= r <= obj.bbox[2] and 
                                    obj.bbox[1] <= c <= obj.bbox[3]):
                                unique_to_obj = False
                                break
                    if not unique_to_obj:
                        break
                
                if unique_to_obj and uc != bg:
                    _uc = uc
                    _bg = bg
                    pieces.append(CrossPiece(
                        f'extract_containing_color_{uc}',
                        lambda inp, col=_uc, b=_bg: _apply_extract_containing(inp, col, b)
                    ))
    
    # Try multicolor object extraction
    mc_objs = detect_objects(inp0, bg, multicolor=True)
    for obj in mc_objs:
        grid = obj.as_multicolor_grid(inp0)
        if grid_shape(grid) == (oh, ow) and grid_eq(grid, out0):
            _size = obj.size
            _bg = bg
            pieces.append(CrossPiece(
                f'extract_multicolor_size{obj.size}',
                lambda inp, sz=_size, b=_bg: _apply_extract_multicolor_by_size(inp, sz, b)
            ))


def _apply_extract_by_color(inp: Grid, color: int, bg: int) -> Optional[Grid]:
    from arc.objects import detect_objects
    objs = [o for o in detect_objects(inp, bg) if o.color == color]
    if not objs:
        return None
    obj = max(objs, key=lambda o: o.size)
    return obj.as_multicolor_grid(inp)


def _apply_extract_by_rank(inp: Grid, color: int, rank: int, bg: int) -> Optional[Grid]:
    from arc.objects import detect_objects
    objs = [o for o in detect_objects(inp, bg) if o.color == color]
    objs = sorted(objs, key=lambda o: o.size, reverse=True)
    if rank >= len(objs):
        return None
    return objs[rank].as_multicolor_grid(inp)


def _apply_extract_containing(inp: Grid, unique_color: int, bg: int) -> Optional[Grid]:
    from arc.objects import detect_objects
    objs = detect_objects(inp, bg, multicolor=True)
    for obj in objs:
        h, w = grid_shape(inp)
        r1, c1, r2, c2 = obj.bbox
        has_color = False
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                if inp[r][c] == unique_color:
                    has_color = True
                    break
            if has_color:
                break
        if has_color:
            return obj.as_multicolor_grid(inp)
    return None


def _apply_extract_multicolor_by_size(inp: Grid, target_size: int, bg: int) -> Optional[Grid]:
    from arc.objects import detect_objects
    objs = detect_objects(inp, bg, multicolor=True)
    # Find object closest in size
    best = None
    best_diff = float('inf')
    for obj in objs:
        diff = abs(obj.size - target_size)
        if diff < best_diff:
            best_diff = diff
            best = obj
    if best is None:
        return None
    return best.as_multicolor_grid(inp)


def solve_cross_engine(train_pairs: List[Tuple[Grid, Grid]], 
                       test_inputs: List[Grid]) -> Tuple[List[List[Grid]], List]:
    """
    Full Cross-Structure solver with puzzle reasoning.
    
    Phase 1: Try original cross_solver (existing DSL)
    Phase 2: Try cross pieces (abstract NB, conditional, objects)
    Phase 3: Try 2-step composition of cross pieces
    Phase 4: Puzzle reasoning — partial match + refinement
    """
    sim = CrossSimulator()
    verified = []
    
    # === Phase 1: Original solver (DSL + NB rule) ===
    orig_predictions, orig_verified = solve_cross(train_pairs, test_inputs)
    verified.extend(orig_verified)
    
    # Always try cross pieces (even if Phase 1 found 2+ candidates)
    # This allows cross_nb and object DSL to be fallback candidates
    cross_pieces = _generate_cross_pieces(train_pairs)
    
    for piece in cross_pieces:
        if sim.verify(piece, train_pairs):
            # Avoid duplicates by name
            existing_names = {getattr(p, 'name', '') for _, p in verified}
            if piece.name not in existing_names:
                verified.append(('cross', piece))
    
    if len(verified) >= 2:
        return _apply_verified(verified, test_inputs), verified
    
    # === Phase 3: Composition of cross pieces with WG programs ===
    if len(verified) < 2 and cross_pieces:
        wg_cands = _generate_whole_grid_candidates(train_pairs)
        
        # Try: cross_piece + WG
        for cp in cross_pieces:
            if len(verified) >= 2:
                break
            mid0 = cp.apply(train_pairs[0][0])
            if mid0 is None:
                continue
            
            for wg in wg_cands:
                res0 = wg.apply(mid0)
                if res0 is None or not grid_eq(res0, train_pairs[0][1]):
                    continue
                
                # Full verify
                ok = True
                for inp, exp in train_pairs[1:]:
                    mid = cp.apply(inp)
                    if mid is None:
                        ok = False; break
                    res = wg.apply(mid)
                    if res is None or not grid_eq(res, exp):
                        ok = False; break
                if ok:
                    verified.append(('cross_compose', (cp, wg)))
                    if len(verified) >= 2:
                        break
        
        # Try: WG + cross_piece
        if len(verified) < 2:
            for wg in wg_cands:
                if len(verified) >= 2:
                    break
                mid0 = wg.apply(train_pairs[0][0])
                if mid0 is None:
                    continue
                
                for cp in cross_pieces:
                    res0 = cp.apply(mid0)
                    if res0 is None or not grid_eq(res0, train_pairs[0][1]):
                        continue
                    
                    ok = True
                    for inp, exp in train_pairs[1:]:
                        mid = wg.apply(inp)
                        if mid is None:
                            ok = False; break
                        res = cp.apply(mid)
                        if res is None or not grid_eq(res, exp):
                            ok = False; break
                    if ok:
                        verified.append(('cross_compose', (wg, cp)))
                        if len(verified) >= 2:
                            break
    
    return _apply_verified(verified, test_inputs), verified


def _apply_verified(verified: List, test_inputs: List[Grid]) -> List[List[Grid]]:
    """Apply verified programs to test inputs.
    
    Selects up to 3 diverse candidates:
    - Up to 2 from Phase 1 (original solver)
    - At least 1 non-NB cross piece if available
    """
    # Prioritize diversity: separate NB-like from non-NB candidates
    nb_names = {'neighborhood_rule', 'cross_nb_r1', 'cross_nb_r2', 
                'structural_nb_r1', 'structural_nb_r2',
                'abstract_nb_r1', 'abstract_nb_r2',
                'count_based_nb'}
    
    nb_verified = []
    non_nb_verified = []
    for item in verified:
        kind, prog = item
        name = getattr(prog, 'name', '')
        if name in nb_names or name.startswith('cross_nb') or name.startswith('structural_nb') or name.startswith('abstract_nb'):
            nb_verified.append(item)
        else:
            non_nb_verified.append(item)
    
    # Build final candidate list: original DSL first, then NB, then cross non-NB
    selected = non_nb_verified[:2] + nb_verified[:2]
    # Deduplicate and limit to 3
    seen = set()
    final = []
    for item in selected:
        name = getattr(item[1], 'name', id(item[1]))
        if name not in seen:
            seen.add(name)
            final.append(item)
        if len(final) >= 3:
            break
    
    # If < 3, add remaining
    for item in verified:
        name = getattr(item[1], 'name', id(item[1]))
        if name not in seen:
            seen.add(name)
            final.append(item)
        if len(final) >= 3:
            break
    
    predictions = []
    for test_inp in test_inputs:
        attempts = []
        for kind, prog in final:
            if kind in ('cell', 'whole'):
                result = prog.apply(test_inp)
            elif kind == 'composite':
                result = prog.apply(test_inp)
            elif kind == 'cross':
                result = prog.apply(test_inp)
            elif kind == 'cross_compose':
                p1, p2 = prog
                mid = p1.apply(test_inp) if hasattr(p1, 'apply') else None
                result = p2.apply(mid) if mid is not None else None
            else:
                result = None
            
            if result is not None:
                # Deduplicate identical predictions
                if not any(grid_eq(result, a) for a in attempts):
                    attempts.append(result)
        predictions.append(attempts)
    
    return predictions
