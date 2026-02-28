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
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors
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
    learn_rotation_invariant_nb_rule, apply_rotation_invariant_nb_rule,
    learn_rotsym_count_nb_rule, apply_rotsym_count_nb_rule,
)
from arc.conditional import (
    learn_conditional_color_rule, apply_conditional_color_rule,
    learn_region_property_rule, apply_region_property_rule,
)
from arc.conditional_transform import (
    learn_conditional_object_transform, apply_conditional_object_transform,
)
from arc.objects import detect_objects, find_matching_objects, object_transform_type
from arc.flood_fill import learn_flood_fill_region, apply_flood_fill_region
from arc.stripe_fill_solver import learn_stripe_fill_rule, apply_stripe_fill_rule
from arc.cross_plus_solver import learn_cross_plus_rule, apply_cross_plus_rule
from arc.extract_summary import learn_fixed_output_summary, apply_fixed_output_summary
from arc.grow_primitives import (
    learn_grow_via_self_stamp, apply_grow_via_self_stamp,
    learn_grow_color_template, apply_grow_color_template,
    learn_grow_fixed_color_template, apply_grow_fixed_color_template,
)
from arc.line_ray_primitives import (
    learn_line_ray_from_objects, apply_line_ray_from_objects,
    learn_fill_object_interior, apply_fill_object_interior,
)


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
    
    # Priority 3.5: Rotation-invariant NB (best generalization for unseen patterns)
    rule = learn_rotation_invariant_nb_rule(train_pairs, radius=1)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            'rot_inv_nb',
            lambda inp, _r=r: apply_rotation_invariant_nb_rule(inp, _r)
        ))
    
    # Priority 3.7: Ultra-coarse rotation-symmetric count NB
    rule = learn_rotsym_count_nb_rule(train_pairs)
    if rule is not None:
        r = rule
        pieces.append(CrossPiece(
            'rotsym_count_nb',
            lambda inp, _r=r: apply_rotsym_count_nb_rule(inp, _r)
        ))

    # Count-based rule
    rule = learn_count_based_rule(train_pairs)
    if rule is not None:
        r = rule
        pieces.append(CrossPiece(
            'count_based_nb',
            lambda inp, _r=r: apply_count_based_rule(inp, _r)
        ))
    
    # === Module 1b: Extended NB Rules (count-based, between, multipass) ===
    try:
        from arc.nb_extended import ALL_EXTENDED_NB
        for ename, elearn, eapply in ALL_EXTENDED_NB:
            try:
                erule = elearn(train_pairs)
                if erule is not None:
                    _er = erule
                    _ea = eapply
                    pieces.insert(0, CrossPiece(
                        f'ext_nb:{ename}',
                        lambda inp, r=_er, fn=_ea: fn(inp, r)
                    ))
            except Exception:
                pass
    except ImportError:
        pass
    
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

    # === Module 2b: Conditional Object Transforms ===
    # Per-object conditional transforms: fill interior, line extension, bg->color
    rule = learn_conditional_object_transform(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            f'conditional_obj:{r.get("strategy", r.get("type"))}',
            lambda inp, _r=r: apply_conditional_object_transform(inp, _r)
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
        
        # Try: draw lines between same-color objects
        pieces.append(CrossPiece('draw_lines_same_color',
            lambda inp: _draw_lines_same_color(inp)))

        # Try: flood fill enclosed bg regions (legacy)
        pieces.append(CrossPiece('flood_fill_enclosed',
            lambda inp: _flood_fill_enclosed(inp)))

    # === Module 24: Flood Fill Region (new primitive) ===
    rule = learn_flood_fill_region(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            f'flood_fill:{rule["type"]}',
            lambda inp, _r=r: apply_flood_fill_region(inp, _r)
        ))

    # === Module 24b: Stripe Fill (row/column fill from colored dots) ===
    rule = learn_stripe_fill_rule(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            f'stripe_fill:{rule["type"]}',
            lambda inp, _r=r: apply_stripe_fill_rule(inp, _r)
        ))

    # === Module 24c: Cross/Plus Pattern (dots create cross patterns) ===
    rule = learn_cross_plus_rule(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            f'cross_plus:{rule["type"]}',
            lambda inp, _r=r: apply_cross_plus_rule(inp, _r)
        ))

    # === Module 25: Fixed-Output Summary (extract fixed-size output from variable input) ===
    rule = learn_fixed_output_summary(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            f'extract_summary:{rule["type"]}',
            lambda inp, _r=r: apply_fixed_output_summary(inp, _r)
        ))

    # === Module 26: Grow/Scale primitives (NxN expansion with templates) ===
    rule = learn_grow_via_self_stamp(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            'grow:self_stamp',
            lambda inp, _r=r: apply_grow_via_self_stamp(inp, _r)
        ))

    rule = learn_grow_fixed_color_template(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            'grow:color_template',
            lambda inp, _r=r: apply_grow_fixed_color_template(inp, _r)
        ))

    rule = learn_grow_color_template(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            'grow:position_template',
            lambda inp, _r=r: apply_grow_color_template(inp, _r)
        ))

    # === Module 27: Line/Ray extension and object interior fill ===
    rule = learn_line_ray_from_objects(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            f'line_ray:{rule["type"]}',
            lambda inp, _r=r: apply_line_ray_from_objects(inp, _r)
        ))

    rule = learn_fill_object_interior(train_pairs)
    if rule is not None:
        r = rule
        pieces.insert(0, CrossPiece(
            f'fill_interior:{rule["type"]}',
            lambda inp, _r=r: apply_fill_object_interior(inp, _r)
        ))

    # === Module 6: Object Correspondence + Conditional Transform ===
    _add_obj_correspondence_pieces(pieces, train_pairs, bg)
    
    # === Module 7: Per-Object Stamp Pattern ===
    _add_per_object_stamp_pieces(pieces, train_pairs)
    
    # === Module 8: Object Position/Placement Transform ===
    _add_obj_transform_pieces(pieces, train_pairs)
    
    # === Module 9: Patch Extraction (shrink tasks) ===
    _add_extract_patch_pieces(pieces, train_pairs)
    
    # === Module 10: Tile + Transform (expand tasks) ===
    _add_tile_transform_pieces(pieces, train_pairs)
    
    # === Module 5: Cross-Compose (multi-step decomposition) ===
    _add_cross_compose_pieces(pieces, train_pairs)
    
    # === Module 4: Grid partition transforms ===
    _add_partition_pieces(pieces, train_pairs)
    
    # Try: extract specific object
    _add_extract_pieces(pieces, train_pairs, bg)
    
    # === Module 11: Grid-to-Summary (block classifier, fold, count) ===
    _add_grid_summary_pieces(pieces, train_pairs)
    
    # === Module 12: Line drawing / connection ===
    _add_line_connect_pieces(pieces, train_pairs)
    
    # === Module 13: Panel operations (split + reduce/select) ===
    _add_panel_ops_pieces(pieces, train_pairs)
    
    # === Module 14: Per-object conditional transforms ===
    _add_per_object_pieces(pieces, train_pairs)
    
    # === Module 15: Gravity ===
    _add_gravity_pieces(pieces, train_pairs)
    
    # === Module 23: Self-stamp ===
    _add_self_stamp_pieces(pieces, train_pairs)
    
    # === Module 16: Symmetry completion ===
    _add_symmetry_pieces(pieces, train_pairs)
    
    # === Module 17: Frame extraction ===
    _add_frame_extract_pieces(pieces, train_pairs)
    
    # === Module 18: Mask/overlay operations ===
    _add_mask_pieces(pieces, train_pairs)
    
    # === Module 19: Residual learners (fill between, project, cross dots, etc.) ===
    _add_residual_pieces(pieces, train_pairs)
    
    # Note: Module 20 primitives are generated separately in _generate_cross_pieces_fast
    # for beam search. Adding them here would slow down Phase 3 composition (O(N²)).
    
    # === Module 21: Residual-Guided pieces (reverse-direction analysis) ===
    try:
        from arc.residual_guided import generate_residual_guided_pieces
        rg_pieces = generate_residual_guided_pieces(train_pairs)
        for rg in rg_pieces:
            pieces.insert(0, rg)  # High priority — targeted by residual analysis
    except Exception:
        pass
    
    # === Module 22: Meta-Cross (concept→role→operation hierarchical routing) ===
    try:
        from arc.meta_cross import generate_meta_cross_pieces
        mc_pieces = generate_meta_cross_pieces(train_pairs)
        for mc in mc_pieces:
            pieces.insert(0, mc)  # Highest priority — concept-guided
    except Exception:
        pass
    
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
    
    # Try: semantic subgrid extraction (selector-based)
    _add_semantic_extract_pieces(pieces, train_pairs, bg)
    
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


def _draw_lines_same_color(inp: Grid) -> Grid:
    """Draw horizontal/vertical lines between same-color objects"""
    from arc.objects import detect_objects
    bg = most_common_color(inp)
    h, w = grid_shape(inp)
    objs = detect_objects(inp, bg)
    result = [row[:] for row in inp]
    
    by_color = {}
    for o in objs:
        by_color.setdefault(o.color, []).append(o)
    
    for color, group in by_color.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                cr1 = int(group[i].center[0])
                cc1 = int(group[i].center[1])
                cr2 = int(group[j].center[0])
                cc2 = int(group[j].center[1])
                if cr1 == cr2:
                    for c in range(min(cc1, cc2), max(cc1, cc2) + 1):
                        if result[cr1][c] == bg:
                            result[cr1][c] = color
                elif cc1 == cc2:
                    for r in range(min(cr1, cr2), max(cr1, cr2) + 1):
                        if result[r][cc1] == bg:
                            result[r][cc1] = color
    return result


def _flood_fill_enclosed(inp: Grid) -> Grid:
    """Fill enclosed bg regions with the surrounding non-bg color"""
    bg = most_common_color(inp)
    h, w = grid_shape(inp)
    
    visited = [[False] * w for _ in range(h)]
    result = [row[:] for row in inp]
    
    # BFS from border to mark all border-connected bg
    border_bg = set()
    queue = []
    for r in range(h):
        for c in [0, w - 1]:
            if inp[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
                border_bg.add((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if inp[r][c] == bg and not visited[r][c]:
                visited[r][c] = True
                queue.append((r, c))
                border_bg.add((r, c))
    
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and inp[nr][nc] == bg:
                visited[nr][nc] = True
                queue.append((nr, nc))
                border_bg.add((nr, nc))
    
    # Fill interior bg
    for r in range(h):
        for c in range(w):
            if inp[r][c] == bg and (r, c) not in border_bg:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] != bg:
                        result[r][c] = inp[nr][nc]
                        break
    return result


def _add_tile_transform_pieces(pieces: List[CrossPiece],
                                train_pairs: List[Tuple[Grid, Grid]]):
    """Add tile+transform pieces for expand tasks"""
    from arc.tile_transform import learn_tile_transform, apply_tile_transform
    
    rule = learn_tile_transform(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'tile:{rule["name"]}',
            lambda inp, r=_rule: apply_tile_transform(inp, r)
        ))


def _add_extract_patch_pieces(pieces: List[CrossPiece],
                               train_pairs: List[Tuple[Grid, Grid]]):
    """Add patch extraction pieces for shrink tasks"""
    from arc.extract_patch import learn_extract_rule, apply_extract_rule
    
    rule = learn_extract_rule(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'extract:{rule["name"]}',
            lambda inp, r=_rule: apply_extract_rule(inp, r)
        ))


def _add_obj_transform_pieces(pieces: List[CrossPiece],
                               train_pairs: List[Tuple[Grid, Grid]]):
    """Add object position/placement transform pieces"""
    from arc.obj_transform import learn_obj_transform, apply_obj_transform
    
    rule = learn_obj_transform(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'transform:{rule["name"]}',
            lambda inp, r=_rule: apply_obj_transform(inp, r)
        ))


def _add_per_object_stamp_pieces(pieces: List[CrossPiece],
                                  train_pairs: List[Tuple[Grid, Grid]]):
    """Add per-object stamp pattern pieces"""
    from arc.per_object_stamp import learn_per_object_stamp, apply_per_object_stamp
    
    rule = learn_per_object_stamp(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'stamp:{rule["name"]}',
            lambda inp, r=_rule: apply_per_object_stamp(inp, r)
        ))


def _add_obj_correspondence_pieces(pieces: List[CrossPiece],
                                    train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Add Object Correspondence-based transform pieces"""
    from arc.obj_correspondence import learn_object_program, apply_object_program
    
    prog = learn_object_program(train_pairs)
    if prog is not None:
        _prog = prog
        pieces.append(CrossPiece(
            f'obj:{prog["name"]}',
            lambda inp, p=_prog: apply_object_program(inp, p)
        ))


def _add_cross_compose_pieces(pieces: List[CrossPiece],
                               train_pairs: List[Tuple[Grid, Grid]]):
    """Add Cross-Structure multi-step programs"""
    from arc.cross_compose import learn_cross_program
    
    prog = learn_cross_program(train_pairs)
    if prog is not None:
        _prog = prog
        pieces.append(CrossPiece(
            f'cross_compose:{prog.name}',
            lambda inp, p=_prog: p.apply(inp)
        ))


def _add_partition_pieces(pieces: List[CrossPiece],
                          train_pairs: List[Tuple[Grid, Grid]]):
    """Add grid partition-based transform pieces"""
    from arc.grid_partition import learn_panel_transform, apply_panel_transform
    
    rule = learn_panel_transform(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'panel_{rule["type"]}',
            lambda inp, r=_rule: apply_panel_transform(inp, r)
        ))


def _add_grid_summary_pieces(pieces: List[CrossPiece],
                             train_pairs: List[Tuple[Grid, Grid]]):
    """Add grid-to-summary pieces (block classifier, fold, count)"""
    from arc.grid_summarize import (
        learn_block_classifier, apply_block_classifier,
        learn_fold_rule, apply_fold_rule,
        learn_count_summary, apply_count_summary,
    )
    
    # Block classifier (e.g., 11x11 → 3x3)
    rule = learn_block_classifier(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'block_classify:{rule["classifier"]}',
            lambda inp, r=_rule: apply_block_classifier(r, inp)
        ))
    
    # Fold rule (e.g., 8x9 → 8x2 by folding column groups)
    rule = learn_fold_rule(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'fold:{rule["type"]}:{rule["op"]}',
            lambda inp, r=_rule: apply_fold_rule(r, inp)
        ))
    
    # Count summary (e.g., NxM → Kx1)
    rule = learn_count_summary(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'count:{rule["type"]}',
            lambda inp, r=_rule: apply_count_summary(r, inp)
        ))


def _add_line_connect_pieces(pieces: List[CrossPiece],
                             train_pairs: List[Tuple[Grid, Grid]]):
    """Add line drawing/connection pieces"""
    from arc.line_connect import (
        learn_l_connect, apply_l_connect,
        learn_cross_projection, apply_cross_projection,
        learn_line_extension, apply_line_extension,
        learn_connect_objects, apply_connect_objects,
    )
    
    # L-shape connection
    rule = learn_l_connect(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'l_connect:{rule["order"]}',
            lambda inp, r=_rule: apply_l_connect(r, inp)
        ))
    
    # Cross projection from dots
    rule = learn_cross_projection(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'cross_project:{rule["strategy"]}',
            lambda inp, r=_rule: apply_cross_projection(r, inp)
        ))
    
    # Line extension to border
    rule = learn_line_extension(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'line_extend:{rule["type"]}',
            lambda inp, r=_rule: apply_line_extension(r, inp)
        ))
    
    # Connect same-color objects
    rule = learn_connect_objects(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'connect_objs:{rule["connect"]}',
            lambda inp, r=_rule: apply_connect_objects(r, inp)
        ))


def _add_semantic_extract_pieces(pieces: List[CrossPiece],
                                  train_pairs: List[Tuple[Grid, Grid]], bg: int):
    """Add extraction pieces based on semantic selectors (not specific colors)"""
    from arc.objects import detect_objects
    from collections import Counter
    
    inp0, out0 = train_pairs[0]
    ih, iw = grid_shape(inp0)
    oh, ow = grid_shape(out0)
    
    if oh >= ih and ow >= iw:
        return
    
    # Selector: minority/majority color bbox
    for sel_name in ['minority', 'majority']:
        def _make_fn(selector):
            def fn(inp):
                h, w = grid_shape(inp)
                bg_local = most_common_color(inp)
                cc = {}
                for r in range(h):
                    for c in range(w):
                        v = inp[r][c]
                        if v != bg_local:
                            cc[v] = cc.get(v, 0) + 1
                if not cc:
                    return None
                color = min(cc, key=cc.get) if selector == 'minority' else max(cc, key=cc.get)
                return _apply_extract_color_bbox(inp, color)
            return fn
        pieces.append(CrossPiece(f'extract_{sel_name}_bbox', _make_fn(sel_name)))
    
    # Selector: Nth largest object
    for n in range(min(5, len(detect_objects(inp0, bg)))):
        _n = n
        pieces.append(CrossPiece(
            f'extract_obj_rank_{n}',
            lambda inp, rank=_n: _extract_nth_obj(inp, rank, False)
        ))
    
    # Selector: Nth largest multicolor object
    for n in range(min(3, len(detect_objects(inp0, bg, multicolor=True)))):
        _n = n
        pieces.append(CrossPiece(
            f'extract_mc_obj_rank_{n}',
            lambda inp, rank=_n: _extract_nth_obj(inp, rank, True)
        ))
    
    # Selector: unique shape object
    pieces.append(CrossPiece('extract_unique_shape', _extract_unique_shape))
    
    # Selector: unique color object
    pieces.append(CrossPiece('extract_unique_color', _extract_unique_color))
    
    # Selector: multicolor region with most distinct colors
    pieces.append(CrossPiece('extract_most_colors_mc', _extract_most_colors_mc))
    
    # Selector: densest object (highest fill ratio)
    pieces.append(CrossPiece('extract_densest', _extract_densest))


def _extract_nth_obj(inp: Grid, n: int, multicolor: bool) -> Optional[Grid]:
    from arc.objects import detect_objects
    bg = most_common_color(inp)
    objs = detect_objects(inp, bg, multicolor=multicolor)
    if n >= len(objs):
        return None
    return objs[n].as_multicolor_grid(inp, bg)


def _extract_unique_shape(inp: Grid) -> Optional[Grid]:
    from arc.objects import detect_objects
    from collections import Counter
    bg = most_common_color(inp)
    objs = detect_objects(inp, bg)
    shape_counts = Counter(o.shape for o in objs)
    for o in objs:
        if shape_counts[o.shape] == 1:
            return o.as_multicolor_grid(inp, bg)
    return None


def _extract_unique_color(inp: Grid) -> Optional[Grid]:
    from arc.objects import detect_objects
    from collections import Counter
    bg = most_common_color(inp)
    objs = detect_objects(inp, bg)
    color_counts = Counter(o.color for o in objs)
    for o in objs:
        if color_counts[o.color] == 1:
            return o.as_multicolor_grid(inp, bg)
    return None


def _extract_most_colors_mc(inp: Grid) -> Optional[Grid]:
    from arc.objects import detect_objects
    bg = most_common_color(inp)
    objs = detect_objects(inp, bg, multicolor=True)
    if not objs:
        return None
    best = max(objs, key=lambda o: len(set(inp[r][c] for r, c in o.cells)))
    return best.as_multicolor_grid(inp, bg)


def _extract_densest(inp: Grid) -> Optional[Grid]:
    from arc.objects import detect_objects
    bg = most_common_color(inp)
    objs = detect_objects(inp, bg)
    if not objs:
        return None
    best = max(objs, key=lambda o: o.size / max(o.height * o.width, 1))
    return best.as_multicolor_grid(inp, bg)


def _apply_extract_color_bbox(inp: Grid, color: int) -> Optional[Grid]:
    """Extract the bounding box of all cells of given color"""
    h, w = grid_shape(inp)
    rows = [r for r in range(h) for c in range(w) if inp[r][c] == color]
    cols = [c for r in range(h) for c in range(w) if inp[r][c] == color]
    if not rows:
        return None
    r1, r2 = min(rows), max(rows)
    c1, c2 = min(cols), max(cols)
    return [inp[r][c1:c2+1] for r in range(r1, r2+1)]


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


# ============================================================
# Module 13-18: New modules (panel_ops, per_object, gravity, symmetry, frame, mask)
# ============================================================

def _add_panel_ops_pieces(pieces: List[CrossPiece],
                          train_pairs: List[Tuple[Grid, Grid]]):
    """Module 13: Panel split + reduce/select"""
    from arc.panel_ops import learn_panel_reduce, apply_panel_reduce
    
    rule = learn_panel_reduce(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'panel_reduce:{_rule["op"]}',
            lambda inp, r=_rule: apply_panel_reduce(inp, r)
        ))


def _add_per_object_pieces(pieces: List[CrossPiece],
                           train_pairs: List[Tuple[Grid, Grid]]):
    """Module 14: Per-object conditional transforms"""
    from arc.per_object import (
        learn_per_object_recolor, apply_per_object_recolor,
        learn_fill_object_bbox, apply_fill_object_bbox,
        learn_remove_objects, apply_remove_objects,
        learn_cross_projection, apply_cross_projection,
        learn_extract_object, apply_extract_object,
        learn_holes_to_color, apply_holes_to_color,
        learn_cluster_histogram, apply_cluster_histogram,
        learn_dynamic_tile, apply_dynamic_tile,
        learn_cell_to_color_block, apply_cell_to_color_block,
        learn_color_to_pattern, apply_color_to_pattern,
    )
    
    # Per-object recolor
    rule = learn_per_object_recolor(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'per_obj_recolor:{_rule["type"]}',
            lambda inp, r=_rule: apply_per_object_recolor(inp, r)
        ))
    
    # Fill object bbox
    rule = learn_fill_object_bbox(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            'fill_obj_bbox',
            lambda inp, r=_rule: apply_fill_object_bbox(inp, r)
        ))
    
    # Remove objects
    rule = learn_remove_objects(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'remove_obj:{_rule["remove_by"]}',
            lambda inp, r=_rule: apply_remove_objects(inp, r)
        ))
    
    # Cross projection
    rule = learn_cross_projection(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            'cross_projection',
            lambda inp, r=_rule: apply_cross_projection(inp, r)
        ))
    
    # Extract object
    rule = learn_extract_object(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            f'extract_obj:{_rule["selector"]}',
            lambda inp, r=_rule: apply_extract_object(inp, r)
        ))

    # Holes-to-color: recolor objects by number of enclosed holes
    rule = learn_holes_to_color(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            'holes_to_color',
            lambda inp, r=_rule: apply_holes_to_color(inp, r)
        ))

    # Cluster histogram: histogram of cluster counts per color
    rule = learn_cluster_histogram(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            'cluster_histogram',
            lambda inp, r=_rule: apply_cluster_histogram(inp, r)
        ))

    # Dynamic tile: tile count depends on input properties
    rule = learn_dynamic_tile(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'dynamic_tile:{_rule["rule"]}',
            lambda inp, r=_rule: apply_dynamic_tile(inp, r)
        ))

    # Cell to color block: each cell → KxK block, K = n_unique_colors
    if learn_cell_to_color_block(train_pairs):
        pieces.insert(0, CrossPiece(
            'cell_to_color_block',
            lambda inp: apply_cell_to_color_block(inp)
        ))

    # Color to KxK pattern: each color → fixed KxK block
    rule = learn_color_to_pattern(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'color_to_pattern_k{_rule["k"]}',
            lambda inp, r=_rule: apply_color_to_pattern(inp, r)
        ))


def _add_self_stamp_pieces(pieces: List[CrossPiece],
                           train_pairs: List[Tuple[Grid, Grid]]):
    """Module 23: Self-stamp (stamp input at positions of specific color)"""
    from arc.per_object import learn_self_stamp, apply_self_stamp
    sc = learn_self_stamp(train_pairs)
    if sc is not None:
        pieces.insert(0, CrossPiece(
            f'self_stamp_c{sc}',
            lambda inp, _sc=sc: apply_self_stamp(inp, _sc)
        ))


def _add_gravity_pieces(pieces: List[CrossPiece],
                        train_pairs: List[Tuple[Grid, Grid]]):
    """Module 15: Gravity operations"""
    from arc.panel_ops import (
        learn_gravity, apply_gravity,
        learn_gravity_with_obstacles, apply_gravity_with_obstacles,
    )
    
    rule = learn_gravity(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'gravity:{_rule["direction"]}',
            lambda inp, r=_rule: apply_gravity(inp, r)
        ))
        return  # Don't try obstacles if simple gravity works
    
    rule = learn_gravity_with_obstacles(train_pairs)
    if rule is not None:
        _rule = rule
        mc = _rule.get('moving_color', 'all')
        pieces.insert(0, CrossPiece(
            f'gravity_obs:{_rule["direction"]}_mc{mc}',
            lambda inp, r=_rule: apply_gravity_with_obstacles(inp, r)
        ))


def _add_symmetry_pieces(pieces: List[CrossPiece],
                         train_pairs: List[Tuple[Grid, Grid]]):
    """Module 16: Symmetry completion"""
    from arc.panel_ops import learn_symmetry_fill, apply_symmetry_fill
    
    rule = learn_symmetry_fill(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'symmetry_fill:{_rule["sym_type"]}',
            lambda inp, r=_rule: apply_symmetry_fill(inp, r)
        ))


def _add_frame_extract_pieces(pieces: List[CrossPiece],
                              train_pairs: List[Tuple[Grid, Grid]]):
    """Module 17: Frame-based extraction"""
    from arc.panel_ops import (
        learn_extract_by_frame, apply_extract_by_frame,
        learn_crop_to_objects, apply_crop_to_objects,
    )
    
    rule = learn_extract_by_frame(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'extract_frame:color{_rule["frame_color"]}',
            lambda inp, r=_rule: apply_extract_by_frame(inp, r)
        ))
    
    rule = learn_crop_to_objects(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.append(CrossPiece(
            'crop_to_objects',
            lambda inp, r=_rule: apply_crop_to_objects(inp, r)
        ))


def _add_mask_pieces(pieces: List[CrossPiece],
                     train_pairs: List[Tuple[Grid, Grid]]):
    """Module 18: Mask/overlay between grid halves"""
    from arc.per_object import learn_mask_apply, apply_mask_apply
    
    rule = learn_mask_apply(train_pairs)
    if rule is not None:
        _rule = rule
        pieces.insert(0, CrossPiece(
            f'mask_apply:{_rule["split_dir"]}',
            lambda inp, r=_rule: apply_mask_apply(inp, r)
        ))


def _add_residual_pieces(pieces: List[CrossPiece],
                         train_pairs: List[Tuple[Grid, Grid]]):
    """Module 19: Residual learners"""
    from arc.residual_learner import ALL_LEARNERS
    
    for name, learn_fn, apply_fn in ALL_LEARNERS:
        try:
            rule = learn_fn(train_pairs)
            if rule is not None:
                _rule = rule
                _apply = apply_fn
                pieces.insert(0, CrossPiece(
                    f'residual:{name}',
                    lambda inp, r=_rule, fn=_apply: fn(inp, r)
                ))
        except Exception:
            pass


def _add_primitive_pieces(pieces: List[CrossPiece],
                          train_pairs: List[Tuple[Grid, Grid]]):
    """Module 20: Parameterless primitives from primitives.py.
    
    These are simple atomic transforms (flip, rotate, fill, draw, etc.)
    that serve as building blocks for beam search composition.
    Only add primitives that produce a different output than input on train[0].
    """
    try:
        from arc.primitives import PARAMETERLESS_PRIMITIVES, get_color_primitives
    except ImportError:
        return
    
    inp0, out0 = train_pairs[0]
    h0, w0 = grid_shape(inp0)
    
    # Parameterless primitives
    for name, fn in PARAMETERLESS_PRIMITIVES:
        try:
            result = fn(inp0)
            if result is None:
                continue
            rh, rw = grid_shape(result)
            # Only add if it actually changes something
            if (rh, rw) == (h0, w0) and grid_eq(result, inp0):
                continue
            pieces.append(CrossPiece(name, lambda inp, _fn=fn: _fn(inp)))
        except Exception:
            pass
    
    # Color-parameterized primitives  
    try:
        color_prims = get_color_primitives(inp0)
        for name, fn in color_prims:
            try:
                result = fn(inp0)
                if result is None:
                    continue
                rh, rw = grid_shape(result)
                if (rh, rw) == (h0, w0) and grid_eq(result, inp0):
                    continue
                pieces.append(CrossPiece(name, lambda inp, _fn=fn: _fn(inp)))
            except Exception:
                pass
    except Exception:
        pass


def _generate_cross_pieces_fast(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Lightweight piece generator for beam search — only fast primitives.
    
    Excludes slow learned rules (NB, object detection, etc.)
    Includes only parameterless primitives + residual learners.
    """
    pieces = []
    
    try:
        from arc.primitives import PARAMETERLESS_PRIMITIVES, get_color_primitives
    except ImportError:
        return pieces
    
    inp0 = train_pairs[0][0]
    h0, w0 = grid_shape(inp0)
    
    # Only add primitives that change the grid AND preserve output size
    # (for beam search, size changes are only useful at the last step)
    out0 = train_pairs[0][1]
    oh, ow = grid_shape(out0)
    same_size = (h0, w0) == (oh, ow)
    
    for name, fn in PARAMETERLESS_PRIMITIVES:
        try:
            result = fn(inp0)
            if result is None:
                continue
            rh, rw = grid_shape(result)
            # For same-size tasks, only keep same-size transforms
            if same_size and (rh, rw) != (h0, w0):
                continue
            if (rh, rw) == (h0, w0) and grid_eq(result, inp0):
                continue
            pieces.append(CrossPiece(name, lambda inp, _fn=fn: _fn(inp)))
        except Exception:
            pass
    
    # Limited color primitives (only remove and keep, skip swaps to reduce count)
    try:
        color_prims = get_color_primitives(inp0)
        for name, fn in color_prims:
            if not (name.startswith('remove_c') or name.startswith('keep_c')):
                continue  # skip swaps and recolor in beam search
            try:
                result = fn(inp0)
                if result is None:
                    continue
                rh, rw = grid_shape(result)
                if same_size and (rh, rw) != (h0, w0):
                    continue
                if (rh, rw) == (h0, w0) and grid_eq(result, inp0):
                    continue
                pieces.append(CrossPiece(name, lambda inp, _fn=fn: _fn(inp)))
            except Exception:
                pass
    except Exception:
        pass
    
    # Also add residual learners (fast)
    try:
        from arc.residual_learner import ALL_LEARNERS
        for rname, rlearn, rapply in ALL_LEARNERS:
            try:
                rule = rlearn(train_pairs)
                if rule is not None:
                    _r = rule
                    _ra = rapply
                    pieces.insert(0, CrossPiece(
                        f'res:{rname}',
                        lambda inp, r=_r, fn=_ra: fn(inp, r)
                    ))
            except Exception:
                pass
    except Exception:
        pass
    
    return pieces


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
    
    # === Phase 1.5: Standalone primitives (fast O(N), always run before early return) ===
    # Ensures draw_diag, flip_h, etc. are tried even if Phase 1 returns early
    try:
        from arc.primitives import PARAMETERLESS_PRIMITIVES, get_color_primitives
        _inp0, _out0 = train_pairs[0]
        _all_prims = list(PARAMETERLESS_PRIMITIVES) + get_color_primitives(_inp0)
        for _pname, _pfn in _all_prims:
            try:
                _r0 = _pfn(_inp0)
                if _r0 is None or not grid_eq(_r0, _out0):
                    continue
                _ok = True
                for _inp, _exp in train_pairs[1:]:
                    _r = _pfn(_inp)
                    if _r is None or not grid_eq(_r, _exp):
                        _ok = False; break
                if _ok:
                    _existing = {getattr(p, 'name', '') for _, p in verified}
                    if _pname not in _existing:
                        verified.insert(0, ('cross', CrossPiece(_pname, lambda inp, fn=_pfn: fn(inp))))
            except Exception:
                pass
    except Exception:
        pass
    
    # === Phase 7 (early): Puzzle Reasoning Language ===
    # Run early because puzzle_lang programs are high-confidence declarative rules
    try:
        from arc.puzzle_lang import synthesize_programs as _synth_puzzle
        _puzzle_progs = _synth_puzzle(train_pairs)
        for _pp in _puzzle_progs:
            try:
                _pvalid = True
                for _pinp, _pout in train_pairs:
                    _pr = _pp.apply_fn(_pinp)
                    if _pr is None or not grid_eq(_pr, _pout):
                        _pvalid = False; break
                if _pvalid:
                    _ptr = _pp.apply_fn(test_inputs[0])
                    if _ptr is not None:
                        verified.insert(0, ('cross',
                            CrossPiece(f'puzzle:{_pp.name}', _pp.apply_fn)))
            except Exception:
                continue
    except Exception:
        pass
    
    # === Phase 1.55: CrossUniverse flow propagation ===
    try:
        from arc.cross_universe import generate_cross_universe_pieces
        _cu_pieces = generate_cross_universe_pieces(train_pairs)
        for _cup in _cu_pieces:
            if sim.verify(_cup, train_pairs):
                verified.append(('cross', _cup))
                if len(verified) >= 2:
                    break
    except Exception:
        pass

    # === Phase 1.56: CrossUniverse3D — 立体cross構造 ===
    try:
        from arc.cross_universe_3d import generate_3d_cross_pieces
        _cu3d_pieces = generate_3d_cross_pieces(train_pairs)
        for _cup3d in _cu3d_pieces:
            if sim.verify(_cup3d, train_pairs):
                verified.append(('cross', _cup3d))
                if len(verified) >= 2:
                    break
    except Exception:
        pass

    # === Phase 1.57: MultiScale Cross Structure ===
    try:
        from arc.cross_multiscale import generate_multiscale_cross_pieces
        _ms_pieces = generate_multiscale_cross_pieces(train_pairs)
        for _msp in _ms_pieces:
            if sim.verify(_msp, train_pairs):
                verified.append(('cross', _msp))
                if len(verified) >= 2:
                    break
    except Exception:
        pass

    all_pieces = []

    # === Phase 1.58: Cross3D Geometry ===
    try:
        from arc.cross3d_geometry import generate_cross3d_geometry_pieces
        _c3d_pieces = generate_cross3d_geometry_pieces(train_pairs)
        all_pieces.extend(_c3d_pieces)
    except Exception:
        pass

    # === Phase 1.57b: Crop/Extract Solver ===
    try:
        from arc.crop_extract_solver import generate_crop_extract_pieces
        _ce_pieces = generate_crop_extract_pieces(train_pairs)
        all_pieces.extend(_ce_pieces)
    except Exception:
        pass

    # === Phase 1.57c: Smart Crop Solver ===
    try:
        from arc.smart_crop_solver import generate_smart_crop_pieces
        _sc_pieces = generate_smart_crop_pieces(train_pairs)
        all_pieces.extend(_sc_pieces)
    except Exception:
        pass

    # === Phase 1.57i: Abstract Neighborhood Solver ===
    # DISABLED: causes regression on 67a423a3 (adds abs_nb piece that confuses NB rule selection)
    # try:
    #     from arc.abstract_nb_solver import generate_abstract_nb_pieces
    #     _anb_pieces = generate_abstract_nb_pieces(train_pairs)
    #     all_pieces.extend(_anb_pieces)
    # except Exception:
    #     pass

    # === Phase 1.57h: Program Search (test-time synthesis) ===
    try:
        from arc.program_search import generate_search_pieces
        _search_pieces = generate_search_pieces(train_pairs)
        all_pieces.extend(_search_pieces)
    except Exception:
        pass

    # === Phase 1.57g: Brute-force Rule Solver ===
    try:
        from arc.brute_rule_solver import generate_brute_pieces
        _brute_pieces = generate_brute_pieces(train_pairs)
        all_pieces.extend(_brute_pieces)
    except Exception:
        pass

    # === Phase 1.57f: Panel Solver ===
    try:
        from arc.panel_solver import generate_panel_pieces
        _panel_pieces = generate_panel_pieces(train_pairs)
        all_pieces.extend(_panel_pieces)
    except Exception:
        pass

    # === Phase 1.57e: Scale/Upscale Solver ===
    try:
        from arc.scale_solver import generate_scale_pieces
        _scale_pieces = generate_scale_pieces(train_pairs)
        all_pieces.extend(_scale_pieces)
    except Exception:
        pass

    # === Phase 1.57d: Color Map Solver ===
    try:
        from arc.color_map_solver import generate_color_map_pieces
        _cm_pieces = generate_color_map_pieces(train_pairs)
        all_pieces.extend(_cm_pieces)
    except Exception:
        pass

    # === Phase 1.58a: Tiny Diff Solver ===
    try:
        from arc.tiny_diff_solver import generate_tiny_diff_pieces
        _td_pieces = generate_tiny_diff_pieces(train_pairs)
        all_pieces.extend(_td_pieces)
    except Exception:
        pass

    # === Phase 1.58b: Probe-based Gravity ===
    try:
        from arc.probe_gravity_solver import generate_probe_gravity_pieces
        _pg_pieces = generate_probe_gravity_pieces(train_pairs)
        all_pieces.extend(_pg_pieces)
    except Exception:
        pass

    # === Phase 1.59: Gravity/Slide Transform ===
    try:
        from arc.gravity_solver import generate_gravity_pieces
        _grav_pieces = generate_gravity_pieces(train_pairs)
        all_pieces.extend(_grav_pieces)
    except Exception:
        pass

    # === Phase 1.60: Flood Fill / Line Extension ===
    try:
        from arc.flood_fill_solver import generate_flood_fill_pieces
        _ff_pieces = generate_flood_fill_pieces(train_pairs)
        all_pieces.extend(_ff_pieces)
    except Exception:
        pass

    # === Phase 1.61: Symmetry Repair ===
    try:
        from arc.symmetry_solver import generate_symmetry_pieces
        _sym_pieces = generate_symmetry_pieces(train_pairs)
        all_pieces.extend(_sym_pieces)
    except Exception:
        pass

    # === Phase 1.59z: Rotating Cross + Conditional Cross (last resort) ===
    try:
        from arc.rotating_cross import generate_rotating_cross_pieces
        _rc_pieces = generate_rotating_cross_pieces(train_pairs)
        all_pieces.extend(_rc_pieces)
    except Exception:
        pass

    # === Phase 1.62a: Cross Probe Fill ===
    try:
        from arc.cross_probe_fill import generate_cross_probe_pieces
        _cpf_pieces = generate_cross_probe_pieces(train_pairs)
        all_pieces.extend(_cpf_pieces)
    except Exception:
        pass

    # === Phase 1.62e: Proximity Recolor Solver ===
    try:
        from arc.proximity_recolor_solver import generate_proximity_recolor_pieces
        _pr_pieces = generate_proximity_recolor_pieces(train_pairs)
        all_pieces.extend(_pr_pieces)
    except Exception:
        pass

    # === Phase 1.62d: Extend to Divider Solver ===
    try:
        from arc.extend_to_divider_solver import generate_extend_to_divider_pieces
        _etd_pieces = generate_extend_to_divider_pieces(train_pairs)
        all_pieces.extend(_etd_pieces)
    except Exception:
        pass

    # === Phase 1.62c: Concentric Fill Solver ===
    try:
        from arc.concentric_fill_solver import generate_concentric_fill_pieces
        _cf_pieces = generate_concentric_fill_pieces(train_pairs)
        all_pieces.extend(_cf_pieces)
    except Exception:
        pass

    # === Phase 1.62b: Panel Extract Solver ===
    try:
        from arc.panel_extract_solver import generate_panel_extract_pieces
        _pe_pieces = generate_panel_extract_pieces(train_pairs)
        all_pieces.extend(_pe_pieces)
    except Exception:
        pass

    # === Phase 1.62a: Periodic Fill Solver ===
    try:
        from arc.periodic_fill_solver import generate_periodic_fill_pieces
        _pf_pieces = generate_periodic_fill_pieces(train_pairs)
        all_pieces.extend(_pf_pieces)
    except Exception:
        pass

    # === Phase 1.61: Role-aware NB Rule (Object IR) ===
    try:
        from arc.role_nb import generate_role_nb_pieces
        _rnb_pieces = generate_role_nb_pieces(train_pairs)
        all_pieces.extend(_rnb_pieces)
    except Exception:
        pass

    # === Phase 1.61b: Topology-based Fill Solver ===
    try:
        from arc.topology_solver import generate_topology_pieces
        _topo_pieces = generate_topology_pieces(train_pairs)
        all_pieces.extend(_topo_pieces)
    except Exception:
        pass

    # === Phase 1.61c: Object Program Synthesis ===
    try:
        from arc.object_program import generate_object_program_pieces
        _op_pieces = generate_object_program_pieces(train_pairs)
        all_pieces.extend(_op_pieces)
    except Exception:
        pass

    # === Phase 1.62e: Fill Enclosed / Connect / Pattern Projection ===
    try:
        from arc.fill_enclosed_solver import generate_fill_enclosed_pieces
        _fe_pieces = generate_fill_enclosed_pieces(train_pairs)
        all_pieces.extend(_fe_pieces)
    except Exception:
        pass

    # === Phase 1.62: Object Movement Engine ===
    try:
        from arc.object_mover import solve_object_movement
        _om_result = solve_object_movement(train_pairs, test_inputs)
        if _om_result is not None:
            def _om_fn(inp, _train=train_pairs, _test=test_inputs):
                r = solve_object_movement(_train, [inp])
                return r[0] if r else None
            all_pieces.append(CrossPiece('object_mover', _om_fn))
    except Exception:
        pass

    # === Phase 1.63: Rectangular Boundary Solver ===
    try:
        from arc.rect_boundary_solver import generate_rect_boundary_pieces
        _rb_pieces = generate_rect_boundary_pieces(train_pairs)
        all_pieces.extend(_rb_pieces)
    except Exception:
        pass

    # === Phase 1.64: Pattern Tiling Solver ===
    try:
        from arc.pattern_tiling_solver import generate_pattern_tiling_pieces
        _pt_pieces = generate_pattern_tiling_pieces(train_pairs)
        all_pieces.extend(_pt_pieces)
    except Exception:
        pass

    # === Phase 1.65: Parallel Cross Layer Engine (kofdai 6構想) ===
    try:
        from arc.cross_parallel_engine import generate_parallel_cross_pieces
        _pcl_pieces = generate_parallel_cross_pieces(train_pairs)
        all_pieces.extend(_pcl_pieces)
    except Exception:
        pass

    # === Phase 1.5x: Verify all_pieces from probe/gravity/flood/symmetry solvers ===
    for _ap in all_pieces:
        try:
            if CrossSimulator.verify(_ap, train_pairs):
                _existing = {getattr(p, 'name', '') for _, p in verified}
                if _ap.name not in _existing:
                    verified.append(('cross', _ap))
                    if len(verified) >= 2:
                        break
        except Exception:
            pass

    if len(verified) >= 2:
        return _apply_verified(verified, test_inputs), verified

    # === Phase 1.6: Convergent stamp application ===
    if cross_pieces:
        for cp in cross_pieces:
            if not cp.name.startswith('stamp:'):
                continue
            _conv_ok = True
            for _ci, _co in train_pairs:
                _x = _ci
                for _ in range(20):
                    try:
                        _y = cp.apply(_x)
                    except Exception:
                        _y = None
                    if _y is None:
                        break
                    if grid_eq(_y, _x):
                        break
                    _x = _y
                if not grid_eq(_x, _co):
                    _conv_ok = False
                    break
            if _conv_ok:
                _cp_ref = cp
                def _conv_apply(inp, _piece=_cp_ref):
                    x = inp
                    for _ in range(20):
                        try:
                            y = _piece.apply(x)
                        except Exception:
                            break
                        if y is None or grid_eq(y, x):
                            break
                        x = y
                    return x
                verified.append(('cross',
                    CrossPiece(f'converge:{_cp_ref.name}', _conv_apply)))
                if len(verified) >= 2:
                    break

    if len(verified) >= 2:
        return _apply_verified(verified, test_inputs), verified

    # === Phase 2.5: Block-level IR (multi-scale reasoning) ===
    try:
        from arc.block_ir import solve_at_block_level
        _block_preds, _block_verified = solve_at_block_level(train_pairs, test_inputs)
        if _block_preds is not None and _block_verified:
            _bname = f'block_ir:{_block_verified[0][1].name}'
            # If we have high-confidence cross3d pieces, don't let block_ir override
            _has_cross3d = any(getattr(p, 'name', '').startswith('cross3d:') for _, p in verified)
            if _has_cross3d:
                # Add as fallback candidate instead of returning immediately
                verified.append(('cross', CrossPiece(_bname, lambda inp, _bp=_block_preds: _bp[0][0] if _bp and _bp[0] else None)))
            else:
                return _block_preds, [('cross', CrossPiece(_bname, lambda inp: None))]
    except Exception:
        pass

    # === Phase 8: ProgramTree (CEGIS条件分岐/ループ合成) ===
    try:
        from arc.program_tree import ProgramTreeSynthesizer, ApplyNode
        _pt_synth = ProgramTreeSynthesizer(
            pieces=cross_pieces, train_pairs=train_pairs, timeout=2.0)
        _pt_result = _pt_synth.synthesize()
        if _pt_result is not None:
            verified.insert(0, ('cross',
                CrossPiece(f'ptree:{_pt_result.describe()[:60]}',
                           _pt_result.apply)))
    except Exception:
        pass

    if len(verified) >= 2:
        return _apply_verified(verified, test_inputs), verified

    # === Phase 9: ARC-CEGIS (transform chain synthesis) ===
    try:
        from arc.arc_cegis import ARCCEGISLoop
        import numpy as _np
        _cegis = ARCCEGISLoop(max_chain_len=2, time_limit_ms=3000)
        _cegis_result = _cegis.solve(
            [{'input': i, 'output': o} for i, o in train_pairs],
            test_inputs[0])
        if _cegis_result.solved and _cegis_result.transform_chain:
            _chain = _cegis_result.transform_chain
            def _apply_chain(inp, chain=_chain):
                x = _np.array(inp, dtype=_np.int8)
                for t in chain:
                    x = t.fn(x)
                return x.tolist()
            # Verify on all train pairs
            _cegis_ok = True
            for _ci, _co in train_pairs:
                _cr = _apply_chain(_ci)
                if not grid_eq(_cr, _co):
                    _cegis_ok = False; break
            if _cegis_ok:
                _desc = '+'.join(t.name for t in _chain)
                verified.insert(0, ('cross',
                    CrossPiece(f'cegis:{_desc}', _apply_chain)))
    except Exception:
        pass

    if len(verified) >= 2:
        return _apply_verified(verified, test_inputs), verified

    # === Phase 3: Composition of cross pieces with WG programs ===
    wg_cands = None
    midpoints = {}
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
    
    # === Phase 3b: cross_piece → cross_piece composition ===
    if len(verified) < 2 and cross_pieces:
        # Pre-compute intermediate results for first train pair to prune
        inp0, out0 = train_pairs[0]
        midpoints = {}  # cp_idx -> intermediate grid
        for i, cp in enumerate(cross_pieces):
            mid = cp.apply(inp0)
            if mid is not None and not grid_eq(mid, inp0):  # skip identity
                midpoints[i] = mid
        
        for i, mid0 in midpoints.items():
            if len(verified) >= 2:
                break
            for j, cp2 in enumerate(cross_pieces):
                if i == j:
                    continue
                res0 = cp2.apply(mid0)
                if res0 is None or not grid_eq(res0, out0):
                    continue
                # Full verify
                ok = True
                for inp, exp in train_pairs[1:]:
                    mid = cross_pieces[i].apply(inp)
                    if mid is None:
                        ok = False; break
                    res = cp2.apply(mid)
                    if res is None or not grid_eq(res, exp):
                        ok = False; break
                if ok:
                    verified.append(('cross_compose', (cross_pieces[i], cp2)))
                    if len(verified) >= 2:
                        break
    
    # === Phase 3c: 3-step composition ===
    if len(verified) < 2 and cross_pieces:
        inp0, out0 = train_pairs[0]
        # Build midpoints if not already done
        if not midpoints:
            midpoints = {}
            for i, cp in enumerate(cross_pieces):
                mid = cp.apply(inp0)
                if mid is not None and not grid_eq(mid, inp0):
                    midpoints[i] = mid
        
        # Limit search: only try top-scoring mid1 candidates (partial match > 0.3)
        scored_mid1 = []
        for i, mid1 in midpoints.items():
            score = CrossSimulator.partial_verify(cross_pieces[i], train_pairs)
            if score > 0.3:
                scored_mid1.append((score, i, mid1))
        scored_mid1.sort(reverse=True)
        scored_mid1 = scored_mid1[:10]  # top 10
        
        for _, i, mid1 in scored_mid1:
            if len(verified) >= 2:
                break
            # mid1 → cross_piece → cross_piece/WG
            mid2_map = {}
            for j, cp2 in enumerate(cross_pieces):
                if j == i:
                    continue
                mid2 = cp2.apply(mid1)
                if mid2 is not None and not grid_eq(mid2, mid1):
                    mid2_map[j] = mid2
            
            # 3-step: cp[i] → cp[j] → cp[k]
            for j, mid2 in mid2_map.items():
                if len(verified) >= 2:
                    break
                for k, cp3 in enumerate(cross_pieces):
                    if k == i or k == j:
                        continue
                    res0 = cp3.apply(mid2)
                    if res0 is None or not grid_eq(res0, out0):
                        continue
                    ok = True
                    for inp, exp in train_pairs[1:]:
                        m1 = cross_pieces[i].apply(inp)
                        if m1 is None: ok = False; break
                        m2 = cross_pieces[j].apply(m1)
                        if m2 is None: ok = False; break
                        r = cp3.apply(m2)
                        if r is None or not grid_eq(r, exp): ok = False; break
                    if ok:
                        verified.append(('cross_compose_3', (cross_pieces[i], cross_pieces[j], cp3)))
                        if len(verified) >= 2:
                            break
            
            # 3-step: cp[i] → cp[j] → WG
            if len(verified) < 2:
                wg_cands_local = wg_cands if wg_cands is not None else _generate_whole_grid_candidates(train_pairs)
                for j, mid2 in mid2_map.items():
                    if len(verified) >= 2:
                        break
                    for wg in wg_cands_local:
                        res0 = wg.apply(mid2)
                        if res0 is None or not grid_eq(res0, out0):
                            continue
                        ok = True
                        for inp, exp in train_pairs[1:]:
                            m1 = cross_pieces[i].apply(inp)
                            if m1 is None: ok = False; break
                            m2 = cross_pieces[j].apply(m1)
                            if m2 is None: ok = False; break
                            r = wg.apply(m2)
                            if r is None or not grid_eq(r, exp): ok = False; break
                        if ok:
                            verified.append(('cross_compose_3', (cross_pieces[i], cross_pieces[j], wg)))
                            if len(verified) >= 2:
                                break
    
    # === Phase 3d: Convergent application (apply piece until stable) ===
    if len(verified) < 2 and cross_pieces:
        for cp in cross_pieces:
            # Only try stamp pieces for convergent application (NB overfits)
            if not cp.name.startswith('stamp:'):
                continue
            _conv_ok = True
            for _ci, _co in train_pairs:
                _x = _ci
                for _ in range(20):
                    try:
                        _y = cp.apply(_x)
                    except Exception:
                        _y = None
                    if _y is None:
                        break
                    if grid_eq(_y, _x):
                        break
                    _x = _y
                if not grid_eq(_x, _co):
                    _conv_ok = False
                    break
            if _conv_ok:
                _cp_ref = cp
                def _conv_apply(inp, _piece=_cp_ref):
                    x = inp
                    for _ in range(20):
                        try:
                            y = _piece.apply(x)
                        except Exception:
                            break
                        if y is None or grid_eq(y, x):
                            break
                        x = y
                    return x
                verified.append(('cross',
                    CrossPiece(f'converge:{_cp_ref.name}', _conv_apply)))
                if len(verified) >= 2:
                    break

    # === Phase 4: Iterative Cross (残差学習) ===
    # Apply best partial-match piece, then re-learn on residual (mid, target) pairs
    # Skip for large grids (too slow)
    _max_cells = max((grid_shape(i)[0] * grid_shape(i)[1]) for i, _ in train_pairs)
    if len(verified) < 2 and cross_pieces and _max_cells <= 400:
        iter_results = _iterative_cross_search(train_pairs, cross_pieces, sim, 
                                               max_rounds=2, time_limit=2.0)
        for ir in iter_results:
            verified.append(ir)
            if len(verified) >= 2:
                break
    
    # Note: standalone primitives are checked in Phase 1.5 (before early return)
    
    # === Phase 5: Multi-Arm Beam Search ===
    # Generalized beam search: N-step chains with top-K beam width
    # Only run if no solution found yet and grid is manageable
    if len(verified) < 2 and _max_cells <= 400:
        try:
            from arc.beam_search import beam_search_with_residual_learners
            beam_results = beam_search_with_residual_learners(
                train_pairs,
                _generate_cross_pieces_fast,
                max_depth=3,
                beam_width=4,
                time_limit=1.5
            )
            for br in beam_results:
                kind, chain = br
                verified.append((kind, chain))
                if len(verified) >= 2:
                    break
        except Exception:
            pass
    
    # === Phase 6: DSL Program Enumeration (exhaustive, non-monotonic) ===
    # Unlike beam search, this explores ALL paths including ones where
    # partial match gets worse before reaching the answer.
    if len(verified) < 2 and _max_cells <= 400:
        try:
            from arc.enumerator import enumerate_solve
            enum_results = enumerate_solve(train_pairs, max_depth=2, time_limit=1.5)
            for ename, efn in enum_results:
                verified.append(('cross', CrossPiece(ename, lambda inp, _fn=efn: _fn(inp))))
                if len(verified) >= 2:
                    break
        except Exception:
            pass
    
    # Phase 7 now runs early (before Phase 3) for priority
    
    return _apply_verified(verified, test_inputs), verified


def _iterative_cross_search(train_pairs: List[Tuple[Grid, Grid]], 
                            initial_pieces: List[CrossPiece],
                            sim: CrossSimulator,
                            max_rounds: int = 3,
                            time_limit: float = 3.0) -> List:
    """Iterative Cross: apply best partial piece, re-generate pieces for residual.
    
    Round 1: Find piece with highest partial match
    Round 2: Generate new pieces from (piece1_output, target) pairs
    Round 3+: Continue until exact match or max_rounds
    
    Returns list of verified (kind, program) tuples.
    """
    import time as _time
    _t0 = _time.time()
    results = []
    
    # Score all initial pieces by partial match
    scored = []
    for i, piece in enumerate(initial_pieces):
        score = sim.partial_verify(piece, train_pairs)
        if 0.3 < score < 1.0:  # Skip exact matches (already handled) and very poor
            scored.append((score, i, piece))
    scored.sort(key=lambda x: -x[0])
    
    # Try top-N best partial matches as round 1 candidates
    for _, _, round1_piece in scored[:10]:
        if _time.time() - _t0 > time_limit:
            break
        
        # Compute intermediate results
        mid_pairs = []
        ok = True
        for inp, out in train_pairs:
            mid = round1_piece.apply(inp)
            if mid is None:
                ok = False
                break
            mid_pairs.append((mid, out))
        if not ok:
            continue
        
        # Round 2: Try residual learners FIRST (fast), then cross pieces (slow)
        round2_pieces = []
        
        # Fast: residual learners
        try:
            from arc.residual_learner import ALL_LEARNERS as RES_LEARNERS
            for rname, rlearn, rapply in RES_LEARNERS:
                try:
                    rrule = rlearn(mid_pairs)
                    if rrule is not None:
                        _rr = rrule
                        _ra = rapply
                        round2_pieces.insert(0, CrossPiece(
                            f'res:{rname}',
                            lambda inp, r=_rr, fn=_ra: fn(inp, r)
                        ))
                except Exception:
                    pass
        except Exception:
            pass
        
        # Slower: cross structure re-generation
        round2_pieces.extend(_generate_cross_pieces(mid_pairs))
        
        # Check if any round2 piece completes the job
        for r2_piece in round2_pieces:
            if sim.verify(r2_piece, mid_pairs):
                # Full verify: round1 + round2 on original train
                full_ok = True
                for inp, out in train_pairs:
                    mid = round1_piece.apply(inp)
                    if mid is None:
                        full_ok = False; break
                    final = r2_piece.apply(mid)
                    if final is None or not grid_eq(final, out):
                        full_ok = False; break
                if full_ok:
                    results.append(('iterative_cross_2', (round1_piece, r2_piece)))
                    return results
        
        # Round 3: Try one more level if round 2 partial matches exist
        if max_rounds >= 3:
            r2_scored = []
            for ri, r2_piece in enumerate(round2_pieces):
                s = sim.partial_verify(r2_piece, mid_pairs)
                if 0.3 < s < 1.0:
                    r2_scored.append((s, ri, r2_piece))
            r2_scored.sort(key=lambda x: -x[0])
            
            for _, _, r2_piece in r2_scored[:5]:
                mid2_pairs = []
                ok2 = True
                for mid, out in mid_pairs:
                    mid2 = r2_piece.apply(mid)
                    if mid2 is None:
                        ok2 = False; break
                    mid2_pairs.append((mid2, out))
                if not ok2:
                    continue
                
                round3_pieces = _generate_cross_pieces(mid2_pairs)
                for r3_piece in round3_pieces:
                    if sim.verify(r3_piece, mid2_pairs):
                        # Full verify
                        full_ok = True
                        for inp, out in train_pairs:
                            m1 = round1_piece.apply(inp)
                            if m1 is None: full_ok = False; break
                            m2 = r2_piece.apply(m1)
                            if m2 is None: full_ok = False; break
                            final = r3_piece.apply(m2)
                            if final is None or not grid_eq(final, out): full_ok = False; break
                        if full_ok:
                            results.append(('iterative_cross_3', (round1_piece, r2_piece, r3_piece)))
                            return results
    
    return results


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
    # High false-positive modules: treat as low-priority fallback
    fallback_prefixes = ('stamp:', 'obj:', 'transform:')
    
    nb_verified = []
    non_nb_verified = []
    fallback_verified = []
    for item in verified:
        kind, prog = item
        name = getattr(prog, 'name', '')
        if any(name.startswith(p) for p in fallback_prefixes):
            fallback_verified.append(item)
        elif name in nb_names or name.startswith('cross_nb') or name.startswith('structural_nb') or name.startswith('abstract_nb'):
            nb_verified.append(item)
        else:
            non_nb_verified.append(item)
    
    # Puzzle-lang programs are highest confidence — always first
    # cross3d programs are high confidence (structural, not overfit)
    puzzle_verified = [item for item in non_nb_verified if getattr(item[1], 'name', '').startswith('puzzle:')]
    cross3d_verified = [item for item in non_nb_verified if getattr(item[1], 'name', '').startswith('cross3d:')]
    other_non_nb = [item for item in non_nb_verified 
                    if not getattr(item[1], 'name', '').startswith('puzzle:')
                    and not getattr(item[1], 'name', '').startswith('cross3d:')]
    
    # Build final candidate list: puzzle first, then cross3d, then other non-NB, then NB, then fallback
    selected = puzzle_verified[:2] + cross3d_verified[:2] + other_non_nb[:2] + nb_verified[:2] + fallback_verified[:1]
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
            elif kind == 'cross_compose_3':
                p1, p2, p3 = prog
                m1 = p1.apply(test_inp) if hasattr(p1, 'apply') else None
                m2 = p2.apply(m1) if m1 is not None else None
                result = p3.apply(m2) if m2 is not None else None
            elif kind == 'iterative_cross_2':
                p1, p2 = prog
                m1 = p1.apply(test_inp)
                result = p2.apply(m1) if m1 is not None else None
            elif kind == 'iterative_cross_3':
                p1, p2, p3 = prog
                m1 = p1.apply(test_inp)
                m2 = p2.apply(m1) if m1 is not None else None
                result = p3.apply(m2) if m2 is not None else None
            elif kind.startswith('beam_depth_'):
                # Multi-arm beam search: chain of N pieces
                chain = prog
                x = test_inp
                for piece in chain:
                    x = piece.apply(x)
                    if x is None:
                        break
                result = x
            else:
                result = None
            
            if result is not None:
                # Deduplicate identical predictions
                if not any(grid_eq(result, a) for a in attempts):
                    attempts.append(result)
        predictions.append(attempts)
    
    return predictions
