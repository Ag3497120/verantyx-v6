"""
arc/object_program.py — Object Program Synthesis for ARC

Instead of pixel-level transformations, operates at object level:
1. Detect objects
2. Learn per-object transformation rules based on object attributes
3. Apply rules to test input

Supports:
- Recolor by attribute (size rank, position, shape class)
- Remove objects by predicate
- Move objects based on rules
- Fill object interiors
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import Counter, defaultdict
from arc.grid import Grid, grid_shape, most_common_color, grid_eq
from arc.object_ir import ObjectIR, ArcObject, build_object_ir


def _obj_signature(obj: ArcObject, n_objects: int) -> dict:
    """Get attribute dictionary for an object."""
    return {
        'color': obj.color,
        'area': obj.area,
        'bbox_h': obj.bbox_h,
        'bbox_w': obj.bbox_w,
        'is_rect': obj.is_rectangular,
        'touches_border': obj.touches_border,
        'compactness': round(obj.compactness, 2),
        'rank_by_area': obj.obj_rank_by_area,
        'is_largest': obj.obj_rank_by_area == 0,
        'is_smallest': obj.obj_rank_by_area == n_objects - 1,
        'n_holes': obj.n_holes,
    }


def _match_objects(ir_in: ObjectIR, ir_out: ObjectIR):
    """Match objects between input and output IR.
    
    Returns list of (in_obj, out_obj, change_type) tuples.
    change_type: 'same', 'recolored', 'moved', 'removed', 'added', 'resized'
    """
    matches = []
    used_out = set()
    
    # First pass: exact position match
    for in_obj in ir_in.objects:
        in_cells = set(in_obj.cells)
        best_match = None
        best_overlap = 0
        
        for out_obj in ir_out.objects:
            if out_obj.obj_id in used_out:
                continue
            out_cells = set(out_obj.cells)
            overlap = len(in_cells & out_cells)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = out_obj
        
        if best_match and best_overlap >= min(in_obj.area, best_match.area) * 0.5:
            used_out.add(best_match.obj_id)
            if in_obj.color == best_match.color and set(in_obj.cells) == set(best_match.cells):
                change = 'same'
            elif set(in_obj.cells) == set(best_match.cells):
                change = 'recolored'
            else:
                change = 'modified'
            matches.append((in_obj, best_match, change))
        else:
            matches.append((in_obj, None, 'removed'))
    
    # Check for added objects
    for out_obj in ir_out.objects:
        if out_obj.obj_id not in used_out:
            matches.append((None, out_obj, 'added'))
    
    return matches


def _learn_recolor_rule(train_pairs, bg):
    """Learn: which objects get recolored and to what color?
    
    Tries rules like:
    - Recolor smallest object to color X
    - Recolor largest object to color X
    - Recolor by area rank
    - Recolor objects of specific color
    """
    rules_per_pair = []
    
    for inp_grid, out_grid in train_pairs:
        ir_in = build_object_ir(inp_grid, bg)
        ir_out = build_object_ir(out_grid, bg)
        
        matches = _match_objects(ir_in, ir_out)
        
        recolored = [(m[0], m[1]) for m in matches if m[2] == 'recolored']
        same = [(m[0], m[1]) for m in matches if m[2] == 'same']
        
        if not recolored:
            return None  # no recoloring in this pair
        
        # For each recolored object, record its attributes + new color
        pair_rules = []
        for in_obj, out_obj in recolored:
            sig = _obj_signature(in_obj, len(ir_in.objects))
            pair_rules.append({
                'sig': sig,
                'old_color': in_obj.color,
                'new_color': out_obj.color,
            })
        
        rules_per_pair.append({
            'recolored': pair_rules,
            'n_recolored': len(recolored),
            'n_same': len(same),
            'n_objects': len(ir_in.objects),
        })
    
    if not rules_per_pair:
        return None
    
    # Try to find a consistent rule
    # Rule 1: recolor by area rank
    for target_rank_key in ['is_smallest', 'is_largest']:
        consistent = True
        new_color = None
        for pr in rules_per_pair:
            for rule in pr['recolored']:
                if rule['sig'][target_rank_key]:
                    if new_color is None:
                        new_color = rule['new_color']
                    elif new_color != rule['new_color']:
                        consistent = False
                        break
                else:
                    consistent = False
                    break
            if not consistent:
                break
        
        if consistent and new_color is not None:
            return {'type': 'recolor_by_rank', 'rank_key': target_rank_key, 'new_color': new_color}
    
    # Rule 2: recolor specific color → new color
    color_map = {}
    consistent = True
    for pr in rules_per_pair:
        for rule in pr['recolored']:
            oc = rule['old_color']
            nc = rule['new_color']
            if oc in color_map:
                if color_map[oc] != nc:
                    consistent = False
                    break
            else:
                color_map[oc] = nc
        if not consistent:
            break
    
    if consistent and color_map:
        return {'type': 'recolor_by_color', 'color_map': color_map}
    
    # Rule 3: recolor all non-bg objects to same new color
    new_colors = set()
    for pr in rules_per_pair:
        for rule in pr['recolored']:
            new_colors.add(rule['new_color'])
    
    if len(new_colors) == 1 and all(pr['n_recolored'] == pr['n_objects'] for pr in rules_per_pair):
        return {'type': 'recolor_all', 'new_color': new_colors.pop()}
    
    return None


def _apply_recolor_rule(inp_grid, rule, bg):
    """Apply a recolor rule."""
    inp = np.array(inp_grid) if not isinstance(inp_grid, np.ndarray) else inp_grid
    ir = build_object_ir(inp, bg)
    out = inp.copy()
    
    rtype = rule['type']
    
    if rtype == 'recolor_by_rank':
        for obj in ir.objects:
            sig = _obj_signature(obj, len(ir.objects))
            if sig[rule['rank_key']]:
                for r, c in obj.cells:
                    out[r, c] = rule['new_color']
    
    elif rtype == 'recolor_by_color':
        cmap = rule['color_map']
        for obj in ir.objects:
            if obj.color in cmap:
                for r, c in obj.cells:
                    out[r, c] = cmap[obj.color]
    
    elif rtype == 'recolor_all':
        for obj in ir.objects:
            for r, c in obj.cells:
                out[r, c] = rule['new_color']
    
    return out.tolist()


def _learn_remove_rule(train_pairs, bg):
    """Learn: which objects are removed (set to bg)?"""
    rules_per_pair = []
    
    for inp_grid, out_grid in train_pairs:
        ir_in = build_object_ir(inp_grid, bg)
        inp = np.array(inp_grid)
        out = np.array(out_grid)
        
        if inp.shape != out.shape:
            return None
        
        removed_objs = []
        kept_objs = []
        for obj in ir_in.objects:
            # Check if all cells of this object are bg in output
            all_bg = all(out[r, c] == bg for r, c in obj.cells)
            if all_bg:
                removed_objs.append(obj)
            else:
                kept_objs.append(obj)
        
        if not removed_objs:
            return None
        
        # Also check: no other changes (kept objects are unchanged)
        result = inp.copy()
        for obj in removed_objs:
            for r, c in obj.cells:
                result[r, c] = bg
        
        if not np.array_equal(result, out):
            return None  # there are other changes beyond removal
        
        rules_per_pair.append({
            'removed': [_obj_signature(o, len(ir_in.objects)) for o in removed_objs],
            'kept': [_obj_signature(o, len(ir_in.objects)) for o in kept_objs],
        })
    
    # Try rules
    # Rule: remove smallest
    if all(len(pr['removed']) == 1 and pr['removed'][0]['is_smallest'] for pr in rules_per_pair):
        return {'type': 'remove_smallest'}
    
    # Rule: remove by color
    remove_colors = set()
    for pr in rules_per_pair:
        for sig in pr['removed']:
            remove_colors.add(sig['color'])
    keep_colors = set()
    for pr in rules_per_pair:
        for sig in pr['kept']:
            keep_colors.add(sig['color'])
    
    if remove_colors and not (remove_colors & keep_colors):
        return {'type': 'remove_by_color', 'colors': remove_colors}
    
    # Rule: remove by area threshold
    max_removed_area = max(sig['area'] for pr in rules_per_pair for sig in pr['removed'])
    min_kept_area = min((sig['area'] for pr in rules_per_pair for sig in pr['kept']), default=999)
    
    if max_removed_area < min_kept_area:
        threshold = (max_removed_area + min_kept_area) // 2
        return {'type': 'remove_by_area', 'max_area': threshold}
    
    return None


def _apply_remove_rule(inp_grid, rule, bg):
    """Apply a remove rule."""
    inp = np.array(inp_grid) if not isinstance(inp_grid, np.ndarray) else inp_grid
    ir = build_object_ir(inp, bg)
    out = inp.copy()
    
    rtype = rule['type']
    
    for obj in ir.objects:
        sig = _obj_signature(obj, len(ir.objects))
        remove = False
        
        if rtype == 'remove_smallest' and sig['is_smallest']:
            remove = True
        elif rtype == 'remove_by_color' and obj.color in rule['colors']:
            remove = True
        elif rtype == 'remove_by_area' and obj.area <= rule['max_area']:
            remove = True
        
        if remove:
            for r, c in obj.cells:
                out[r, c] = bg
    
    return out.tolist()


def generate_object_program_pieces(train_pairs, bg=None):
    """Generate CrossPiece candidates from object-level programs."""
    from arc.cross_engine import CrossPiece
    
    if bg is None:
        bg = most_common_color(train_pairs[0][0])
    
    # Check same size
    for inp, out in train_pairs:
        if grid_shape(inp) != grid_shape(out):
            return []
    
    pieces = []
    
    # Try recolor rules
    try:
        rule = _learn_recolor_rule(train_pairs, bg)
        if rule:
            _r = rule
            _bg = bg
            
            def make_recolor_fn(r_, bg_):
                return lambda inp: _apply_recolor_rule(inp, r_, bg_)
            
            fn = make_recolor_fn(_r, _bg)
            
            # Verify
            ok = all(fn(ig) == og for ig, og in train_pairs)
            if ok:
                pieces.append(CrossPiece(
                    name=f"obj_prog:{_r['type']}",
                    apply_fn=fn,
                    version=1
                ))
    except Exception:
        pass
    
    # Try remove rules
    try:
        rule = _learn_remove_rule(train_pairs, bg)
        if rule:
            _r = rule
            _bg = bg
            
            def make_remove_fn(r_, bg_):
                return lambda inp: _apply_remove_rule(inp, r_, bg_)
            
            fn = make_remove_fn(_r, _bg)
            
            ok = all(fn(ig) == og for ig, og in train_pairs)
            if ok:
                pieces.append(CrossPiece(
                    name=f"obj_prog:{_r['type']}",
                    apply_fn=fn,
                    version=1
                ))
    except Exception:
        pass
    
    return pieces
