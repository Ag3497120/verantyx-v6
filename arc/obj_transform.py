"""
arc/obj_transform.py — Cross Simulator Position/Placement Transform

Multi-layer Cross decomposition for object position transforms:
  Layer 1: Object detection + correspondence (input↔output matching)
  Layer 2: Transform classification (move/copy/reflect/align/scale)
  Layer 3: Transform parameter learning (direction, distance, axis)
  Cross: Selector × Transform × Parameter → Program

Handles:
- Object movement (translate by fixed offset or to target position)
- Object copying (stamp at multiple positions)
- Reflection (horizontal/vertical/diagonal flip of object arrangement)
- Alignment (sort objects by property, align to grid)
- Gravity (drop objects in a direction)
"""

from typing import List, Tuple, Optional, Dict, Set
from collections import Counter, defaultdict
from arc.grid import Grid, grid_shape, grid_eq, most_common_color
from arc.objects import detect_objects, find_matching_objects, ArcObject
import copy


def learn_obj_transform(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn an object position/placement transform."""
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    bg = most_common_color(train_pairs[0][0])
    
    for strategy in [
        _learn_uniform_translate,
        _learn_gravity_drop,
        _learn_copy_to_all,
        _learn_reflect_arrangement,
        _learn_sort_objects,
        _learn_move_to_contact,
    ]:
        rule = strategy(train_pairs, bg)
        if rule is not None:
            # Full verification
            ok = True
            for inp, out in train_pairs:
                result = apply_obj_transform(inp, rule)
                if result is None or not grid_eq(result, out):
                    ok = False
                    break
            if ok:
                return rule
    
    return None


def _learn_uniform_translate(train_pairs, bg):
    """All objects move by the same (dr, dc) offset."""
    
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        if len(in_objs) != len(out_objs):
            return None
    
    offsets = []
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        matches = find_matching_objects(in_objs, out_objs)
        
        if len(matches) != len(in_objs):
            return None
        
        pair_offsets = set()
        for oi, oo in matches:
            if oi.shape != oo.shape:
                return None
            dr = int(round(oo.center[0] - oi.center[0]))
            dc = int(round(oo.center[1] - oi.center[1]))
            pair_offsets.add((dr, dc))
        
        if len(pair_offsets) != 1:
            return None
        offsets.append(pair_offsets.pop())
    
    # All pairs must have same offset
    if len(set(offsets)) != 1:
        return None
    
    dr, dc = offsets[0]
    if dr == 0 and dc == 0:
        return None
    
    return {
        'type': 'uniform_translate',
        'dr': dr,
        'dc': dc,
        'name': f'translate_dr{dr}_dc{dc}',
    }


def _learn_gravity_drop(train_pairs, bg):
    """Objects drop in a direction until hitting another object or border."""
    
    for direction in ['down', 'up', 'left', 'right']:
        consistent = True
        for inp, out in train_pairs:
            result = _apply_gravity(inp, bg, direction)
            if not grid_eq(result, out):
                consistent = False
                break
        
        if consistent:
            return {
                'type': 'gravity',
                'direction': direction,
                'name': f'gravity_{direction}',
            }
    
    return None


def _apply_gravity(inp, bg, direction):
    """Apply gravity in a direction."""
    h, w = grid_shape(inp)
    result = [[bg] * w for _ in range(h)]
    
    if direction == 'down':
        for c in range(w):
            write_pos = h - 1
            for r in range(h - 1, -1, -1):
                if inp[r][c] != bg:
                    result[write_pos][c] = inp[r][c]
                    write_pos -= 1
    elif direction == 'up':
        for c in range(w):
            write_pos = 0
            for r in range(h):
                if inp[r][c] != bg:
                    result[write_pos][c] = inp[r][c]
                    write_pos += 1
    elif direction == 'right':
        for r in range(h):
            write_pos = w - 1
            for c in range(w - 1, -1, -1):
                if inp[r][c] != bg:
                    result[r][write_pos] = inp[r][c]
                    write_pos -= 1
    elif direction == 'left':
        for r in range(h):
            write_pos = 0
            for c in range(w):
                if inp[r][c] != bg:
                    result[r][write_pos] = inp[r][c]
                    write_pos += 1
    
    return result


def _learn_copy_to_all(train_pairs, bg):
    """One unique object is copied to positions of all same-colored markers."""
    
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        if len(in_objs) < 2:
            return None
    
    # Find: one object has unique shape, others are single-cell markers
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        
        # Find single-cell objects (markers) and multi-cell objects (templates)
        markers = [o for o in in_objs if o.size == 1]
        templates = [o for o in in_objs if o.size > 1]
        
        if not markers or not templates:
            return None
    
    # Try: template is stamped at each marker position
    consistent = True
    template_selector = None  # 'largest', 'unique_shape', etc.
    
    for selector in ['largest', 'unique_color']:
        consistent = True
        for inp, out in train_pairs:
            in_objs = detect_objects(inp, bg)
            
            if selector == 'largest':
                max_size = max(o.size for o in in_objs)
                templates = [o for o in in_objs if o.size == max_size]
                markers = [o for o in in_objs if o.size < max_size]
            elif selector == 'unique_color':
                color_counts = Counter(o.color for o in in_objs)
                templates = [o for o in in_objs if color_counts[o.color] == 1]
                markers = [o for o in in_objs if color_counts[o.color] > 1]
            
            if len(templates) != 1 or not markers:
                consistent = False
                break
            
            tmpl = templates[0]
            # Build expected output: copy template to each marker position
            h, w = grid_shape(inp)
            result = [row[:] for row in inp]
            
            # Remove markers from result
            for m in markers:
                for r, c in m.cells:
                    result[r][c] = bg
            
            # Stamp template at marker positions
            tr, tc = int(round(tmpl.center[0])), int(round(tmpl.center[1]))
            for m in markers:
                mr, mc = int(round(m.center[0])), int(round(m.center[1]))
                for or_, oc in tmpl.cells:
                    nr = or_ - tr + mr
                    nc = oc - tc + mc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr][nc] = tmpl.color
            
            if not grid_eq(result, out):
                consistent = False
                break
        
        if consistent:
            return {
                'type': 'copy_template_to_markers',
                'selector': selector,
                'name': f'copy_{selector}_to_markers',
            }
    
    return None


def _learn_reflect_arrangement(train_pairs, bg):
    """Reflect entire grid content along an axis."""
    
    for axis in ['horizontal', 'vertical', 'both']:
        consistent = True
        for inp, out in train_pairs:
            h, w = grid_shape(inp)
            result = [row[:] for row in inp]
            
            if axis in ('horizontal', 'both'):
                flipped = [[result[h-1-r][c] for c in range(w)] for r in range(h)]
                # Merge: non-bg from both
                for r in range(h):
                    for c in range(w):
                        if result[r][c] == bg and flipped[r][c] != bg:
                            result[r][c] = flipped[r][c]
                        elif flipped[r][c] != bg and result[r][c] != bg:
                            pass  # keep original
            
            if axis in ('vertical', 'both'):
                flipped = [[result[r][w-1-c] for c in range(w)] for r in range(h)]
                for r in range(h):
                    for c in range(w):
                        if result[r][c] == bg and flipped[r][c] != bg:
                            result[r][c] = flipped[r][c]
            
            if not grid_eq(result, out):
                consistent = False
                break
        
        if consistent:
            return {
                'type': 'reflect_merge',
                'axis': axis,
                'name': f'reflect_merge_{axis}',
            }
    
    return None


def _learn_sort_objects(train_pairs, bg):
    """Sort objects by size/color and rearrange positions."""
    
    # Check: objects in output are same objects but in different positions
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        out_objs = detect_objects(out, bg)
        if len(in_objs) != len(out_objs):
            return None
        # Each input shape must appear in output
        in_shapes = Counter(o.shape for o in in_objs)
        out_shapes = Counter(o.shape for o in out_objs)
        if in_shapes != out_shapes:
            return None
    
    # Try: sort by size → position ordering
    for sort_key in ['size_asc', 'size_desc', 'color_asc']:
        consistent = True
        for inp, out in train_pairs:
            in_objs = detect_objects(inp, bg)
            out_objs = detect_objects(out, bg)
            
            # Get position ordering from output
            if sort_key == 'size_asc':
                sorted_in = sorted(in_objs, key=lambda o: o.size)
            elif sort_key == 'size_desc':
                sorted_in = sorted(in_objs, key=lambda o: -o.size)
            elif sort_key == 'color_asc':
                sorted_in = sorted(in_objs, key=lambda o: o.color)
            
            # Output positions (sorted by position)
            out_sorted = sorted(out_objs, key=lambda o: (o.bbox[0], o.bbox[1]))
            in_pos_sorted = sorted(in_objs, key=lambda o: (o.bbox[0], o.bbox[1]))
            
            if len(sorted_in) != len(out_sorted):
                consistent = False; break
            
            # Check: sorted_in[i].shape == out_sorted[i].shape?
            for si, oo in zip(sorted_in, out_sorted):
                if si.shape != oo.shape or si.color != oo.color:
                    consistent = False; break
            if not consistent: break
        
        if consistent:
            return {
                'type': 'sort_objects',
                'sort_key': sort_key,
                'name': f'sort_{sort_key}',
            }
    
    return None


def _learn_move_to_contact(train_pairs, bg):
    """Move objects until they touch each other or a specific target."""
    
    # Simple case: 2 objects, one moves toward the other
    for inp, out in train_pairs:
        in_objs = detect_objects(inp, bg)
        if len(in_objs) != 2:
            return None
    
    # Which object moves? Try both
    for mover_idx in [0, 1]:
        target_idx = 1 - mover_idx
        consistent = True
        move_dirs = []
        
        for inp, out in train_pairs:
            in_objs = detect_objects(inp, bg)
            out_objs = detect_objects(out, bg)
            matches = find_matching_objects(in_objs, out_objs)
            
            if len(matches) != 2:
                consistent = False; break
            
            # Check: target doesn't move
            target_in = in_objs[target_idx]
            moved = None
            stationary = None
            for oi, oo in matches:
                if oi.bbox == oo.bbox:
                    stationary = oi
                else:
                    moved = (oi, oo)
            
            if moved is None or stationary is None:
                consistent = False; break
            
            oi, oo = moved
            # Direction: toward stationary
            dr = int(round(stationary.center[0] - oi.center[0]))
            dc = int(round(stationary.center[1] - oi.center[1]))
            
            if dr != 0:
                dr = dr // abs(dr)
            if dc != 0:
                dc = dc // abs(dc)
            
            move_dirs.append((dr, dc))
        
        if not consistent:
            continue
        
        # All pairs same direction?
        if len(set(move_dirs)) != 1:
            continue
        
        dr, dc = move_dirs[0]
        
        # Verify: move until adjacent
        ok = True
        for inp, out in train_pairs:
            result = _apply_move_to_contact(inp, bg, mover_idx, dr, dc)
            if not grid_eq(result, out):
                ok = False; break
        
        if ok:
            return {
                'type': 'move_to_contact',
                'mover': 'smaller' if mover_idx == 0 else 'larger',
                'dr': dr,
                'dc': dc,
                'name': f'move_to_contact_{dr}_{dc}',
            }
    
    return None


def _apply_move_to_contact(inp, bg, mover_idx, dr, dc):
    """Move one object toward another until they're adjacent."""
    h, w = grid_shape(inp)
    objs = detect_objects(inp, bg)
    if len(objs) != 2:
        return inp
    
    mover = objs[mover_idx]
    target = objs[1 - mover_idx]
    target_cells = set(target.cells)
    
    result = [row[:] for row in inp]
    
    # Clear mover from result
    for r, c in mover.cells:
        result[r][c] = bg
    
    # Move mover step by step until adjacent to target or at border
    current_cells = list(mover.cells)
    for _ in range(max(h, w)):
        next_cells = [(r + dr, c + dc) for r, c in current_cells]
        
        # Check if any next cell overlaps target or goes out of bounds
        if any(not (0 <= r < h and 0 <= c < w) for r, c in next_cells):
            break
        if any((r, c) in target_cells for r, c in next_cells):
            break
        
        current_cells = next_cells
    
    # Place mover at final position
    for (r, c), (_, _) in zip(current_cells, mover.cells):
        if 0 <= r < h and 0 <= c < w:
            result[r][c] = mover.color
    
    return result


def apply_obj_transform(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply an object position/placement transform."""
    bg = most_common_color(inp)
    h, w = grid_shape(inp)
    rtype = rule['type']
    
    if rtype == 'uniform_translate':
        dr, dc = rule['dr'], rule['dc']
        result = [[bg] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != bg:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr][nc] = inp[r][c]
        return result
    
    elif rtype == 'gravity':
        return _apply_gravity(inp, bg, rule['direction'])
    
    elif rtype == 'copy_template_to_markers':
        objs = detect_objects(inp, bg)
        selector = rule['selector']
        
        if selector == 'largest':
            max_size = max(o.size for o in objs)
            templates = [o for o in objs if o.size == max_size]
            markers = [o for o in objs if o.size < max_size]
        elif selector == 'unique_color':
            color_counts = Counter(o.color for o in objs)
            templates = [o for o in objs if color_counts[o.color] == 1]
            markers = [o for o in objs if color_counts[o.color] > 1]
        else:
            return None
        
        if len(templates) != 1:
            return None
        
        tmpl = templates[0]
        tr, tc = int(round(tmpl.center[0])), int(round(tmpl.center[1]))
        result = [row[:] for row in inp]
        
        for m in markers:
            for r, c in m.cells:
                result[r][c] = bg
        
        for m in markers:
            mr, mc = int(round(m.center[0])), int(round(m.center[1]))
            for or_, oc in tmpl.cells:
                nr = or_ - tr + mr
                nc = oc - tc + mc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr][nc] = tmpl.color
        
        return result
    
    elif rtype == 'reflect_merge':
        axis = rule['axis']
        result = [row[:] for row in inp]
        
        if axis in ('horizontal', 'both'):
            for r in range(h):
                for c in range(w):
                    if result[r][c] == bg and inp[h-1-r][c] != bg:
                        result[r][c] = inp[h-1-r][c]
        
        if axis in ('vertical', 'both'):
            snapshot = [row[:] for row in result]
            for r in range(h):
                for c in range(w):
                    if result[r][c] == bg and snapshot[r][w-1-c] != bg:
                        result[r][c] = snapshot[r][w-1-c]
        
        return result
    
    elif rtype == 'sort_objects':
        objs = detect_objects(inp, bg)
        sort_key = rule['sort_key']
        
        if sort_key == 'size_asc':
            sorted_objs = sorted(objs, key=lambda o: o.size)
        elif sort_key == 'size_desc':
            sorted_objs = sorted(objs, key=lambda o: -o.size)
        elif sort_key == 'color_asc':
            sorted_objs = sorted(objs, key=lambda o: o.color)
        else:
            return None
        
        # Position targets: original positions sorted by position
        pos_sorted = sorted(objs, key=lambda o: (o.bbox[0], o.bbox[1]))
        
        result = [[bg] * w for _ in range(h)]
        for src_obj, tgt_obj in zip(sorted_objs, pos_sorted):
            dr = tgt_obj.bbox[0] - src_obj.bbox[0]
            dc = tgt_obj.bbox[1] - src_obj.bbox[1]
            for r, c in src_obj.cells:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr][nc] = src_obj.color
        
        return result
    
    elif rtype == 'move_to_contact':
        objs = detect_objects(inp, bg)
        if len(objs) != 2:
            return None
        # Determine mover by relative size
        mover_idx = 0 if rule['mover'] == 'smaller' else 1
        if objs[0].size > objs[1].size:
            mover_idx = 1 - mover_idx
        return _apply_move_to_contact(inp, bg, mover_idx, rule['dr'], rule['dc'])
    
    return None
