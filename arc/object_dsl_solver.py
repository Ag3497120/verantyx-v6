"""
arc/object_dsl_solver.py — Object-level DSL solver

Recognizes connected components and applies object-level transformations:
- Recolor by rank (size, position)
- Fill object bounding box
- Stamp/copy objects
- Remove objects by property
- Sort objects
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter
from scipy.ndimage import label as connected_components
from arc.cross_engine import CrossPiece

Grid = List[List[int]]

def _bg(g):
    return int(Counter(g.flatten()).most_common(1)[0][0])

def _get_objects(g, bg):
    """Get list of (mask, color, size, bbox, center) for each connected component"""
    mask = (g != bg).astype(int)
    labeled, n = connected_components(mask)
    objects = []
    for oid in range(1, n + 1):
        obj_mask = labeled == oid
        cells = g[obj_mask]
        color = int(Counter(cells.flatten()).most_common(1)[0][0])
        size = int(np.sum(obj_mask))
        rows, cols = np.where(obj_mask)
        bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
        center = (float(rows.mean()), float(cols.mean()))
        objects.append({
            'mask': obj_mask,
            'color': color,
            'size': size,
            'bbox': bbox,
            'center': center,
            'id': oid,
        })
    return objects

def _try_recolor_by_size_rank(train_pairs, bg):
    """Try recoloring objects by size rank"""
    # For each training pair, get objects and their target colors
    all_mappings = []
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return None
        
        objects = _get_objects(inp, bg)
        if not objects:
            return None
        
        # Sort by size
        sorted_objs = sorted(objects, key=lambda o: o['size'])
        
        mapping = {}
        for rank, obj in enumerate(sorted_objs):
            # What color does this object become in output?
            obj_cells_out = out[obj['mask']]
            out_color = int(Counter(obj_cells_out.flatten()).most_common(1)[0][0])
            mapping[rank] = out_color
        
        all_mappings.append(mapping)
    
    # Check consistency
    if len(all_mappings) < 2:
        return None
    
    ref = all_mappings[0]
    for m in all_mappings[1:]:
        if len(m) != len(ref):
            return None
        for k in ref:
            if k not in m or m[k] != ref[k]:
                return None
    
    return ref

def _try_recolor_by_position(train_pairs, bg, key='row'):
    """Try recoloring objects by position (top-to-bottom or left-to-right)"""
    all_mappings = []
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return None
        
        objects = _get_objects(inp, bg)
        if not objects:
            return None
        
        if key == 'row':
            sorted_objs = sorted(objects, key=lambda o: o['center'][0])
        else:
            sorted_objs = sorted(objects, key=lambda o: o['center'][1])
        
        mapping = {}
        for rank, obj in enumerate(sorted_objs):
            obj_cells_out = out[obj['mask']]
            out_color = int(Counter(obj_cells_out.flatten()).most_common(1)[0][0])
            mapping[rank] = out_color
        
        all_mappings.append(mapping)
    
    if len(all_mappings) < 2:
        return None
    
    ref = all_mappings[0]
    for m in all_mappings[1:]:
        if len(m) != len(ref):
            return None
        for k in ref:
            if k not in m or m[k] != ref[k]:
                return None
    
    return ref

def _try_fill_object_bbox(train_pairs, bg):
    """Check if output = fill each object's bounding box with its color"""
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return False
        
        pred = inp.copy()
        objects = _get_objects(inp, bg)
        for obj in objects:
            r1, c1, r2, c2 = obj['bbox']
            pred[r1:r2+1, c1:c2+1] = obj['color']
        
        if not np.array_equal(pred, out):
            return False
    return True

def _try_remove_smallest(train_pairs, bg):
    """Check if output = remove smallest object(s)"""
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return False
        
        objects = _get_objects(inp, bg)
        if not objects:
            return False
        
        min_size = min(o['size'] for o in objects)
        pred = inp.copy()
        for obj in objects:
            if obj['size'] == min_size:
                pred[obj['mask']] = bg
        
        if not np.array_equal(pred, out):
            return False
    return True

def _try_remove_largest(train_pairs, bg):
    """Check if output = remove largest object(s)"""
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return False
        
        objects = _get_objects(inp, bg)
        if not objects:
            return False
        
        max_size = max(o['size'] for o in objects)
        pred = inp.copy()
        for obj in objects:
            if obj['size'] == max_size:
                pred[obj['mask']] = bg
        
        if not np.array_equal(pred, out):
            return False
    return True

def _try_keep_color_with_most_objects(train_pairs, bg):
    """Keep only objects of the color that has the most objects"""
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return False
        
        objects = _get_objects(inp, bg)
        color_counts = Counter(o['color'] for o in objects)
        if not color_counts:
            return False
        keep_color = color_counts.most_common(1)[0][0]
        
        pred = np.full_like(inp, bg)
        for obj in objects:
            if obj['color'] == keep_color:
                pred[obj['mask']] = inp[obj['mask']]
        
        if not np.array_equal(pred, out):
            return False
    return True

def _try_recolor_by_neighbor_count(train_pairs, bg):
    """Recolor each object based on the number of other objects adjacent to it"""
    # Count adjacent objects for each object
    all_mappings = []
    for inp_raw, out_raw in train_pairs:
        inp = np.array(inp_raw)
        out = np.array(out_raw)
        if inp.shape != out.shape:
            return None
        
        objects = _get_objects(inp, bg)
        if not objects:
            return None
        
        H, W = inp.shape
        mapping = {}
        for obj in objects:
            # Count neighboring objects
            nb_count = 0
            expanded = np.zeros_like(obj['mask'])
            rows, cols = np.where(obj['mask'])
            for r, c in zip(rows, cols):
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        expanded[nr, nc] = True
            
            for other in objects:
                if other['id'] == obj['id']:
                    continue
                if np.any(expanded & other['mask']):
                    nb_count += 1
            
            obj_cells_out = out[obj['mask']]
            out_color = int(Counter(obj_cells_out.flatten()).most_common(1)[0][0])
            
            if nb_count in mapping:
                if mapping[nb_count] != out_color:
                    return None
            else:
                mapping[nb_count] = out_color
        
        all_mappings.append(mapping)
    
    if len(all_mappings) < 2:
        return None
    
    ref = all_mappings[0]
    for m in all_mappings[1:]:
        for k in ref:
            if k in m and m[k] != ref[k]:
                return None
    
    return ref


def generate_object_dsl_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Generate object-level transformation pieces"""
    pieces = []
    
    if not train_pairs:
        return pieces
    
    inp0 = np.array(train_pairs[0][0])
    out0 = np.array(train_pairs[0][1])
    
    if inp0.shape != out0.shape:
        return pieces
    
    bg = _bg(inp0)
    
    # 1. Recolor by size rank
    size_map = _try_recolor_by_size_rank(train_pairs, bg)
    if size_map is not None:
        _sm = dict(size_map)
        _bg_val = bg
        def apply_size_recolor(inp_raw, sm=_sm, bg_val=_bg_val):
            inp = np.array(inp_raw)
            objects = _get_objects(inp, bg_val)
            sorted_objs = sorted(objects, key=lambda o: o['size'])
            out = inp.copy()
            for rank, obj in enumerate(sorted_objs):
                if rank in sm:
                    out[obj['mask']] = sm[rank]
            return out.tolist()
        pieces.append(CrossPiece('obj_recolor_size_rank', apply_size_recolor))
    
    # 2. Recolor by position
    for key in ['row', 'col']:
        pos_map = _try_recolor_by_position(train_pairs, bg, key)
        if pos_map is not None:
            _pm = dict(pos_map)
            _bg_val = bg
            _key = key
            def apply_pos_recolor(inp_raw, pm=_pm, bg_val=_bg_val, k=_key):
                inp = np.array(inp_raw)
                objects = _get_objects(inp, bg_val)
                if k == 'row':
                    sorted_objs = sorted(objects, key=lambda o: o['center'][0])
                else:
                    sorted_objs = sorted(objects, key=lambda o: o['center'][1])
                out = inp.copy()
                for rank, obj in enumerate(sorted_objs):
                    if rank in pm:
                        out[obj['mask']] = pm[rank]
                return out.tolist()
            pieces.append(CrossPiece(f'obj_recolor_pos_{key}', apply_pos_recolor))
    
    # 3. Fill object bounding boxes
    if _try_fill_object_bbox(train_pairs, bg):
        _bg_val = bg
        def apply_fill_bbox(inp_raw, bg_val=_bg_val):
            inp = np.array(inp_raw)
            objects = _get_objects(inp, bg_val)
            out = inp.copy()
            for obj in objects:
                r1, c1, r2, c2 = obj['bbox']
                out[r1:r2+1, c1:c2+1] = obj['color']
            return out.tolist()
        pieces.append(CrossPiece('obj_fill_bbox', apply_fill_bbox))
    
    # 4. Remove smallest objects
    if _try_remove_smallest(train_pairs, bg):
        _bg_val = bg
        def apply_remove_smallest(inp_raw, bg_val=_bg_val):
            inp = np.array(inp_raw)
            objects = _get_objects(inp, bg_val)
            min_size = min(o['size'] for o in objects) if objects else 0
            out = inp.copy()
            for obj in objects:
                if obj['size'] == min_size:
                    out[obj['mask']] = bg_val
            return out.tolist()
        pieces.append(CrossPiece('obj_remove_smallest', apply_remove_smallest))
    
    # 5. Remove largest objects
    if _try_remove_largest(train_pairs, bg):
        _bg_val = bg
        def apply_remove_largest(inp_raw, bg_val=_bg_val):
            inp = np.array(inp_raw)
            objects = _get_objects(inp, bg_val)
            max_size = max(o['size'] for o in objects) if objects else 0
            out = inp.copy()
            for obj in objects:
                if obj['size'] == max_size:
                    out[obj['mask']] = bg_val
            return out.tolist()
        pieces.append(CrossPiece('obj_remove_largest', apply_remove_largest))
    
    # 6. Keep only objects of most common color
    if _try_keep_color_with_most_objects(train_pairs, bg):
        _bg_val = bg
        def apply_keep_most_color(inp_raw, bg_val=_bg_val):
            inp = np.array(inp_raw)
            objects = _get_objects(inp, bg_val)
            color_counts = Counter(o['color'] for o in objects)
            keep_color = color_counts.most_common(1)[0][0] if color_counts else 0
            out = np.full_like(inp, bg_val)
            for obj in objects:
                if obj['color'] == keep_color:
                    out[obj['mask']] = inp[obj['mask']]
            return out.tolist()
        pieces.append(CrossPiece('obj_keep_most_color', apply_keep_most_color))
    
    # 7. Recolor by neighbor count
    nb_map = _try_recolor_by_neighbor_count(train_pairs, bg)
    if nb_map is not None:
        _nm = dict(nb_map)
        _bg_val = bg
        def apply_nb_recolor(inp_raw, nm=_nm, bg_val=_bg_val):
            inp = np.array(inp_raw)
            objects = _get_objects(inp, bg_val)
            H, W = inp.shape
            out = inp.copy()
            for obj in objects:
                expanded = np.zeros((H, W), dtype=bool)
                rows, cols = np.where(obj['mask'])
                for r, c in zip(rows, cols):
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W:
                            expanded[nr, nc] = True
                nb_count = sum(1 for other in objects if other['id'] != obj['id'] and np.any(expanded & other['mask']))
                if nb_count in nm:
                    out[obj['mask']] = nm[nb_count]
            return out.tolist()
        pieces.append(CrossPiece('obj_recolor_nb_count', apply_nb_recolor))
    
    return pieces
