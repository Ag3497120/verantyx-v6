"""
arc/extract_patch.py — Patch Extraction from Input Grid

Cross decomposition for shrink tasks:
  Layer 1: Determine output size from training pairs
  Layer 2: Find extraction rule (WHERE to extract)
  Layer 3: Find transform rule (HOW to post-process extracted patch)

Extraction strategies:
- Unique subgrid (the one region that differs from the repeated pattern)
- Most colorful region
- Region around a marker
- Object bounding box extraction
- Majority/minority vote across repeated subgrids
"""

from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors
from arc.objects import detect_objects


def learn_extract_rule(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn a patch extraction rule from training pairs."""
    
    # Check: all outputs same size?
    out_sizes = [grid_shape(o) for _, o in train_pairs]
    if len(set(out_sizes)) != 1:
        return None
    
    oh, ow = out_sizes[0]
    
    # All inputs must be larger than output
    for inp, _ in train_pairs:
        ih, iw = grid_shape(inp)
        if ih < oh or iw < ow:
            return None
    
    for strategy in [
        _learn_unique_subgrid,
        _learn_most_colorful_patch,
        _learn_marker_extract,
        _learn_nonbg_bbox_extract,
        _learn_subgrid_vote,
        _learn_unique_object_extract,
    ]:
        rule = strategy(train_pairs, oh, ow)
        if rule is not None:
            # Verify
            ok = True
            for inp, out in train_pairs:
                result = apply_extract_rule(inp, rule)
                if result is None or not grid_eq(result, out):
                    ok = False; break
            if ok:
                return rule
    
    return None


def _learn_unique_subgrid(train_pairs, oh, ow):
    """Extract the unique subgrid from a tiled/repeated pattern."""
    
    for inp, out in train_pairs:
        ih, iw = grid_shape(inp)
        if ih % oh != 0 or iw % ow != 0:
            return None
        
        # Extract all oh×ow patches
        patches = []
        for r0 in range(0, ih, oh):
            for c0 in range(0, iw, ow):
                patch = [inp[r0+r][c0:c0+ow] for r in range(oh)]
                patches.append((r0, c0, patch))
        
        if len(patches) < 2:
            return None
        
        # Find the unique patch (appears once, others are identical/similar)
        patch_strs = [str(p) for _, _, p in patches]
        counts = Counter(patch_strs)
        
        if len(counts) < 2:
            return None
    
    # Try: output = the patch that appears least often (unique)
    consistent = True
    for inp, out in train_pairs:
        ih, iw = grid_shape(inp)
        patches = []
        for r0 in range(0, ih, oh):
            for c0 in range(0, iw, ow):
                patch = [inp[r0+r][c0:c0+ow] for r in range(oh)]
                patches.append(patch)
        
        patch_strs = [str(p) for p in patches]
        counts = Counter(patch_strs)
        min_count = min(counts.values())
        
        # Find least common patch
        for p, ps in zip(patches, patch_strs):
            if counts[ps] == min_count:
                if not grid_eq(p, out):
                    consistent = False
                break
        if not consistent:
            break
    
    if consistent:
        return {
            'type': 'unique_subgrid',
            'oh': oh, 'ow': ow,
            'name': f'extract_unique_{oh}x{ow}',
        }
    
    return None


def _learn_most_colorful_patch(train_pairs, oh, ow):
    """Extract the patch with the most distinct colors."""
    
    consistent = True
    for inp, out in train_pairs:
        ih, iw = grid_shape(inp)
        bg = most_common_color(inp)
        
        best_patch = None
        best_colors = 0
        
        for r0 in range(ih - oh + 1):
            for c0 in range(iw - ow + 1):
                patch = [inp[r0+r][c0:c0+ow] for r in range(oh)]
                n_colors = len(set(v for row in patch for v in row))
                if n_colors > best_colors:
                    best_colors = n_colors
                    best_patch = patch
        
        if best_patch is None or not grid_eq(best_patch, out):
            consistent = False; break
    
    if consistent:
        return {
            'type': 'most_colorful_patch',
            'oh': oh, 'ow': ow,
            'name': f'extract_most_colorful_{oh}x{ow}',
        }
    return None


def _learn_marker_extract(train_pairs, oh, ow):
    """Extract patch around a specific marker color."""
    
    for marker_color in range(10):
        consistent = True
        for inp, out in train_pairs:
            ih, iw = grid_shape(inp)
            bg = most_common_color(inp)
            if marker_color == bg:
                consistent = False; break
            
            # Find marker positions
            markers = [(r, c) for r in range(ih) for c in range(iw) 
                       if inp[r][c] == marker_color]
            if not markers:
                consistent = False; break
            
            # Try extracting patch centered on marker centroid
            mr = sum(r for r, c in markers) // len(markers)
            mc = sum(c for r, c in markers) // len(markers)
            
            r0 = mr - oh // 2
            c0 = mc - ow // 2
            
            if r0 < 0 or c0 < 0 or r0 + oh > ih or c0 + ow > iw:
                consistent = False; break
            
            patch = [inp[r0+r][c0:c0+ow] for r in range(oh)]
            if not grid_eq(patch, out):
                consistent = False; break
        
        if consistent:
            return {
                'type': 'marker_extract',
                'marker_color': marker_color,
                'oh': oh, 'ow': ow,
                'name': f'extract_at_marker_{marker_color}_{oh}x{ow}',
            }
    
    return None


def _learn_nonbg_bbox_extract(train_pairs, oh, ow):
    """Extract the bounding box of all non-bg cells (or the largest object)."""
    
    consistent = True
    for inp, out in train_pairs:
        ih, iw = grid_shape(inp)
        bg = most_common_color(inp)
        
        # Find bbox of all non-bg
        rows = [r for r in range(ih) for c in range(iw) if inp[r][c] != bg]
        cols = [c for r in range(ih) for c in range(iw) if inp[r][c] != bg]
        if not rows:
            consistent = False; break
        
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)
        
        if r2 - r1 + 1 != oh or c2 - c1 + 1 != ow:
            consistent = False; break
        
        patch = [inp[r1+r][c1:c1+ow] for r in range(oh)]
        if not grid_eq(patch, out):
            consistent = False; break
    
    if consistent:
        return {
            'type': 'nonbg_bbox',
            'oh': oh, 'ow': ow,
            'name': f'extract_nonbg_bbox_{oh}x{ow}',
        }
    return None


def _learn_subgrid_vote(train_pairs, oh, ow):
    """Output = majority/minority vote across tiled subgrids."""
    
    for vote_type in ['majority', 'minority', 'xor']:
        consistent = True
        for inp, out in train_pairs:
            ih, iw = grid_shape(inp)
            if ih % oh != 0 or iw % ow != 0:
                consistent = False; break
            
            # Extract all patches
            patches = []
            for r0 in range(0, ih, oh):
                for c0 in range(0, iw, ow):
                    patch = [inp[r0+r][c0:c0+ow] for r in range(oh)]
                    patches.append(patch)
            
            if len(patches) < 2:
                consistent = False; break
            
            bg = most_common_color(inp)
            result = [[bg] * ow for _ in range(oh)]
            
            for r in range(oh):
                for c in range(ow):
                    values = [p[r][c] for p in patches]
                    counts = Counter(values)
                    
                    if vote_type == 'majority':
                        result[r][c] = counts.most_common(1)[0][0]
                    elif vote_type == 'minority':
                        result[r][c] = counts.most_common()[-1][0]
                    elif vote_type == 'xor':
                        # Non-bg value if exactly one patch has non-bg
                        non_bg = [v for v in values if v != bg]
                        if len(non_bg) == 1:
                            result[r][c] = non_bg[0]
                        elif len(non_bg) == 0:
                            result[r][c] = bg
                        else:
                            # Multiple non-bg: use if all same
                            if len(set(non_bg)) == 1:
                                result[r][c] = non_bg[0]
                            else:
                                result[r][c] = bg
            
            if not grid_eq(result, out):
                consistent = False; break
        
        if consistent:
            return {
                'type': 'subgrid_vote',
                'vote': vote_type,
                'oh': oh, 'ow': ow,
                'name': f'extract_vote_{vote_type}_{oh}x{ow}',
            }
    
    return None


def _learn_unique_object_extract(train_pairs, oh, ow):
    """Extract the bounding box of a unique object (by color, shape, or size)."""
    
    for selector in ['smallest', 'largest', 'unique_color', 'unique_shape']:
        consistent = True
        for inp, out in train_pairs:
            bg = most_common_color(inp)
            objs = detect_objects(inp, bg)
            if not objs:
                consistent = False; break
            
            if selector == 'smallest':
                target = min(objs, key=lambda o: o.size)
            elif selector == 'largest':
                target = max(objs, key=lambda o: o.size)
            elif selector == 'unique_color':
                color_counts = Counter(o.color for o in objs)
                unique = [o for o in objs if color_counts[o.color] == 1]
                if len(unique) != 1:
                    consistent = False; break
                target = unique[0]
            elif selector == 'unique_shape':
                shape_counts = Counter(o.shape for o in objs)
                unique = [o for o in objs if shape_counts[o.shape] == 1]
                if len(unique) != 1:
                    consistent = False; break
                target = unique[0]
            
            r1, c1, r2, c2 = target.bbox
            if r2 - r1 + 1 != oh or c2 - c1 + 1 != ow:
                consistent = False; break
            
            patch = [inp[r1+r][c1:c1+ow] for r in range(oh)]
            if not grid_eq(patch, out):
                consistent = False; break
        
        if consistent:
            return {
                'type': 'unique_object_extract',
                'selector': selector,
                'oh': oh, 'ow': ow,
                'name': f'extract_{selector}_obj_{oh}x{ow}',
            }
    
    return None


def apply_extract_rule(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply an extraction rule to get the output."""
    ih, iw = grid_shape(inp)
    oh, ow = rule['oh'], rule['ow']
    rtype = rule['type']
    bg = most_common_color(inp)
    
    if rtype == 'unique_subgrid':
        if ih % oh != 0 or iw % ow != 0:
            return None
        patches = []
        for r0 in range(0, ih, oh):
            for c0 in range(0, iw, ow):
                patch = [inp[r0+r][c0:c0+ow] for r in range(oh)]
                patches.append(patch)
        
        patch_strs = [str(p) for p in patches]
        counts = Counter(patch_strs)
        min_count = min(counts.values())
        
        for p, ps in zip(patches, patch_strs):
            if counts[ps] == min_count:
                return p
        return None
    
    elif rtype == 'most_colorful_patch':
        best_patch = None
        best_colors = 0
        for r0 in range(ih - oh + 1):
            for c0 in range(iw - ow + 1):
                patch = [inp[r0+r][c0:c0+ow] for r in range(oh)]
                n_colors = len(set(v for row in patch for v in row))
                if n_colors > best_colors:
                    best_colors = n_colors
                    best_patch = patch
        return best_patch
    
    elif rtype == 'marker_extract':
        marker_color = rule['marker_color']
        markers = [(r, c) for r in range(ih) for c in range(iw) 
                   if inp[r][c] == marker_color]
        if not markers:
            return None
        mr = sum(r for r, c in markers) // len(markers)
        mc = sum(c for r, c in markers) // len(markers)
        r0 = mr - oh // 2
        c0 = mc - ow // 2
        if r0 < 0 or c0 < 0 or r0 + oh > ih or c0 + ow > iw:
            return None
        return [inp[r0+r][c0:c0+ow] for r in range(oh)]
    
    elif rtype == 'nonbg_bbox':
        rows = [r for r in range(ih) for c in range(iw) if inp[r][c] != bg]
        cols = [c for r in range(ih) for c in range(iw) if inp[r][c] != bg]
        if not rows:
            return None
        r1, c1 = min(rows), min(cols)
        return [inp[r1+r][c1:c1+ow] for r in range(oh)]
    
    elif rtype == 'subgrid_vote':
        vote_type = rule['vote']
        if ih % oh != 0 or iw % ow != 0:
            return None
        patches = []
        for r0 in range(0, ih, oh):
            for c0 in range(0, iw, ow):
                patches.append([inp[r0+r][c0:c0+ow] for r in range(oh)])
        
        result = [[bg] * ow for _ in range(oh)]
        for r in range(oh):
            for c in range(ow):
                values = [p[r][c] for p in patches]
                counts = Counter(values)
                if vote_type == 'majority':
                    result[r][c] = counts.most_common(1)[0][0]
                elif vote_type == 'minority':
                    result[r][c] = counts.most_common()[-1][0]
                elif vote_type == 'xor':
                    non_bg = [v for v in values if v != bg]
                    if len(non_bg) == 1:
                        result[r][c] = non_bg[0]
                    elif non_bg and len(set(non_bg)) == 1:
                        result[r][c] = non_bg[0]
                    else:
                        result[r][c] = bg
        return result
    
    elif rtype == 'unique_object_extract':
        selector = rule['selector']
        objs = detect_objects(inp, bg)
        if not objs:
            return None
        
        if selector == 'smallest':
            target = min(objs, key=lambda o: o.size)
        elif selector == 'largest':
            target = max(objs, key=lambda o: o.size)
        elif selector == 'unique_color':
            color_counts = Counter(o.color for o in objs)
            unique = [o for o in objs if color_counts[o.color] == 1]
            if not unique:
                return None
            target = unique[0]
        elif selector == 'unique_shape':
            shape_counts = Counter(o.shape for o in objs)
            unique = [o for o in objs if shape_counts[o.shape] == 1]
            if not unique:
                return None
            target = unique[0]
        else:
            return None
        
        r1, c1, r2, c2 = target.bbox
        return [inp[r1+r][c1:c1+(c2-c1+1)] for r in range(r2-r1+1)]
    
    return None
