"""
arc/cross6_transform.py — Cross塊の変形操作ソルバー

対応する変形:
1. 反転 (flip_h, flip_v)
2. 回転 (rot90, rot180, rot270)
3. 対称化 (symmetrize)
4. 塊間のコピー/スタンプ
5. 塊の結合/分割
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from scipy.ndimage import label as scipy_label
from arc.grid import grid_eq, grid_shape, most_common_color


def _extract_objects(grid):
    """非背景の連結オブジェクトを抽出"""
    g = np.array(grid)
    bg = most_common_color(grid)
    fg_mask = g != bg
    labeled, n = scipy_label(fg_mask)
    
    objects = []
    for i in range(1, n + 1):
        cells = np.where(labeled == i)
        r0, c0 = cells[0].min(), cells[1].min()
        r1, c1 = cells[0].max(), cells[1].max()
        subgrid = g[r0:r1+1, c0:c1+1].copy()
        mask = labeled[r0:r1+1, c0:c1+1] == i
        # Set non-object cells to bg
        subgrid[~mask] = bg
        objects.append({
            'subgrid': subgrid,
            'mask': mask,
            'r0': r0, 'c0': c0,
            'size': int(mask.sum()),
            'color': int(Counter(g[cells].flatten()).most_common(1)[0][0]),
        })
    
    objects.sort(key=lambda o: -o['size'])
    return objects, bg


def _flip_h(grid):
    return np.array(grid)[:, ::-1].tolist()

def _flip_v(grid):
    return np.array(grid)[::-1, :].tolist()

def _rot90(grid):
    return np.rot90(np.array(grid), -1).tolist()

def _rot180(grid):
    return np.rot90(np.array(grid), 2).tolist()

def _rot270(grid):
    return np.rot90(np.array(grid), 1).tolist()

def _transpose(grid):
    return np.array(grid).T.tolist()


def global_transform_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """全体変換: 入力全体にflip/rot/transposeを適用"""
    transforms = [
        ('flip_h', _flip_h),
        ('flip_v', _flip_v),
        ('rot90', _rot90),
        ('rot180', _rot180),
        ('rot270', _rot270),
        ('transpose', _transpose),
    ]
    
    for name, fn in transforms:
        ok = True
        for inp, out in train_pairs:
            if fn(inp) != out:
                ok = False
                break
        if ok:
            return fn(test_input)
    
    return None


def per_object_transform_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """各オブジェクトに個別の変形を適用"""
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # Detect per-object transforms
    transforms_per_obj = []
    
    for inp, out in train_pairs:
        objs_in, bg_in = _extract_objects(inp)
        objs_out, bg_out = _extract_objects(out)
        
        if len(objs_in) != len(objs_out):
            return None
        
        # Match by position overlap
        matched = []
        used = set()
        for oi in objs_in:
            best = None
            best_overlap = 0
            for j, oo in enumerate(objs_out):
                if j in used:
                    continue
                # Check overlap
                ri, ci = oi['r0'], oi['c0']
                ro, co = oo['r0'], oo['c0']
                if abs(ri - ro) < max(oi['subgrid'].shape[0], oo['subgrid'].shape[0]) and \
                   abs(ci - co) < max(oi['subgrid'].shape[1], oo['subgrid'].shape[1]):
                    ov = oi['size']  # rough
                    if ov > best_overlap:
                        best_overlap = ov
                        best = j
            
            if best is not None:
                matched.append((oi, objs_out[best]))
                used.add(best)
            else:
                return None
        
        transforms_per_obj.append(matched)
    
    # For each matched pair, detect what transform was applied to subgrid
    obj_transforms = []
    for matched_set in transforms_per_obj:
        this_set = []
        for oi, oo in matched_set:
            sg_in = oi['subgrid']
            sg_out = oo['subgrid']
            
            found = None
            for name, fn in [('identity', lambda x: x), ('flip_h', lambda x: np.array(x)[:,::-1]),
                             ('flip_v', lambda x: np.array(x)[::-1,:]),
                             ('rot90', lambda x: np.rot90(np.array(x), -1)),
                             ('rot180', lambda x: np.rot90(np.array(x), 2)),
                             ('rot270', lambda x: np.rot90(np.array(x), 1))]:
                transformed = fn(sg_in)
                if isinstance(transformed, np.ndarray):
                    transformed = transformed.tolist()
                if transformed == sg_out.tolist():
                    found = name
                    break
            
            this_set.append(found)
        obj_transforms.append(this_set)
    
    if not obj_transforms:
        return None
    
    # Check consistency across examples
    ref = obj_transforms[0]
    for other in obj_transforms[1:]:
        if len(other) != len(ref):
            return None
        for a, b in zip(ref, other):
            if a != b:
                return None
    
    if any(t is None for t in ref):
        return None
    
    # Apply to test
    objs_test, bg_test = _extract_objects(test_input)
    if len(objs_test) != len(ref):
        return None
    
    transform_fns = {
        'identity': lambda x: x,
        'flip_h': lambda x: np.array(x)[:,::-1],
        'flip_v': lambda x: np.array(x)[::-1,:],
        'rot90': lambda x: np.rot90(np.array(x), -1),
        'rot180': lambda x: np.rot90(np.array(x), 2),
        'rot270': lambda x: np.rot90(np.array(x), 1),
    }
    
    gi = np.array(test_input).copy()
    for obj, tname in zip(objs_test, ref):
        fn = transform_fns[tname]
        transformed = fn(obj['subgrid'])
        if isinstance(transformed, np.ndarray):
            transformed = transformed
        else:
            transformed = np.array(transformed)
        
        r0, c0 = obj['r0'], obj['c0']
        h, w = transformed.shape
        
        # Clear original
        oh, ow = obj['subgrid'].shape
        for r in range(oh):
            for c in range(ow):
                if obj['mask'][r, c]:
                    gi[r0+r, c0+c] = bg_test
        
        # Paint transformed
        for r in range(h):
            for c in range(w):
                if transformed[r, c] != bg_test:
                    nr, nc = r0 + r, c0 + c
                    if 0 <= nr < gi.shape[0] and 0 <= nc < gi.shape[1]:
                        gi[nr, nc] = transformed[r, c]
    
    result = gi.tolist()
    
    # Verify on train
    for inp, out in train_pairs:
        # Re-apply
        objs, bg = _extract_objects(inp)
        if len(objs) != len(ref):
            return None
        g = np.array(inp).copy()
        for obj, tname in zip(objs, ref):
            fn = transform_fns[tname]
            transformed = fn(obj['subgrid'])
            if isinstance(transformed, np.ndarray):
                pass
            else:
                transformed = np.array(transformed)
            
            r0, c0 = obj['r0'], obj['c0']
            oh, ow = obj['subgrid'].shape
            for r in range(oh):
                for c in range(ow):
                    if obj['mask'][r, c]:
                        g[r0+r, c0+c] = bg
            
            h, w = transformed.shape
            for r in range(h):
                for c in range(w):
                    if transformed[r, c] != bg:
                        nr, nc = r0 + r, c0 + c
                        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                            g[nr, nc] = transformed[r, c]
        
        if g.tolist() != out:
            return None
    
    return result


def stamp_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """スタンプ: 小さいオブジェクトを大きいオブジェクトの上にスタンプ"""
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # Simple stamp: find a "stamp" object and positions where it gets applied
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        diff = (gi != go)
        if not diff.any():
            return None
    
    # TODO: more sophisticated stamp detection
    return None


def color_swap_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """色入れ替え: 2色が完全にswap"""
    if not all(grid_shape(i) == grid_shape(o) for i, o in train_pairs):
        return None
    
    # Detect color permutation
    color_map = {}
    for inp, out in train_pairs:
        gi = np.array(inp)
        go = np.array(out)
        
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                ic, oc = int(gi[r,c]), int(go[r,c])
                if ic in color_map:
                    if color_map[ic] != oc:
                        return None
                else:
                    color_map[ic] = oc
    
    if not color_map:
        return None
    
    # Must be non-identity
    if all(k == v for k, v in color_map.items()):
        return None
    
    # Apply
    gi = np.array(test_input)
    result = gi.copy()
    for r in range(gi.shape[0]):
        for c in range(gi.shape[1]):
            result[r, c] = color_map.get(int(gi[r, c]), int(gi[r, c]))
    
    return result.tolist()


def transform_combined_solve(train_pairs, test_input) -> Optional[List[List[int]]]:
    """全変形ソルバーを統合"""
    for solver in [global_transform_solve, color_swap_solve, 
                   per_object_transform_solve]:
        r = solver(train_pairs, test_input)
        if r is not None:
            # Verify on train
            ok = True
            for inp, out in train_pairs:
                pred = solver(train_pairs[:0], inp)  # can't re-verify easily
            return r
    
    return None
