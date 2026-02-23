"""
arc/tile_transform.py — Tile + Transform (Expand tasks)

Output = NxM grid of transformed copies of input.
Transforms: identity, rotate90/180/270, flip_h, flip_v, flip_h(rot90), flip_v(rot90)
"""

from typing import List, Tuple, Optional, Dict
from arc.grid import Grid, grid_shape, grid_eq


def rotate90(g: Grid) -> Grid:
    h, w = grid_shape(g)
    return [[g[h-1-c][r] for c in range(h)] for r in range(w)]

def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def flip_v(g: Grid) -> Grid:
    return g[::-1]

def _all_transforms(g: Grid) -> list:
    """Generate 8 transforms (identity + 3 rotations + 4 flips)"""
    ih, iw = grid_shape(g)
    r90 = rotate90(g)
    r180 = rotate90(r90)
    r270 = rotate90(r180)
    transforms = [g, r90, r180, r270, flip_h(g), flip_v(g), flip_h(r90), flip_v(r90)]
    # Only keep transforms that maintain size (square grids allow all, rect grids filter)
    return [(i, t) for i, t in enumerate(transforms) if grid_shape(t) == (ih, iw)]


def learn_tile_transform(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn a tile+transform rule."""
    
    # All inputs same size? Not required but typical
    # All outputs must be exact multiples of input
    rh_rw = None
    
    for inp, out in train_pairs:
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        if oh % ih != 0 or ow % iw != 0:
            return None
        rh, rw = oh // ih, ow // iw
        if rh == 1 and rw == 1:
            return None
        if rh_rw is None:
            rh_rw = (rh, rw)
        elif rh_rw != (rh, rw):
            return None
    
    rh, rw = rh_rw
    
    # Learn tile_map from first pair
    inp0, out0 = train_pairs[0]
    ih, iw = grid_shape(inp0)
    
    avail = _all_transforms(inp0)
    tile_map = {}
    
    for tr in range(rh):
        for tc in range(rw):
            r0, c0 = tr * ih, tc * iw
            tile = [out0[r0+r][c0:c0+iw] for r in range(ih)]
            found = False
            for ti, t in avail:
                if grid_eq(t, tile):
                    tile_map[(tr, tc)] = ti
                    found = True
                    break
            if not found:
                return None
    
    # Verify on all pairs
    for inp, out in train_pairs[1:]:
        ih2, iw2 = grid_shape(inp)
        avail2 = _all_transforms(inp)
        
        for (tr, tc), ti in tile_map.items():
            # Find the transform with same index
            match = None
            for idx, t in avail2:
                if idx == ti:
                    match = t; break
            if match is None:
                return None
            
            r0, c0 = tr * ih2, tc * iw2
            tile = [out[r0+r][c0:c0+iw2] for r in range(ih2)]
            if not grid_eq(match, tile):
                return None
    
    return {
        'type': 'tile_transform',
        'rh': rh, 'rw': rw,
        'tile_map': {f'{tr},{tc}': ti for (tr, tc), ti in tile_map.items()},
        'name': f'tile_{rh}x{rw}_transform',
    }


def apply_tile_transform(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply tile+transform rule."""
    ih, iw = grid_shape(inp)
    rh, rw = rule['rh'], rule['rw']
    oh, ow = ih * rh, iw * rw
    
    avail = dict(_all_transforms(inp))
    tile_map = {tuple(map(int, k.split(','))): v for k, v in rule['tile_map'].items()}
    
    result = [[0] * ow for _ in range(oh)]
    for (tr, tc), ti in tile_map.items():
        if ti not in avail:
            return None
        tile = avail[ti]
        for r in range(ih):
            for c in range(iw):
                result[tr * ih + r][tc * iw + c] = tile[r][c]
    
    return result
