def transform(grid_list):
    import numpy as np
    from collections import Counter
    
    grid = np.array(grid_list)
    H, W = grid.shape
    out = grid.copy()
    
    def tile_line(vals):
        n = len(vals)
        nz = [(i, int(vals[i])) for i in range(n) if vals[i] != 0]
        if len(nz) < 2:
            return None
        
        colors = Counter(v for _, v in nz)
        repeating = [c for c, cnt in colors.items() if cnt >= 2]
        if len(repeating) != 1:
            return None
        rep_color = repeating[0]
        
        others = [(i, v) for i, v in nz if v != rep_color]
        if len(others) > 1:
            return None
        
        rep_positions = sorted([i for i, v in nz if v == rep_color])
        if len(rep_positions) < 2:
            return None
        
        period = rep_positions[1] - rep_positions[0]
        for k in range(2, len(rep_positions)):
            if rep_positions[k] - rep_positions[k-1] != period:
                return None
        if period <= 0:
            return None
        
        # Find all periodic positions
        anchor = rep_positions[0]
        periodic = set()
        for k in range(-n, 2*n):
            pos = anchor + k * period
            if 0 <= pos < n:
                periodic.add(pos)
        
        result = [0] * n
        
        if others:
            other_pos, other_color = others[0]
            if other_pos in periodic:
                # Replace repeating with other color across range of all non-zero positions
                lo = min(other_pos, rep_positions[0])
                hi = max(other_pos, rep_positions[-1])
                for pos in periodic:
                    if lo <= pos <= hi:
                        result[pos] = other_color
            else:
                # Extend repeating color to all periodic positions, keep other at its place
                for pos in periodic:
                    result[pos] = rep_color
                result[other_pos] = other_color
        else:
            # Just extend repeating color
            for pos in periodic:
                result[pos] = rep_color
        
        # Check for change
        if all(result[i] == int(vals[i]) for i in range(n)):
            return None
        
        return result
    
    row_tiles = {}
    for r in range(H):
        res = tile_line(grid[r])
        if res:
            row_tiles[r] = res
    
    col_tiles = {}
    for c in range(W):
        res = tile_line(grid[:, c])
        if res:
            col_tiles[c] = res
    
    for r, res in row_tiles.items():
        for c in range(W):
            out[r][c] = res[c]
    
    for c, res in col_tiles.items():
        for r in range(H):
            out[r][c] = res[r]
    
    return out.tolist()
