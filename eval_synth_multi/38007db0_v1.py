def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    
    # Find border rows: rows where all cells have the same value
    def find_uniform_lines(axis):
        n = rows if axis == 0 else cols
        lines = []
        for i in range(n):
            line = g[i, :] if axis == 0 else g[:, i]
            if len(set(line.tolist())) == 1:
                lines.append(i)
        return lines
    
    h_borders = find_uniform_lines(0)
    v_borders = find_uniform_lines(1)
    
    tile_h = h_borders[1] - h_borders[0] + 1
    tile_w = v_borders[1] - v_borders[0] + 1
    
    n_tile_rows = len(h_borders) - 1
    n_tile_cols = len(v_borders) - 1
    
    result_tiles = []
    for i in range(n_tile_rows):
        row_tiles = []
        for j in range(n_tile_cols):
            r1 = h_borders[i]
            c1 = v_borders[j]
            tile = g[r1:r1+tile_h, c1:c1+tile_w].copy()
            row_tiles.append(tile)
        
        # Find unique tile
        n = len(row_tiles)
        unique_idx = 0
        for idx in range(n):
            matches = sum(1 for j in range(n) if j != idx and np.array_equal(row_tiles[idx], row_tiles[j]))
            if matches == 0:
                unique_idx = idx
                break
        result_tiles.append(row_tiles[unique_idx])
    
    # Stack - but shared border rows need dedup
    # Just take the first tile fully, then for subsequent tiles skip the first row (shared border)
    result = result_tiles[0]
    for t in result_tiles[1:]:
        result = np.vstack([result, t[1:]])
    
    return result.tolist()
