def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape  # Should be 3x3
    Z = int(np.sum(g == 0))  # count of zeros
    N = R * C - Z  # count of non-zeros
    out_size = Z * 3
    out = np.zeros((out_size, out_size), dtype=int)
    count = 0
    for tile_r in range(Z):
        for tile_c in range(Z):
            if count < N:
                out[tile_r*3:(tile_r+1)*3, tile_c*3:(tile_c+1)*3] = g
                count += 1
    return out.tolist()
