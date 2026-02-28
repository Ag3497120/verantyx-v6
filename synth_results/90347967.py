def transform(grid):
    import numpy as np
    g = np.array(grid)
    R, C = g.shape
    nz = np.argwhere(g != 0)
    if len(nz) == 0:
        return grid
    r_min, c_min = nz.min(0)
    r_max, c_max = nz.max(0)
    sub = g[r_min:r_max+1, c_min:c_max+1]
    rot = sub[::-1, ::-1]
    H, W = rot.shape
    r_off = max(0, 2*r_min - r_max)
    c_off = int(c_max)
    out = np.zeros_like(g)
    # Ensure placement fits
    if r_off + H <= R and c_off + W <= C:
        out[r_off:r_off+H, c_off:c_off+W] = rot
    return out.tolist()
