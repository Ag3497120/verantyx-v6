def transform(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    # Find non-zero rows and determine pattern
    nz_rows = [r for r in range(H) if np.any(g[r] != 0)]
    if not nz_rows:
        return grid
    start = nz_rows[0]
    end = nz_rows[-1]
    period = end - start + 1
    pattern = g[start:start+period, :]
    # Phase: the output[0] should be pattern[(0 - start) % period]
    # i.e., the origin of tiling is such that output[start] = pattern[0]
    out = np.zeros_like(g)
    for r in range(H):
        phase_idx = (r - start) % period
        out[r] = pattern[phase_idx]
    return out.tolist()
