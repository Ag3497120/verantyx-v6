
def transform(grid):
    import numpy as np
    g = np.array(grid)
    h, w = g.shape
    n = h // w  # number of blocks
    blocks = [g[i*w:(i+1)*w, :] for i in range(n)]
    # Find the block that is not diagonally symmetric (transpose != itself)
    for b in blocks:
        if not np.array_equal(b, b.T):
            return b.tolist()
    # Fallback: return first non-symmetric or last
    return blocks[-1].tolist()
