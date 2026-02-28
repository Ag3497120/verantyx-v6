def transform(grid):
    import numpy as np
    g = np.array(grid)
    N, M = g.shape  # Usually N==M
    H, W = N*N, M*M
    out = np.zeros((H, W), dtype=int)
    for i in range(N):
        for j in range(M):
            target = g[i, j]
            block = np.where(g == target, g, 0)
            out[i*N:(i+1)*N, j*M:(j+1)*M] = block
    return out.tolist()
