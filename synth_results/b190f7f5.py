def transform(grid):
    import numpy as np
    a = np.array(grid)
    return np.fliplr(np.flipud(a.T)).tolist()