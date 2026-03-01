import numpy as np

def transform(grid):
    arr = np.array(grid)
    result = arr.copy()
    r8, c8 = np.where(arr == 8)
    if len(r8) == 0: return grid
    rmin, rmax, cmin, cmax = r8.min(), r8.max(), c8.min(), c8.max()
    for r in range(rmin, rmax+1):
        for c in range(cmin, cmax+1):
            if arr[r,c] == 1:
                result[r,c] = 3
    return result.tolist()
