import numpy as np
from scipy.ndimage import label

def transform(grid):
    arr = np.array(grid)
    non_zero = (arr != 0)
    labeled, n = label(non_zero)
    
    blocks = []
    for j in range(1, n+1):
        r_loc, c_loc = np.where(labeled == j)
        rmin, rmax = r_loc.min(), r_loc.max()
        cmin, cmax = c_loc.min(), c_loc.max()
        blocks.append(arr[rmin:rmax+1, cmin:cmax+1])
    
    if not blocks: return grid
    
    # Find the non-background value (the mark)
    all_vals = set(arr.flatten().tolist()) - {0}
    bg = max(all_vals, key=lambda v: (arr==v).sum())
    mark = next(v for v in all_vals if v != bg)
    
    def count_runs(b):
        count = 0
        for r in range(b.shape[0]):
            for c in range(b.shape[1]-2):
                if b[r,c]==mark and b[r,c+1]==mark and b[r,c+2]==mark: count+=1
        for c in range(b.shape[1]):
            for r in range(b.shape[0]-2):
                if b[r,c]==mark and b[r+1,c]==mark and b[r+2,c]==mark: count+=1
        return count
    
    best = max(blocks, key=count_runs)
    return best.tolist()
