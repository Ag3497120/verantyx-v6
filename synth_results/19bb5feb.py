import numpy as np

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find 8-background region bounding box
    r8, c8 = np.where(arr == 8)
    if len(r8) == 0: return grid
    rmin, rmax = r8.min(), r8.max()
    cmin, cmax = c8.min(), c8.max()
    
    center_r = (rmin + rmax) / 2.0
    center_c = (cmin + cmax) / 2.0
    
    # Find colored 2x2 blocks inside 8-region
    result = [[0,0],[0,0]]
    
    other_vals = set(arr.flatten().tolist()) - {0, 8}
    for v in other_vals:
        r_v, c_v = np.where(arr == v)
        if len(r_v) == 0: continue
        avg_r = r_v.mean()
        avg_c = c_v.mean()
        
        quad_r = 0 if avg_r < center_r else 1
        quad_c = 0 if avg_c < center_c else 1
        result[quad_r][quad_c] = int(v)
    
    return result
