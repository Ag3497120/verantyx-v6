def transform(grid):
    import numpy as np
    arr = np.array(grid)
    # fliplr then swap 2<->0 (2→0, 0→8... actually 0→8)
    flipped = np.fliplr(arr)
    swapped = np.where(flipped==2, 0, np.where(flipped==0, 8, flipped))
    # Determine which side to put input: left edge all 0? put input left, swap right
    left_col = arr[:,0]
    right_col = arr[:,-1]
    if all(left_col == 0):
        return np.hstack([arr, swapped]).tolist()
    else:
        return np.hstack([swapped, arr]).tolist()