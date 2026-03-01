import numpy as np

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    result = arr.copy()
    
    # Find rows/cols with max number of zeros
    row_zeros = [(r, int((arr[r]==0).sum())) for r in range(rows)]
    col_zeros = [(c, int((arr[:,c]==0).sum())) for c in range(cols)]
    
    max_rv = max(v for r, v in row_zeros)
    max_cv = max(v for c, v in col_zeros)
    
    zero_rows = [r for r, v in row_zeros if v == max_rv]
    zero_cols = [c for c, v in col_zeros if v == max_cv]
    
    # Zero out non-zero, non-2 cells in those rows and cols
    for r in zero_rows:
        for c in range(cols):
            if result[r, c] != 0 and result[r, c] != 2:
                result[r, c] = 0
    for c in zero_cols:
        for r in range(rows):
            if result[r, c] != 0 and result[r, c] != 2:
                result[r, c] = 0
    
    return result.tolist()
