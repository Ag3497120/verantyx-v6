import numpy as np

def transform(grid):
    g = np.array(grid)
    h, w = g.shape
    out = np.zeros_like(g)
    
    # Find columns with 5s
    col_lengths = {}
    for c in range(w):
        count = np.sum(g[:, c] == 5)
        if count > 0:
            col_lengths[c] = count
    
    if not col_lengths:
        return grid
    
    max_len = max(col_lengths.values())
    min_len = min(col_lengths.values())
    
    # Tallest column(s) -> 1, shortest column(s) -> 2
    for c, length in col_lengths.items():
        rows = np.where(g[:, c] == 5)[0]
        if length == max_len:
            out[rows, c] = 1
        elif length == min_len:
            out[rows, c] = 2
        # Others: 0 (removed)
    
    return out.tolist()
