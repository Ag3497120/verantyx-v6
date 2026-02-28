def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out = np.zeros((h, w - 2), dtype=int)
    
    for r in range(h):
        row = grid[r]
        new_row = []
        i = 0
        while i < w:
            if i + 2 < w and row[i] == 5 and row[i + 2] == 5:
                mid = row[i + 1]
                if mid == 0:
                    new_row.extend([row[i - 1]] * 3 if i > 0 and row[i - 1] != 5 else [0, 0, 0])
                else:
                    new_row.extend([mid] * 3)
                i += 3
            else:
                if i < w - 2 or (i >= w - 2 and not (row[i] == 5 and i + 2 < w)):
                    new_row.append(row[i])
                i += 1
        out[r, :len(new_row)] = new_row[:w - 2]
    
    return out.tolist()