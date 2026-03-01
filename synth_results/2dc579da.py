import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find separator color (appears in complete rows/columns)
    from collections import Counter
    color_counts = Counter(grid.flatten())
    
    # Find rows and columns that are all one color
    sep_rows = []
    sep_cols = []
    
    for i in range(h):
        if len(set(grid[i, :])) == 1:
            sep_rows.append(i)
    
    for j in range(w):
        if len(set(grid[:, j])) == 1:
            sep_cols.append(j)
    
    # If we have separators, extract the region without them
    if sep_rows and sep_cols:
        # Define regions by separators
        row_ranges = []
        prev = 0
        for sep in sep_rows:
            if sep > prev:
                row_ranges.append((prev, sep))
            prev = sep + 1
        if prev < h:
            row_ranges.append((prev, h))
        
        col_ranges = []
        prev = 0
        for sep in sep_cols:
            if sep > prev:
                col_ranges.append((prev, sep))
            prev = sep + 1
        if prev < w:
            col_ranges.append((prev, w))
        
        # Find the region with the most variation (contains the interesting pattern)
        best_region = None
        max_unique = 0
        
        for r1, r2 in row_ranges:
            for c1, c2 in col_ranges:
                region = grid[r1:r2, c1:c2]
                unique_count = len(set(region.flatten()))
                if unique_count > max_unique:
                    max_unique = unique_count
                    best_region = region
        
        if best_region is not None:
            return best_region.tolist()
    
    return grid.tolist()
