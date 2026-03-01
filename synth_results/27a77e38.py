import numpy as np
from collections import Counter

def transform(grid):
    """Place most common color from top section at center of bottom."""
    grid = np.array(grid)
    result = grid.copy()
    h, w = grid.shape
    
    # Find the row of all 5s (separator)
    sep_row = -1
    for r in range(h):
        if all(grid[r] == 5):
            sep_row = r
            break
    
    if sep_row == -1:
        return result.tolist()
    
    # Get top section (before separator)
    top_section = grid[:sep_row]
    
    # Count all non-5 colors in top section
    colors = [c for row in top_section for c in row if c != 5]
    if not colors:
        return result.tolist()
    
    most_common = Counter(colors).most_common(1)[0][0]
    
    # Place in center of last row
    center_col = w // 2
    last_row = h - 1
    result[last_row, center_col] = most_common
    
    return result.tolist()
