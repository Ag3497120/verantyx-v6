import numpy as np

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    output = grid.copy()
    
    # Find the key pattern in row 0
    key_colors = []
    key_cols = []
    for j in range(w):
        if grid[0, j] != 0:
            key_colors.append(grid[0, j])
            key_cols.append(j)
    
    if not key_colors:
        return output.tolist()
    
    # Find markers
    markers = {}
    for i in range(1, h):
        for j in range(w):
            val = grid[i, j]
            if val in key_colors:
                key_idx = key_colors.index(val)
                if j != key_cols[key_idx]:
                    if i not in markers:
                        markers[i] = {}
                    markers[i][val] = j
    
    # Track which colors are still active
    active_colors = set(key_colors)
    
    for i in range(1, h):
        if i in markers:
            # Draw marker lines
            for color, marker_col in markers[i].items():
                key_idx = key_colors.index(color)
                key_col = key_cols[key_idx]
                if marker_col < key_col:
                    output[i, marker_col:key_col+1] = color
                elif marker_col > key_col:
                    output[i, key_col:marker_col+1] = color
                # Deactivate this color after its marker row
                if color in active_colors:
                    active_colors.remove(color)
            
            # Draw other active key colors
            for idx, color in enumerate(key_colors):
                if color in active_colors or color in markers[i]:
                    output[i, key_cols[idx]] = color
        else:
            # Draw active key colors
            for idx, color in enumerate(key_colors):
                if color in active_colors:
                    output[i, key_cols[idx]] = color
    
    return output.tolist()
