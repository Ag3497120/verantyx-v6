import math
from collections import Counter, defaultdict

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [[0]*cols for _ in range(rows)]
    
    # Find all columns with non-zero values
    col_data = defaultdict(list)
    for c in range(cols):
        for r in range(rows):
            if grid[r][c] != 0:
                col_data[c].append(grid[r][c])
    
    for center_col, vals in col_data.items():
        total = len(vals)
        h = int(math.isqrt(total))
        if h*h != total:
            continue
        
        cnt = Counter(vals)
        sorted_colors = sorted(cnt.keys(), key=lambda v: cnt[v])
        
        # Assign colors to layers
        layer_colors = []
        layer_idx = 0
        for color in sorted_colors:
            remaining = cnt[color]
            while remaining > 0:
                layer_size = 2*layer_idx + 1
                layer_colors.append((layer_idx, color))
                remaining -= layer_size
                layer_idx += 1
        
        # Place triangle at bottom
        for layer_i, color in layer_colors:
            row_idx = rows - h + layer_i
            width = 2*layer_i + 1
            start_col = center_col - layer_i
            for j in range(width):
                c = start_col + j
                if 0 <= c < cols:
                    result[row_idx][c] = color
    
    return result
