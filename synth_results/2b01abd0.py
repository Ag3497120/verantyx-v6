import numpy as np

def transform(grid):
    arr = np.array(grid)
    rows, cols = arr.shape
    
    # Find separator (all-1 row or column)
    sep_row = None
    sep_col = None
    for r in range(rows):
        if all(arr[r,c]==1 for c in range(cols)):
            sep_row = r; break
    for c in range(cols):
        if all(arr[r,c]==1 for r in range(rows)):
            sep_col = c; break
    
    result = arr.copy()
    
    if sep_col is not None:
        # Find non-0, non-1 cells on each side
        left_cells = {(r,c): arr[r,c] for r in range(rows) for c in range(sep_col) if arr[r,c] not in (0,1)}
        right_cells = {(r,c): arr[r,c] for r in range(rows) for c in range(sep_col+1,cols) if arr[r,c] not in (0,1)}
        
        if left_cells and not right_cells:
            # Source is left, reflect to right
            vals = list(set(left_cells.values()))
            if len(vals) >= 2:
                v1, v2 = vals[0], vals[1]
            else:
                v1 = v2 = vals[0]
            swap = {v1: v2, v2: v1}
            # Swap colors in original
            for (r,c), v in left_cells.items():
                result[r,c] = swap.get(v, v)
            # Reflect to right side (mirror across sep_col)
            for (r,c), v in left_cells.items():
                new_c = sep_col + (sep_col - c)
                if 0 <= new_c < cols:
                    result[r,new_c] = v
        elif right_cells and not left_cells:
            vals = list(set(right_cells.values()))
            if len(vals) >= 2:
                v1, v2 = vals[0], vals[1]
            else:
                v1 = v2 = vals[0]
            swap = {v1: v2, v2: v1}
            for (r,c), v in right_cells.items():
                result[r,c] = swap.get(v, v)
            for (r,c), v in right_cells.items():
                new_c = sep_col - (c - sep_col)
                if 0 <= new_c < cols:
                    result[r,new_c] = v
        elif right_cells and left_cells:
            # If both sides have cells, swap the two colors
            all_vals = set(left_cells.values()) | set(right_cells.values())
            all_vals = list(all_vals)
            if len(all_vals) >= 2:
                v1, v2 = all_vals[0], all_vals[1]
                swap = {v1: v2, v2: v1}
            else:
                swap = {}
            for (r,c), v in right_cells.items():
                result[r,c] = swap.get(v,v)
            for (r,c), v in left_cells.items():
                new_c = sep_col + (sep_col - c)
                if 0 <= new_c < cols:
                    result[r,new_c] = v
    
    elif sep_row is not None:
        top_cells = {(r,c): arr[r,c] for r in range(sep_row) for c in range(cols) if arr[r,c] not in (0,1)}
        bot_cells = {(r,c): arr[r,c] for r in range(sep_row+1,rows) for c in range(cols) if arr[r,c] not in (0,1)}
        
        if top_cells and not bot_cells:
            vals = list(set(top_cells.values()))
            if len(vals) >= 2: v1, v2 = vals[0], vals[1]
            else: v1 = v2 = vals[0]
            swap = {v1: v2, v2: v1}
            for (r,c), v in top_cells.items():
                result[r,c] = swap.get(v, v)
            for (r,c), v in top_cells.items():
                new_r = sep_row + (sep_row - r)
                if 0 <= new_r < rows:
                    result[new_r,c] = v
        elif bot_cells and not top_cells:
            vals = list(set(bot_cells.values()))
            if len(vals) >= 2: v1, v2 = vals[0], vals[1]
            else: v1 = v2 = vals[0]
            swap = {v1: v2, v2: v1}
            for (r,c), v in bot_cells.items():
                result[r,c] = swap.get(v, v)
            for (r,c), v in bot_cells.items():
                new_r = sep_row - (r - sep_row)
                if 0 <= new_r < rows:
                    result[new_r,c] = v
        elif top_cells and bot_cells:
            all_vals = list(set(top_cells.values()) | set(bot_cells.values()))
            if len(all_vals) >= 2: v1, v2 = all_vals[0], all_vals[1]; swap = {v1:v2, v2:v1}
            else: swap = {}
            for (r,c), v in top_cells.items():
                result[r,c] = swap.get(v,v)
            for (r,c), v in top_cells.items():
                new_r = sep_row + (sep_row - r)
                if 0 <= new_r < rows:
                    result[new_r,c] = v
    
    return result.tolist()
