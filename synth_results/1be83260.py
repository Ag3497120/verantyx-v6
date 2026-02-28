def transform(grid):
    import numpy as np
    grid = np.array(grid)
    h, w = grid.shape
    out_rows = []
    
    i = 0
    while i < h:
        if np.all(grid[i] == 0):
            i += 1
            continue
        start = i
        while i < h and not np.all(grid[i] == 0):
            i += 1
        block = grid[start:i]
        
        col = 0
        while col < w and np.all(block[:, col] == 0):
            col += 1
        left = col
        col = w - 1
        while col >= 0 and np.all(block[:, col] == 0):
            col -= 1
        right = col + 1
        block = block[:, left:right]
        
        bh, bw = block.shape
        mid = bw // 2
        left_block = block[:, :mid]
        right_block = block[:, mid+1:]
        
        left_h, left_w = left_block.shape
        right_h, right_w = right_block.shape
        
        if left_h == 7 and left_w == 5:
            pattern_h = 7
            pattern_w = 5
            left_pattern = left_block[:pattern_h, :pattern_w]
            right_pattern = right_block[:pattern_h, :pattern_w]
            
            left_colors = sorted(set(left_pattern.flatten()) - {0})
            right_colors = sorted(set(right_pattern.flatten()) - {0})
            
            if len(left_colors) >= 2 and len(right_colors) >= 2:
                left_map = {left_colors[0]: right_colors[0], left_colors[1]: right_colors[1]}
                right_map = {right_colors[0]: left_colors[0], right_colors[1]: left_colors[1]}
            else:
                left_map = {}
                right_map = {}
            
            new_left = np.zeros_like(left_pattern)
            for r in range(pattern_h):
                for c in range(pattern_w):
                    val = left_pattern[r, c]
                    new_left[r, c] = left_map.get(val, val)
            
            new_right = np.zeros_like(right_pattern)
            for r in range(pattern_h):
                for c in range(pattern_w):
                    val = right_pattern[r, c]
                    new_right[r, c] = right_map.get(val, val)
            
            separator = np.full((pattern_h, 1), right_colors[0] if len(right_colors) > 0 else 1)
            combined = np.hstack([new_left, separator, new_right])
            out_rows.append(combined)
        else:
            out_rows.append(block)
    
    if out_rows:
        return np.vstack(out_rows).tolist()
    return []