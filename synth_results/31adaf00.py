def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    zero = (g == 0)
    
    dp = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if zero[r, c]:
                if r == 0 or c == 0:
                    dp[r, c] = 1
                else:
                    dp[r, c] = min(dp[r-1, c], dp[r, c-1], dp[r-1, c-1]) + 1
    
    for r in range(rows):
        for c in range(cols):
            s = dp[r, c]
            if s < 2:
                continue
            # Strict local max: dp[r,c] > all neighbors
            is_local_max = True
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if dp[nr, nc] >= s:  # neighbor is >= s → not strict local max
                            is_local_max = False
                            break
                if not is_local_max:
                    break
            if is_local_max:
                r1, c1 = r - s + 1, c - s + 1
                result[r1:r+1, c1:c+1] = 1
    
    return result.tolist()
