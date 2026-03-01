def transform(grid):
    R, C = len(grid), len(grid[0])
    eights = [(r,c) for r in range(R) for c in range(C) if grid[r][c] == 8]
    if not eights:
        return grid
    min_r = min(r for r,c in eights); max_r = max(r for r,c in eights)
    min_c = min(c for r,c in eights); max_c = max(c for r,c in eights)
    result = []
    for i in range(max_r - min_r + 1):
        row = []
        for j in range(max_c - min_c + 1):
            r = min_r + i; c = min_c + j
            # Try 180-rotation about (15.5, 15.5)
            nr, nc = 31 - r, 31 - c
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != 8:
                row.append(grid[nr][nc])
            else:
                # Fallback: transpose
                if 0 <= c < R and 0 <= r < C:
                    row.append(grid[c][r])
                else:
                    row.append(0)
        result.append(row)
    return result
