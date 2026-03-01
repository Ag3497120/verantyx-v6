def transform(grid):
    rows = len(grid); cols = len(grid[0])
    result = [row[:] for row in grid]
    cluster = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c]==3]
    if not cluster: return result
    minr=min(r for r,c in cluster); maxr=max(r for r,c in cluster)
    minc=min(c for r,c in cluster); maxc=max(c for r,c in cluster)
    for r in range(rows):
        for c in range(cols):
            v = grid[r][c]
            if v in (0,3): continue
            if minr <= r <= maxr:
                if c < minc:
                    for cc in range(c+1, minc): result[r][cc] = v
                elif c > maxc:
                    for cc in range(maxc+1, c): result[r][cc] = v
            elif minc <= c <= maxc:
                if r < minr:
                    for rr in range(r+1, minr): result[rr][c] = v
                elif r > maxr:
                    for rr in range(maxr+1, r): result[rr][c] = v
    return result
