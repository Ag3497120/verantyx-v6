
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    markers = {}
    for c in range(cols):
        v = grid[0][c]
        if v != 0 and v != 5:
            markers[c] = v
    if not markers:
        return grid
    out = [row[:] for row in grid]
    visited = [[False]*cols for _ in range(rows)]
    def bfs(sr, sc):
        comp = []
        stack = [(sr, sc)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols: continue
            if visited[r][c] or grid[r][c] != 5: continue
            visited[r][c] = True
            comp.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((r+dr, c+dc))
        return comp
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not visited[r][c]:
                comp = bfs(r, c)
                avg_col = sum(cc for _, cc in comp) / len(comp)
                best_col = min(markers.keys(), key=lambda mc: abs(mc - avg_col))
                for cr, cc in comp:
                    out[cr][cc] = markers[best_col]
    return out
