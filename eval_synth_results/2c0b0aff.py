def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False]*cols for _ in range(rows)]
    regions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                min_r, max_r, min_c, max_c = r, r, c, c
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                            min_r = min(min_r, nr)
                            max_r = max(max_r, nr)
                            min_c = min(min_c, nc)
                            max_c = max(max_c, nc)
                region = []
                for rr in range(min_r, max_r+1):
                    row = []
                    for cc in range(min_c, max_c+1):
                        row.append(grid[rr][cc])
                    region.append(row)
                regions.append(region)
    def count_crosses(reg):
        h = len(reg)
        w = len(reg[0])
        count = 0
        for r in range(1, h-1):
            for c in range(1, w-1):
                if (reg[r][c] == 3 and reg[r-1][c] == 3 and reg[r+1][c] == 3
                    and reg[r][c-1] == 3 and reg[r][c+1] == 3):
                    count += 1
        return count
    best = max(regions, key=count_crosses)
    return best
