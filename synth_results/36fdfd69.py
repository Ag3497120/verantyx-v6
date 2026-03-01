from collections import deque

def transform(grid):
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    twos = [(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    if not twos: return result
    visited = [False]*len(twos)
    groups = []
    for i in range(len(twos)):
        if visited[i]: continue
        g = [twos[i]]
        visited[i] = True
        q = deque([i])
        while q:
            idx = q.popleft()
            r1,c1 = twos[idx]
            for j in range(len(twos)):
                if not visited[j]:
                    r2,c2 = twos[j]
                    if max(abs(r1-r2),abs(c1-c2)) <= 2:
                        visited[j] = True
                        g.append(twos[j])
                        q.append(j)
        groups.append(g)
    for g in groups:
        min_r = min(r for r,c in g)
        max_r = max(r for r,c in g)
        min_c = min(c for r,c in g)
        max_c = max(c for r,c in g)
        for r in range(min_r, max_r+1):
            for c in range(min_c, max_c+1):
                if grid[r][c] != 2 and grid[r][c] != 0:
                    result[r][c] = 4
    return result
