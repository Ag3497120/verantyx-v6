def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background value (most frequent number)
    freq = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            freq[val] = freq.get(val, 0) + 1
    background = max(freq.items(), key=lambda x: x[1])[0]
    
    # Helper for BFS/DFS
    visited = [[False] * cols for _ in range(rows)]
    components = []  # each component: (value, list of (r, c))
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != background:
                val = grid[r][c]
                stack = [(r, c)]
                visited[r][c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == val:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((val, cells))
    
    # Sort components by reading order of their first cell
    components.sort(key=lambda comp: (comp[1][0][0], comp[1][0][1]))
    
    # Build output
    output = []
    for val, cells in components:
        for _ in cells:
            output.append([val])
    return output