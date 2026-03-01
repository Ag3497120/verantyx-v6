from collections import Counter, deque

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1
    
    sorted_colors = color_counts.most_common()
    bg_color = sorted_colors[0][0]
    wall_color = sorted_colors[1][0]
    
    visited = [[False]*cols for _ in range(rows)]
    components = []
    
    def flood_fill(sr, sc):
        stack = [(sr, sc)]
        cells = []
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] == wall_color:
                continue
            visited[r][c] = True
            cells.append((r, c))
            stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])
        return cells
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != wall_color:
                comp = flood_fill(r, c)
                if comp:
                    components.append(comp)
    
    output = [row[:] for row in grid]
    DIRS8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    for comp in components:
        comp_set = set(comp)
        
        depth_map = {}
        queue = deque()
        
        for r, c in comp:
            is_boundary = False
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                is_boundary = True
            else:
                for dr, dc in DIRS8:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) not in comp_set:
                        is_boundary = True
                        break
            if is_boundary:
                depth_map[(r,c)] = 0
                queue.append((r, c))
        
        while queue:
            r, c = queue.popleft()
            d = depth_map[(r,c)]
            for dr, dc in DIRS8:
                nr, nc = r+dr, c+dc
                if (nr, nc) in comp_set and (nr, nc) not in depth_map:
                    depth_map[(nr, nc)] = d + 1
                    queue.append((nr, nc))
        
        seeds = {}
        for r, c in comp:
            if grid[r][c] != bg_color:
                d = depth_map[(r,c)]
                seeds[d] = grid[r][c]
        
        if not seeds:
            continue
        
        max_depth = max(seeds.keys())
        color_seq = []
        for d in range(max_depth + 1):
            color_seq.append(seeds.get(d, bg_color))
        
        seq_len = len(color_seq)
        for r, c in comp:
            d = depth_map[(r,c)]
            output[r][c] = color_seq[d % seq_len]
    
    return output
