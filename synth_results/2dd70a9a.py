def transform(grid):
    import numpy as np
    from collections import deque
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    three_cells = set(map(tuple, np.argwhere(g == 3)))
    two_cells = set(map(tuple, np.argwhere(g == 2)))
    
    if not three_cells or not two_cells:
        return grid
    
    # BFS shortest path from any cell adjacent to 3-cluster
    # to any cell adjacent to 2-cluster
    # Moving only through 0 cells (not 8 or other values)
    # Tracking direction changes (at most 2)
    
    def get_adjacent(cells, avoid):
        adj = []
        for r, c in cells:
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if (0 <= nr < rows and 0 <= nc < cols and 
                    (nr, nc) not in avoid and g[nr, nc] == 0):
                    adj.append((nr, nc))
        return adj
    
    src_adj = get_adjacent(three_cells, three_cells | two_cells)
    
    # State: (r, c, direction, turns)
    visited = {}
    queue = deque()
    
    for r, c in src_adj:
        queue.append((r, c, -1, 0, [(r, c)]))
        visited[(r, c, -1, 0)] = True
    
    found_path = None
    
    while queue and found_path is None:
        r, c, direction, turns, path = queue.popleft()
        
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            
            # Check if reached adjacent to target
            if (nr, nc) in two_cells:
                found_path = path
                break
            
            if (nr, nc) in three_cells:
                continue
            if g[nr, nc] != 0:  # can only move through empty cells
                continue
            
            new_dir = 0 if dr != 0 else 1
            new_turns = turns + (1 if direction != -1 and new_dir != direction else 0)
            if new_turns > 2:
                continue
            
            key = (nr, nc, new_dir, new_turns)
            if key not in visited:
                visited[key] = True
                queue.append((nr, nc, new_dir, new_turns, path + [(nr, nc)]))
        
        if found_path:
            break
    
    if found_path:
        for r, c in found_path:
            result[r, c] = 3
    
    return result.tolist()
