def transform(grid):
    import numpy as np
    from collections import deque
    
    g = np.array(grid)
    h, w = g.shape
    
    # Find 2x2 blocks of 2s
    blocks = []
    for r in range(h-1):
        for c in range(w-1):
            if g[r,c]==2 and g[r,c+1]==2 and g[r+1,c]==2 and g[r+1,c+1]==2:
                blocks.append((r,c))
    
    if len(blocks) < 2:
        return [[0]]
    
    block_a, block_b = blocks[0], blocks[1]
    
    def get_block_cells(br, bc):
        return {(br,bc),(br,bc+1),(br+1,bc),(br+1,bc+1)}
    
    def get_adjacent_non_block(br, bc, block_a_cells, block_b_cells):
        """Get cells adjacent (8-connectivity) to block but not in any block"""
        all_block_cells = block_a_cells | block_b_cells
        adj = set()
        for r,c in [(br,bc),(br,bc+1),(br+1,bc),(br+1,bc+1)]:
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr==0 and dc==0: continue
                    nr,nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and (nr,nc) not in all_block_cells:
                        adj.add((nr,nc))
        return adj
    
    block_a_cells = get_block_cells(*block_a)
    block_b_cells = get_block_cells(*block_b)
    all_block_cells = block_a_cells | block_b_cells
    
    adj_a = get_adjacent_non_block(*block_a, block_a_cells, block_b_cells)
    adj_b = get_adjacent_non_block(*block_b, block_a_cells, block_b_cells)
    
    # Find connected component of 8s starting from cells adjacent to Block A
    start_8s = {(r,c) for r,c in adj_a if g[r,c]==8}
    
    if not start_8s:
        return [[0]]
    
    # BFS through 8s
    visited = set(start_8s)
    queue = deque(start_8s)
    
    while queue:
        r, c = queue.popleft()
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr==0 and dc==0: continue
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and g[nr,nc]==8:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
    
    # Check if any cell adjacent to Block B is in the 8-connected component
    adj_b_8s = {(r,c) for r,c in adj_b if g[r,c]==8}
    
    if visited & adj_b_8s:
        return [[8]]
    else:
        return [[0]]
