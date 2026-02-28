def transform(grid):
    return _solve(grid)

def solve_762cd429(grid):
    g = [list(r) for r in grid]
    H, W = len(g), len(g[0])
    out = [row[:] for row in g]
    
    # Find the 2x2 seed (usually at rows r, r+1, cols 0, 1)
    seed = None
    seed_r, seed_c = None, None
    for r in range(H-1):
        for c in range(W-1):
            if any(grid[r+dr][c+dc] != 0 for dr in range(2) for dc in range(2)):
                if g[r][c] != 0 or g[r][c+1] != 0 or g[r+1][c] != 0 or g[r+1][c+1] != 0:
                    # Check it's the only non-zero 2x2 near left edge
                    if c < 3:
                        seed = [[g[r][c], g[r][c+1]], [g[r+1][c], g[r+1][c+1]]]
                        seed_r, seed_c = r, c
                        break
        if seed: break
    
    if not seed: return grid
    
    # Generate expansions
    center_r = seed_r  # top of seed
    start_col = seed_c + 2  # next scale starts after seed
    scale = 2  # each cell -> scale x scale block
    
    while start_col < W:
        # This level's block: size = scale x scale, placed at rows and cols
        block_size = scale
        start_r = center_r - block_size // 2 + 1  # center around seed
        
        for i in range(2):
            for j in range(2):
                cell_val = seed[i][j]
                # Fill block_size//2 x block_size//2 area
                half = block_size // 2
                r0 = start_r + i * half
                c0 = start_col + j * half
                for dr in range(half):
                    for dc in range(half):
                        nr, nc = r0+dr, c0+dc
                        if 0<=nr<H and 0<=nc<W:
                            out[nr][nc] = cell_val
        
        start_col += block_size
        scale *= 2
    
    return out


_solve = solve_762cd429
