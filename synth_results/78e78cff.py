def transform(grid):
    return _solve(grid)

def solve_78e78cff(grid):
    g = np.array(grid)
    H, W = g.shape
    out = g.copy()
    
    # Find the unique "seed" cell (not background, not border color)
    values = Counter(g.flatten())
    bg = values.most_common(1)[0][0]
    non_bg = [(r,c) for r in range(H) for c in range(W) if g[r][c] != bg]
    
    # Classify: border color and seed color
    border_colors = set()
    seed_cell = None
    seed_color = None
    
    # Border color = value that appears in lines/patterns (not a singleton)
    color_counts = Counter(g[r][c] for r,c in non_bg)
    # Seed = smallest count (usually 1)
    seed_color = min(color_counts, key=lambda x: (color_counts[x], x))
    seed_positions = [(r,c) for r,c in non_bg if g[r][c] == seed_color]
    border_color = [v for v in color_counts if v != seed_color][0]
    
    # Flood fill from seed position
    if seed_positions:
        sr, sc = seed_positions[0]
        # BFS flood fill blocked by border_color
        visited = set()
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            if (r,c) in visited: continue
            visited.add((r,c))
            out[r][c] = seed_color
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and g[nr][nc] != border_color:
                    q.append((nr, nc))
    
    return out.tolist()


_solve = solve_78e78cff
