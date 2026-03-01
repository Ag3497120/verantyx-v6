
def transform(grid):
    from collections import Counter
    R, C = len(grid), len(grid[0])
    cnt = Counter(grid[r][c] for r in range(R) for c in range(C))
    bg = cnt.most_common(1)[0][0]
    # Find the template shape: connected region of one color
    # Find markers: 2, 3, 5, 8 scattered individually
    # Template: the largest non-bg connected component
    # Markers: isolated cells of specific colors
    
    # Identify all non-bg colors and their cells
    nz = {}
    for r in range(R):
        for c in range(C):
            v = grid[r][c]
            if v != bg:
                nz.setdefault(v, []).append((r,c))
    
    if not nz: return [row[:] for row in grid]
    
    # The template is the color with the most cells (excluding small isolated ones)
    template_color = max(nz, key=lambda k: len(nz[k]))
    template_cells = nz[template_color]
    
    # Find template bounding box and shape
    min_r = min(r for r,c in template_cells)
    max_r = max(r for r,c in template_cells)
    min_c = min(c for r,c in template_cells)
    max_c = max(c for r,c in template_cells)
    
    # Template shape relative to center
    cr = (min_r + max_r) // 2
    cc = (min_c + max_c) // 2
    template_offsets = [(r-cr, c-cc) for r,c in template_cells]
    
    # Markers: isolated cells (not template color, not bg)
    out = [row[:] for row in grid]
    for color, cells in nz.items():
        if color == template_color: continue
        for r,c in cells:
            # Stamp template at this position with marker's color
            for dr,dc in template_offsets:
                nr,nc = r+dr, c+dc
                if 0<=nr<R and 0<=nc<C:
                    out[nr][nc] = color
    return out
