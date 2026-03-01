import numpy as np

def transform(grid):
    g = np.array(grid)
    H, W = g.shape
    bg = max(set(g.flatten()), key=lambda c: np.sum(g == c))
    
    # Find all non-bg cells
    non_bg = list(zip(*np.where(g != bg)))
    
    # Group by spatial proximity using connected components with 8-connectivity
    visited = set()
    components = []
    
    def bfs(start):
        queue = [start]
        visited.add(start)
        cells = [start]
        while queue:
            r, c = queue.pop(0)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r+dr, c+dc
                    if (nr, nc) not in visited and 0 <= nr < H and 0 <= nc < W and g[nr, nc] != bg:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        cells.append((nr, nc))
        return cells
    
    for r, c in non_bg:
        if (r, c) not in visited:
            cells = bfs((r, c))
            components.append(cells)
    
    # Extract pattern for each component
    def extract_pattern(cells):
        rows = [c[0] for c in cells]
        cols = [c[1] for c in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        pat = np.full((r1-r0+1, c1-c0+1), bg, dtype=int)
        for r, c in cells:
            pat[r-r0, c-c0] = g[r, c]
        colors = set(int(g[r, c]) for r, c in cells)
        return pat, colors
    
    patterns = [extract_pattern(c) for c in components]
    
    # Template has single non-bg color; stamp has multiple
    template = None
    stamp = None
    for pat, colors in patterns:
        non_bg_colors = colors - {int(bg)}
        if len(non_bg_colors) <= 1:
            if template is None or pat.size > (template.size if isinstance(template, np.ndarray) else 0):
                template = pat
        else:
            stamp = pat
    
    if stamp is None:
        # Both might be single-color; larger is template
        patterns.sort(key=lambda x: x[0].size)
        stamp = patterns[0][0]
        template = patterns[-1][0]
    
    # Build output
    th, tw = template.shape
    sh, sw = stamp.shape
    out = np.full((th * sh, tw * sw), bg, dtype=int)
    
    for tr in range(th):
        for tc in range(tw):
            if template[tr, tc] != bg:
                out[tr*sh:(tr+1)*sh, tc*sw:(tc+1)*sw] = stamp
    
    return out.tolist()
