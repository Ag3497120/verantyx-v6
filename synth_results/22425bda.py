def transform(grid):
    import numpy as np
    from collections import Counter, defaultdict
    
    g = np.array(grid)
    h, w = g.shape
    bg = 7
    
    def line_color(cells, total_len):
        non_bg = [v for v in cells if v != bg]
        if len(non_bg) < 2:
            return None
        c = Counter(non_bg)
        dom, cnt = c.most_common(1)[0]
        if cnt >= total_len * 0.4 and cnt >= 2:
            return dom
        return None
    
    lines = []  # (color, type_priority, path)
    # type_priority: h=0 < v=1 < dm=2 < da=3
    
    for r in range(h):
        col = line_color(g[r].tolist(), w)
        if col is not None:
            lines.append((col, 0, {(r, c) for c in range(w)}))
    
    for c in range(w):
        col = line_color(g[:, c].tolist(), h)
        if col is not None:
            lines.append((col, 1, {(r, c) for r in range(h)}))
    
    for d in range(-(w-1), h):
        path = [(r, r-d) for r in range(h) if 0 <= r-d < w]
        if len(path) < 2:
            continue
        col = line_color([int(g[r, c]) for r, c in path], len(path))
        if col is not None:
            lines.append((col, 2, set(path)))
    
    for s in range(h+w-1):
        path = [(r, s-r) for r in range(h) if 0 <= s-r < w]
        if len(path) < 2:
            continue
        col = line_color([int(g[r, c]) for r, c in path], len(path))
        if col is not None:
            lines.append((col, 3, set(path)))
    
    # Deduplicate (color, type)
    seen = {}
    unique_lines = []
    for line in lines:
        key = (line[0], line[1])
        if key not in seen:
            seen[key] = line
            unique_lines.append(line)
    lines = unique_lines
    
    # For each color, track its type_priority (min, for tiebreaking)
    color_type = {}
    for col, tp, path in lines:
        if col not in color_type or tp < color_type[col]:
            color_type[col] = tp
    
    colors = list(set(l[0] for l in lines))
    if not colors:
        return grid
    
    dominates = defaultdict(set)
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            c1, t1, p1 = lines[i]
            c2, t2, p2 = lines[j]
            if c1 == c2:
                continue
            for r, c in p1 & p2:
                val = int(g[r, c])
                if val == c1:
                    dominates[c1].add(c2)
                elif val == c2:
                    dominates[c2].add(c1)
    
    sorted_colors = []
    remaining = set(colors)
    while remaining:
        def wins_in(c):
            return len(dominates[c] & remaining)
        def dominated_by_in(c):
            return sum(1 for c2 in remaining if c in dominates[c2])
        
        min_wins = min(wins_in(c) for c in remaining)
        candidates = [c for c in remaining if wins_in(c) == min_wins]
        # Sort: most dominated first, then by line type (h < v < dm < da)
        candidates.sort(key=lambda c: (-dominated_by_in(c), color_type.get(c, 99)))
        chosen = candidates[0]
        sorted_colors.append(chosen)
        remaining.remove(chosen)
    
    return [sorted_colors]
