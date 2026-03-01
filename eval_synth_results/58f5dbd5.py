import json

with open('/private/tmp/arc-agi-2/data/evaluation/58f5dbd5.json') as f:
    data = json.load(f)

def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find background color (most common)
    from collections import Counter
    flat = [c for row in grid for c in row]
    bg = Counter(flat).most_common(1)[0][0]
    
    # Find 5x5 solid non-bg blocks
    rects = []  # (r, c, color)
    visited_rect = set()
    for r in range(rows - 4):
        for c in range(cols - 4):
            color = grid[r][c]
            if color == bg:
                continue
            # Check if 5x5 block of same color
            is_solid = True
            for dr in range(5):
                for dc in range(5):
                    if grid[r+dr][c+dc] != color:
                        is_solid = False
                        break
                if not is_solid:
                    break
            if is_solid:
                key = (r, c)
                if key not in visited_rect:
                    rects.append((r, c, color))
                    visited_rect.add(key)
    
    # Deduplicate overlapping rects - keep only those at top-left corners
    # A 5x5 solid block at (r,c) means (r,c) through (r+4,c+4) are all same color
    # We want to find the actual rect positions
    # Filter: only keep if bordered by bg on top-left
    final_rects = []
    rect_cells = set()
    for r, c, color in rects:
        # Check if this is the top-left corner (bordered by bg above and left)
        top_ok = (r == 0 or all(grid[r-1][c+dc] == bg for dc in range(5)))
        left_ok = (c == 0 or all(grid[r+dr][c-1] == bg for dr in range(5)))
        if top_ok and left_ok:
            final_rects.append((r, c, color))
            for dr in range(5):
                for dc in range(5):
                    rect_cells.add((r+dr, c+dc))
    
    # Find 3x3 patterns (non-bg cells not in rects)
    # Look for 3x3 bounding boxes of non-bg, non-rect cells
    patterns = {}  # color -> 3x3 grid
    pattern_visited = set()
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r,c) not in rect_cells and (r,c) not in pattern_visited:
                # Found a non-bg cell outside rects - find its color
                color = grid[r][c]
                if color in patterns:
                    # Could be same color pattern, skip if already found
                    pattern_visited.add((r,c))
                    continue
                
                # BFS to find all cells of this cluster
                from collections import deque
                queue = deque([(r, c)])
                cluster = set()
                cluster.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr,nc) not in cluster and (nr,nc) not in rect_cells:
                            if grid[nr][nc] != bg:
                                cluster.add((nr, nc))
                                queue.append((nr, nc))
                
                # Find bounding box
                min_r = min(cr for cr, cc in cluster)
                max_r = max(cr for cr, cc in cluster)
                min_c = min(cc for cr, cc in cluster)
                max_c = max(cc for cr, cc in cluster)
                
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                
                if h == 3 and w == 3:
                    # Extract 3x3 pattern
                    pat = []
                    for pr in range(min_r, max_r + 1):
                        row = []
                        for pc in range(min_c, max_c + 1):
                            row.append(grid[pr][pc])
                        pat.append(row)
                    
                    # Determine pattern color (non-bg color in the pattern)
                    pat_colors = set()
                    for pr in pat:
                        for pc in pr:
                            if pc != bg:
                                pat_colors.add(pc)
                    
                    if len(pat_colors) == 1:
                        pcolor = pat_colors.pop()
                        if pcolor not in patterns:
                            patterns[pcolor] = pat
                
                for cell in cluster:
                    pattern_visited.add(cell)
    
    # Determine rect grid arrangement
    # Sort rects by row then col
    final_rects.sort(key=lambda x: (x[0], x[1]))
    
    # Find unique row and col positions
    rect_rows = sorted(set(r for r, c, color in final_rects))
    rect_cols = sorted(set(c for r, c, color in final_rects))
    
    R = len(rect_rows)
    C = len(rect_cols)
    
    # Build output grid
    out_h = R * 6 + 1
    out_w = C * 6 + 1
    output = [[bg] * out_w for _ in range(out_h)]
    
    # Place each rect
    rect_map = {(r, c): color for r, c, color in final_rects}
    
    for ri, rr in enumerate(rect_rows):
        for ci, cc in enumerate(rect_cols):
            if (rr, cc) not in rect_map:
                continue
            color = rect_map[(rr, cc)]
            
            # Output position for this rect
            out_r = 1 + ri * 6
            out_c = 1 + ci * 6
            
            # Fill 5x5 with color
            for dr in range(5):
                for dc in range(5):
                    output[out_r + dr][out_c + dc] = color
            
            # If we have a matching pattern, embed it inverted in interior
            if color in patterns:
                pat = patterns[color]
                for dr in range(3):
                    for dc in range(3):
                        # Swap: color <-> bg
                        if pat[dr][dc] == color:
                            output[out_r + 1 + dr][out_c + 1 + dc] = bg
                        else:
                            output[out_r + 1 + dr][out_c + 1 + dc] = color
    
    return output

# Test on all training examples
for i, example in enumerate(data['train']):
    result = transform(example['input'])
    expected = example['output']
    if result == expected:
        print(f"Train {i}: PASS")
    else:
        print(f"Train {i}: FAIL")
        print(f"  Expected rows: {len(expected)}, cols: {len(expected[0])}")
        print(f"  Got rows: {len(result)}, cols: {len(result[0])}")
        for r in range(min(len(result), len(expected))):
            if r < len(result) and r < len(expected) and result[r] != expected[r]:
                print(f"  Row {r} diff: got {result[r]}")
                print(f"           exp {expected[r]}")
