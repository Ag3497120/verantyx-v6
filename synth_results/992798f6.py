def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    pt2 = None; pt1 = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2: pt2 = (r,c)
            elif grid[r][c] == 1: pt1 = (r,c)
    
    if pt2 is None or pt1 is None:
        return grid
    
    r2,c2 = pt2; r1,c1 = pt1
    dr = abs(r1-r2); dc = abs(c1-c2)
    row_dir = 1 if r1>r2 else -1
    col_dir = 1 if c1>c2 else -1
    
    result = [row[:] for row in grid]
    
    # Path: 1 diagonal, then straight (major axis - minor axis) steps, then (minor-2) diagonal
    # major = max(dr,dc), minor = min(dr,dc)
    major = max(dr,dc); minor = min(dr,dc)
    major_dir = (row_dir, 0) if dr >= dc else (0, col_dir)
    diag_dir = (row_dir, col_dir)
    
    path = []
    r, c = r2, c2
    
    # 1 diagonal step first
    r += diag_dir[0]; c += diag_dir[1]
    path.append((r,c))
    
    # (major - minor) straight steps
    for _ in range(major - minor):
        r += major_dir[0]; c += major_dir[1]
        path.append((r,c))
    
    # (minor - 2) diagonal steps (all but the last which is the endpoint)
    for _ in range(minor - 2):
        r += diag_dir[0]; c += diag_dir[1]
        path.append((r,c))
    
    for pr, pc in path:
        if (pr, pc) != pt1 and (pr, pc) != pt2:
            result[pr][pc] = 3
    
    return result
