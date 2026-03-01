def transform(grid):
    rows, cols = len(grid), len(grid[0])
    bg = grid[0][0]
    out = [row[:] for row in grid]
    
    # Find all 2-cells and their connected components (axes)
    two_cells = {(r,c) for r in range(rows) for c in range(cols) if grid[r][c] == 2}
    visited = set()
    axes = []
    for r, c in two_cells:
        if (r, c) in visited: continue
        comp = []
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited: continue
            if (cr, cc) not in two_cells: continue
            visited.add((cr, cc))
            comp.append((cr, cc))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cr+dr, cc+dc))
        axes.append(comp)
    
    # Find all non-bg, non-2 shapes (connected components, 4-connected)
    shape_visited = set()
    shapes = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == bg or grid[r][c] == 2 or (r,c) in shape_visited:
                continue
            color = grid[r][c]
            comp = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if (cr, cc) in shape_visited: continue
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols: continue
                if grid[cr][cc] != color: continue
                shape_visited.add((cr, cc))
                comp.append((cr, cc))
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        if dr == 0 and dc == 0: continue
                        stack.append((cr+dr, cc+dc))
            shapes.append((color, comp))
    
    # Match each shape to an axis (shape must be adjacent to axis, 8-connected)
    for color, shape_cells in shapes:
        shape_set = set(shape_cells)
        best_axis = None
        for axis in axes:
            axis_set = set(axis)
            # Check if shape is 8-connected to this axis
            adjacent = False
            for r, c in shape_cells:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if (r+dr, c+dc) in axis_set:
                            adjacent = True; break
                    if adjacent: break
                if adjacent: break
            if adjacent:
                best_axis = axis
                break
        
        if best_axis is None:
            continue
        
        # Determine axis type and reflect
        axis_rs = [r for r, c in best_axis]
        axis_cs = [c for r, c in best_axis]
        
        if len(best_axis) == 1:
            # Single cell: 180° rotation (point reflection)
            cr, cc = best_axis[0]
            for r, c in shape_cells:
                nr, nc = 2*cr - r, 2*cc - c
                if 0 <= nr < rows and 0 <= nc < cols and out[nr][nc] == bg:
                    out[nr][nc] = color
        elif max(axis_rs) - min(axis_rs) == 0:
            # Horizontal axis: reflect rows
            axis_row = axis_rs[0]
            for r, c in shape_cells:
                nr = 2 * axis_row - r
                if 0 <= nr < rows and out[nr][c] == bg:
                    out[nr][c] = color
        else:
            # Vertical axis: reflect cols
            axis_col = axis_cs[0]
            for r, c in shape_cells:
                nc = 2 * axis_col - c
                if 0 <= nc < cols and out[r][nc] == bg:
                    out[r][nc] = color
    
    return out
