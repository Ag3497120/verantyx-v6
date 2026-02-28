def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find the axis lines (rows/cols that are all the same value = axis marker)
    # The grid is divided into quadrants by these axes
    # The shape in one quadrant is reflected to fill all quadrants
    
    # Find axis row and col (the row/col where all cells = same non-zero value)
    axis_row = None
    axis_col = None
    axis_val = None
    
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            axis_row = r
            axis_val = grid[r][0]
            break
    
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1 and col_vals[0] != 0:
            axis_col = c
            break
    
    if axis_row is None or axis_col is None:
        return [list(row) for row in grid]
    
    # Extract the non-axis content (the shape) from the quadrant that has it
    # Find which quadrant contains the shape
    shape = {}
    for r in range(rows):
        for c in range(cols):
            if r != axis_row and c != axis_col and grid[r][c] != 0:
                shape[(r, c)] = grid[r][c]
    
    if not shape:
        return [list(row) for row in grid]
    
    # Replace the shape value with the axis value in all reflections
    shape_val = list(shape.values())[0]
    result = [[0]*cols for _ in range(rows)]
    
    # Copy axis lines
    for c in range(cols):
        result[axis_row][c] = axis_val
    for r in range(rows):
        result[r][axis_col] = axis_val
    
    # Get shape cells' relative positions to the axis
    for (r, c), v in shape.items():
        dr = r - axis_row
        dc = c - axis_col
        # Place in all 4 quadrants
        for sr in [1, -1]:
            for sc in [1, -1]:
                nr = axis_row + sr * abs(dr)
                nc = axis_col + sc * abs(dc)
                if 0 <= nr < rows and 0 <= nc < cols and nr != axis_row and nc != axis_col:
                    result[nr][nc] = shape_val
    
    return result
