def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    # Find 2-lines (rows or cols where mostly 2s)
    h_lines = [r for r in range(rows) if sum(1 for c in range(cols) if grid[r][c]==2) > cols//2]
    v_lines = [c for c in range(cols) if sum(1 for r in range(rows) if grid[r][c]==2) > rows//2]
    
    # Find boxes (3x3 bordered with 2 at center)
    boxes = []
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r][c] == 2:
                border = [(r-1,c-1),(r-1,c),(r-1,c+1),(r,c-1),(r,c+1),(r+1,c-1),(r+1,c),(r+1,c+1)]
                if all(0<=br<rows and 0<=bc<cols and grid[br][bc]==3 for br,bc in border):
                    boxes.append((r, c))
    
    result = [row[:] for row in grid]
    
    def place_shadow(cr, cc):
        """Place a 3x3 shadow (1s for border, 2 for center) at center (cr,cc)"""
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                r2, c2 = cr+dr, cc+dc
                if 0<=r2<rows and 0<=c2<cols:
                    if dr==0 and dc==0:
                        pass  # center stays as 2 (already is)
                    elif grid[r2][c2] not in (3,):  # don't overwrite box 3s
                        if result[r2][c2] != 3:
                            result[r2][c2] = 1
    
    for br, bc in boxes:
        # Determine direction: perpendicular to the 2-line
        on_h_line = br in h_lines
        on_v_line = bc in v_lines
        
        if on_h_line:
            # Extend VERTICALLY (up and down)
            for direction in [-1, 1]:
                r = br + direction
                while 0 <= r < rows:
                    if r in h_lines:
                        # Crossing another horizontal 2-line: place shadow
                        place_shadow(r, bc)
                        r += direction  # continue after shadow
                    elif grid[r][bc] == 3:
                        r += direction  # skip box border
                    elif grid[r][bc] == 0:
                        result[r][bc] = 1
                        r += direction
                    else:
                        r += direction
        
        if on_v_line:
            # Extend HORIZONTALLY (left and right)
            for direction in [-1, 1]:
                c = bc + direction
                while 0 <= c < cols:
                    if c in v_lines:
                        place_shadow(br, c)
                        c += direction
                    elif grid[br][c] == 3:
                        c += direction
                    elif grid[br][c] == 0:
                        result[br][c] = 1
                        c += direction
                    else:
                        c += direction
    
    return result
