def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    result = [list(row) for row in grid]
    
    # Find all 3s
    threes = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 3]
    
    for (pr, pc) in threes:
        # Draw template around (pr, pc):
        # Row pr-2: 5s at pc-2 to pc+2
        # Row pr-1: 2 at pc-2, 5 at pc, 2 at pc+2
        # Row pr: 2 at pc-2, keep 3 at pc, 2 at pc+2
        # Row pr+1: 2 at pc-2, 2 at pc+2
        # Row pr+2: entire row = 2 except 8s at pc-2 to pc+2
        
        # Top bar
        r = pr - 2
        if 0 <= r < rows:
            for c in range(pc-2, pc+3):
                if 0 <= c < cols:
                    result[r][c] = 5
        
        # Second row
        r = pr - 1
        if 0 <= r < rows:
            for c in range(pc-2, pc+3):
                if 0 <= c < cols:
                    if c == pc:
                        result[r][c] = 5
                    elif c == pc-2 or c == pc+2:
                        result[r][c] = 2
        
        # 3's row: sides
        r = pr
        for c in [pc-2, pc+2]:
            if 0 <= c < cols:
                result[r][c] = 2
        
        # Row below
        r = pr + 1
        if 0 <= r < rows:
            for c in [pc-2, pc+2]:
                if 0 <= c < cols:
                    result[r][c] = 2
        
        # Bottom bar (row pr+2): entire row is 2, with 8s at pc-2 to pc+2
        r = pr + 2
        if 0 <= r < rows:
            # Fill entire row with 2
            for c in range(cols):
                result[r][c] = 2
            # Place 8s at pc-2 to pc+2
            for c in range(pc-2, pc+3):
                if 0 <= c < cols:
                    result[r][c] = 8
    
    return result
