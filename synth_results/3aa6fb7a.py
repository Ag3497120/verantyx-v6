def transform(grid):
    grid = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    
    # Find all 2x2 windows and check for L-shapes
    for i in range(h - 1):
        for j in range(w - 1):
            # Count 8s in this 2x2 window
            eights_positions = []
            for di in range(2):
                for dj in range(2):
                    if grid[i+di][j+dj] == 8:
                        eights_positions.append((di, dj))
            
            # Check if this forms an L (exactly 3 eights)
            if len(eights_positions) == 3:
                # Find the missing position
                all_positions = {(0,0), (0,1), (1,0), (1,1)}
                empty_pos = list(all_positions - set(eights_positions))[0]
                er, ec = empty_pos
                
                # Place a 1 at the empty position if it's currently 0
                if grid[i + er][j + ec] == 0:
                    grid[i + er][j + ec] = 1
    
    return grid
