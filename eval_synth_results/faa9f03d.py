
def transform(grid):
    R, C = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    # Find all "2" cells (corner markers)
    twos = [(r,c) for r in range(R) for c in range(C) if grid[r][c]==2]
    if not twos: return out
    
    # The 2s mark corners of rectangle frames
    # For each 2: find what lines connect to it and redraw
    # Strategy: find rectangles (pairs of 2s that form corners)
    # Each rectangle has horizontal sides and vertical sides of different colors
    # The line that connects the 2s draws over/under the crossing line
    
    # Simple approach: find connected pairs of 2s
    # The 2 connects to adjacent colored cells
    # Find the two colors meeting at each 2 corner
    
    # Actually: for each 2, look at which colors are adjacent
    # Then: the rectangle formed by 4 corners has a horizontal color and vertical color
    # At corners: the two colors meet and the "foreground" color should be drawn
    
    # Look at where 2s are connected by lines
    bg = 0
    for r,c in twos:
        # Find what lines connect here
        connected = []
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr,c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc] not in (0, 2):
                connected.append((grid[nr][nc], dr, dc))
        
        if len(connected) >= 2:
            # Take the most common nearby color as dominant
            # Actually: draw through the intersection
            # Replace 2 with the first connected color
            if connected:
                out[r][c] = connected[0][0]
    
    return out
