def transform(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    
    # Fill borders
    for i in range(h):
        out[i][0] = 3
        out[i][w-1] = 3
    for j in range(w):
        out[0][j] = 3
        out[h-1][j] = 3
    
    # Spiral drawing
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # left, down, right, up
    dir_idx = 0
    step_length = w - 2  # initial horizontal segment length (top row minus corners)
    r, c = 0, w - 2  # start position (top row, second from right)
    
    while step_length > 0:
        dr, dc = directions[dir_idx]
        for _ in range(step_length):
            out[r][c] = 3
            r += dr
            c += dc
        # Move back one step (we overshot)
        r -= dr
        c -= dc
        # Turn
        dir_idx = (dir_idx + 1) % 4
        # Adjust step length: after every two turns, decrease by 2
        if dir_idx % 2 == 0:
            step_length -= 2
        # Move to next starting position
        dr, dc = directions[dir_idx]
        r += dr
        c += dc
    
    return out