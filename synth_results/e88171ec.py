def transform(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Find the rectangular region of 0s that's fully surrounded by 2s
    # Strategy: find the largest contiguous all-zero rectangle
    # Using largest rectangle in histogram
    def largest_rect_of_zeros(grid, rows, cols):
        # Use stack-based approach to find largest all-0 rectangle
        heights = [0] * cols
        best = (0, 0, 0, 0, 0)  # area, r0, c0, r1, c1
        for r in range(rows):
            for c in range(cols):
                heights[c] = heights[c] + 1 if grid[r][c] == 0 else 0
            # Largest rect in heights
            stack = []
            for c in range(cols + 1):
                h = heights[c] if c < cols else 0
                start = c
                while stack and stack[-1][1] > h:
                    c2, h2 = stack.pop()
                    area = h2 * (c - c2)
                    if area > best[0]:
                        best = (area, r - h2 + 1, c2, r, c - 1)
                    start = c2
                stack.append((start, h))
        return best
    
    area, r0, c0, r1, c1 = largest_rect_of_zeros(grid, rows, cols)
    if area == 0: return grid
    
    result = [row[:] for row in grid]
    for r in range(r0, r1+1):
        for c in range(c0, c1+1):
            result[r][c] = 8
    return result
