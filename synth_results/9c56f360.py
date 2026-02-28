def transform(grid):
    import numpy as np
    g = [list(row) for row in grid]
    h, w = len(g), len(g[0])
    # Find rows with 3-cells, slide them left until hitting 8 or wall
    for r in range(h):
        row = g[r]
        # Find contiguous groups of 3s in this row
        # Each group slides left to just after the nearest 8 (or wall)
        # Process from left to right, finding 3-groups
        i = 0
        while i < w:
            if row[i] == 3:
                # Find end of 3-group
                j = i
                while j < w and row[j] == 3:
                    j += 1
                group_len = j - i
                # Find the leftmost empty position for this group
                # (after last 8 to the left of i)
                # Clear current positions
                for k in range(i, j):
                    row[k] = 0
                # Find insertion point: slide left to just after an 8 or wall
                # Find the rightmost 8 to the LEFT of position i
                new_pos = 0
                for k in range(i - 1, -1, -1):
                    if row[k] == 8:
                        new_pos = k + 1
                        break
                # Check if there's enough space
                # Insert group at new_pos
                for k in range(group_len):
                    if new_pos + k < w:
                        row[new_pos + k] = 3
                i = j
            else:
                i += 1
        g[r] = row
    return g
