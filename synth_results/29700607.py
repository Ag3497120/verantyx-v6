def transform(grid):
    import numpy as np
    g = np.array(grid)
    rows, cols = g.shape
    result = g.copy()
    
    from collections import defaultdict
    color_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if g[r, c] != 0:
                color_cells[g[r, c]].append((r, c))
    
    for color, cells in color_cells.items():
        if len(cells) == 1:
            # Single cell: extend to opposite edge
            r, c = cells[0]
            if r == 0:
                # Top edge → extend down entire column
                for nr in range(rows):
                    result[nr, c] = color
            elif r == rows - 1:
                # Bottom edge → extend up entire column
                for nr in range(rows):
                    result[nr, c] = color
            elif c == 0:
                # Left edge → extend right entire row
                for nc in range(cols):
                    result[r, nc] = color
            elif c == cols - 1:
                # Right edge → extend left entire row
                for nc in range(cols):
                    result[r, nc] = color
        elif len(cells) == 2:
            r1, c1 = cells[0]
            r2, c2 = cells[1]
            if r1 == r2:
                # Same row: fill between
                for c in range(min(c1, c2), max(c1, c2)+1):
                    result[r1, c] = color
            elif c1 == c2:
                # Same col: fill between
                for r in range(min(r1, r2), max(r1, r2)+1):
                    result[r, c1] = color
            else:
                # L-shape: vertical from cell1 col to cell2 row, horizontal from cell2 to intersection
                # Intersection is at (r2, c1)
                for r in range(min(r1, r2), max(r1, r2)+1):
                    result[r, c1] = color
                for c in range(min(c1, c2), max(c1, c2)+1):
                    result[r2, c] = color
    
    return result.tolist()
