import numpy as np

def transform(grid):
    """Replace markers with color of nearest border."""
    grid = np.array(grid)
    result = grid.copy()
    h, w = grid.shape
    
    # Find border colors
    top_color = grid[0, 0] if len(set(grid[0])) == 1 and grid[0, 0] != 0 else None
    bottom_color = grid[h-1, 0] if len(set(grid[h-1])) == 1 and grid[h-1, 0] != 0 else None
    left_color = grid[0, 0] if len(set(grid[:, 0])) == 1 and grid[0, 0] != 0 else None
    right_color = grid[0, w-1] if len(set(grid[:, w-1])) == 1 and grid[0, w-1] != 0 else None
    
    # Find all 3s and replace with nearest border color
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 3:
                # Calculate distances to borders
                dist_top = r
                dist_bottom = h - 1 - r
                dist_left = c
                dist_right = w - 1 - c
                
                # Find minimum distance and corresponding color
                min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
                
                if min_dist == dist_top and top_color is not None:
                    result[r, c] = top_color
                elif min_dist == dist_bottom and bottom_color is not None:
                    result[r, c] = bottom_color
                elif min_dist == dist_left and left_color is not None:
                    result[r, c] = left_color
                elif min_dist == dist_right and right_color is not None:
                    result[r, c] = right_color
    
    return result.tolist()
