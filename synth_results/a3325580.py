import numpy as np

def transform(grid):
    g = np.array(grid)
    
    # Find all non-zero shapes and their cell counts
    unique_vals = [v for v in np.unique(g).tolist() if v != 0]
    
    shape_info = {}
    for v in unique_vals:
        cells = np.argwhere(g == v)
        shape_info[v] = {
            'count': len(cells),
            'min_col': int(cells[:, 1].min())
        }
    
    # Find max cell count
    max_count = max(info['count'] for info in shape_info.values())
    
    # Get shapes with max count, sorted by leftmost column
    max_shapes = [(info['min_col'], v) for v, info in shape_info.items() if info['count'] == max_count]
    max_shapes.sort()
    colors = [v for _, v in max_shapes]
    
    # Output: max_count rows, len(colors) cols
    return [[c for c in colors] for _ in range(max_count)]
