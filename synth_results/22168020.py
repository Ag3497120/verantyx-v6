import numpy as np
from math import gcd
from functools import reduce

def transform(grid):
    grid = np.array(grid)
    h, w = grid.shape
    
    # Find the smallest repeating pattern width
    for pattern_width in range(1, w + 1):
        if w % pattern_width != 0:
            continue
        
        # Check if this pattern repeats
        num_reps = w // pattern_width
        all_match = True
        
        for rep_idx in range(1, num_reps):
            chunk = grid[:, rep_idx * pattern_width:(rep_idx + 1) * pattern_width]
            first_chunk = grid[:, :pattern_width]
            if not np.array_equal(chunk, first_chunk):
                all_match = False
                break
        
        if all_match:
            return grid[:, :pattern_width].tolist()
    
    return grid.tolist()
