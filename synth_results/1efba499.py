import numpy as np
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape
    
    # Create output array
    result = arr.copy()
    
    # Find all plus patterns
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center = arr[i, j]
            if center == 0:
                continue
                
            # Check for plus pattern: center and 4 orthogonal neighbors all same color
            if (arr[i-1, j] == center and arr[i+1, j] == center and 
                arr[i, j-1] == center and arr[i, j+1] == center):
                
                # Get the color from the top neighbor's position
                top_color = arr[i-1, j]
                
                # Replace center with top color
                result[i, j] = top_color
                
                # Replace arms with 0
                result[i-1, j] = 0  # top
                result[i+1, j] = 0  # bottom  
                result[i, j-1] = 0  # left
                result[i, j+1] = 0  # right
    
    return result.tolist()