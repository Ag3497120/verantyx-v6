import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape
    
    # Create output array initialized with zeros
    output = np.zeros_like(arr)
    
    # Pad array for neighbor counting
    padded = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
    
    for i in range(rows):
        for j in range(cols):
            if arr[i, j] == 8:
                # Get the 3x3 neighborhood (including center)
                neighborhood = padded[i:i+3, j:j+3]
                # Count neighbors (excluding center)
                neighbor_count = np.sum(neighborhood) // 8 - 1
                
                # If exactly 2 or 3 neighbors are 8, output becomes 2
                if neighbor_count == 2 or neighbor_count == 3:
                    output[i, j] = 2
                # Otherwise remains 0 (which is already the default)
    
    return output.tolist()