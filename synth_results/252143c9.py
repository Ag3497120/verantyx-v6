import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    arr = np.array(grid, dtype=int)
    rows, cols = arr.shape
    result = np.full_like(arr, 7)  # Start with all 7s
    
    # For each cell that is not 7, copy its value to the cell below it
    for r in range(rows - 1):  # Don't process last row since we copy downward
        for c in range(cols):
            if arr[r, c] != 7:
                result[r + 1, c] = arr[r, c]
    
    return result.tolist()