import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid_np = np.array(grid, dtype=int)
    output = np.zeros_like(grid_np)
    
    # Find all non-zero positions
    non_zero_positions = np.argwhere(grid_np > 0)
    
    if len(non_zero_positions) == 0:
        return output.tolist()
    
    # Find the maximum value in the input
    max_val = np.max(grid_np)
    
    # Find the bottom-rightmost non-zero cell
    # For bottom-right, we want max row + max col (or find the one with largest row+col sum)
    br_position = max(non_zero_positions, key=lambda pos: (pos[0], pos[1]))
    start_row, start_col = br_position
    
    # Create the diagonal pattern
    rows, cols = grid_np.shape
    
    # The pattern consists of 3 consecutive diagonals starting from the bottom-right cell
    # Each diagonal goes from the start position to the bottom-right corner
    for offset in range(3):
        for i in range(rows):
            for j in range(cols):
                # Check if cell is on the diagonal path from start to bottom-right
                # The condition: (i - start_row) == (j - start_col) + offset
                # But we need to ensure it goes toward bottom-right
                if (i - start_row) == (j - start_col) + offset:
                    if i >= start_row and j >= start_col:
                        output[i, j] = max_val
    
    return output.tolist()