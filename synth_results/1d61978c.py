import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    grid = np.array(grid, dtype=int)
    output = grid.copy()
    
    rows, cols = grid.shape
    
    # Process each diagonal (top-left to bottom-right)
    # Diagonals are identified by (row - col) being constant
    for d in range(-rows + 1, cols):
        # Find all positions with value 5 in this diagonal
        positions = []
        for r in range(rows):
            c = r - d
            if 0 <= c < cols and grid[r, c] == 5:
                positions.append((r, c))
        
        # Sort by row (or column) to get order along diagonal
        positions.sort()  # sorts by row then column
        
        # Replace first 5 with 2, others with 8
        for i, (r, c) in enumerate(positions):
            output[r, c] = 2 if i == 0 else 8
    
    return output.tolist()