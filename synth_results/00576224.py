def transform(grid):
    """
    Pattern: Tile the input grid 3x horizontally, then create alternating normal/flipped blocks vertically.
    For a 2x2 input, creates 6x6 output: 2 rows normal, 2 rows flipped, 2 rows normal.
    """
    result = []
    
    # First block: normal rows, each repeated 3 times horizontally
    for row in grid:
        result.append(row * 3)
    
    # Second block: flipped rows, each repeated 3 times horizontally
    for row in grid:
        result.append(row[::-1] * 3)
    
    # Third block: normal rows again
    for row in grid:
        result.append(row * 3)
    
    return result
