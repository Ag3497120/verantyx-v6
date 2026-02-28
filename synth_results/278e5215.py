import numpy as np
from typing import List

def transform(grid: List[List[int]]) -> List[List[int]]:
    arr = np.array(grid, dtype=int)
    
    # Find the digit shape region (made of 5's and 0's)
    digit_mask = np.zeros_like(arr, dtype=bool)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 5:
                # Flood fill to find connected region of 5's
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1] and arr[x, y] == 5 and not digit_mask[x, y]:
                        digit_mask[x, y] = True
                        stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
                break
        if digit_mask.any():
            break
    
    # Find the palette region (non-zero, not 5, not in digit region)
    palette_mask = (arr != 0) & (arr != 5) & (~digit_mask)
    
    if not palette_mask.any():
        return []
    
    # Get bounding box of palette
    palette_rows = np.where(palette_mask.any(axis=1))[0]
    palette_cols = np.where(palette_mask.any(axis=0))[0]
    
    if len(palette_rows) == 0 or len(palette_cols) == 0:
        return []
    
    r1, r2 = palette_rows[0], palette_rows[-1]
    c1, c2 = palette_cols[0], palette_cols[-1]
    
    palette_region = arr[r1:r2+1, c1:c2+1]
    
    # Get background color (color surrounding palette)
    # Look for non-zero color just outside palette region
    bg_color = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not (r1 <= i <= r2 and c1 <= j <= c2) and arr[i, j] != 0 and arr[i, j] != 5:
                bg_color = arr[i, j]
                break
        if bg_color != 0:
            break
    
    # Get bounding box of digit shape
    digit_rows = np.where(digit_mask.any(axis=1))[0]
    digit_cols = np.where(digit_mask.any(axis=0))[0]
    
    if len(digit_rows) == 0 or len(digit_cols) == 0:
        return []
    
    # Extract digit shape region
    digit_region = arr[digit_rows[0]:digit_rows[-1]+1, digit_cols[0]:digit_cols[-1]+1]
    digit_mask_region = digit_mask[digit_rows[0]:digit_rows[-1]+1, digit_cols[0]:digit_cols[-1]+1]
    
    # Create output by mapping digit shape to palette
    output = np.zeros_like(digit_region)
    
    for i in range(digit_region.shape[0]):
        for j in range(digit_region.shape[1]):
            if digit_mask_region[i, j]:
                # Map position in digit region to position in palette
                palette_i = i % palette_region.shape[0]
                palette_j = j % palette_region.shape[1]
                output[i, j] = palette_region[palette_i, palette_j]
            else:
                output[i, j] = bg_color
    
    return output.tolist()