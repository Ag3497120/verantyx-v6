def transform(grid):
    h, w = len(grid), len(grid[0])
    
    # Find all 3x3 patterns that use only one color (plus zeros)
    patterns = []
    for i in range(h - 2):
        for j in range(w - 2):
            pattern = tuple(tuple(grid[i+di][j+dj] for dj in range(3)) for di in range(3))
            colors = set(grid[i+di][j+dj] for di in range(3) for dj in range(3) if grid[i+di][j+dj] != 0)
            # Only patterns with exactly one non-zero color and 3-6 cells of that color
            non_zero_count = sum(1 for di in range(3) for dj in range(3) if grid[i+di][j+dj] != 0)
            if len(colors) == 1 and 3 <= non_zero_count <= 6:
                patterns.append(pattern)
    
    if not patterns:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Find most common pattern
    from collections import Counter
    pattern_counts = Counter(patterns)
    most_common = pattern_counts.most_common(1)[0][0]
    
    return [list(row) for row in most_common]
