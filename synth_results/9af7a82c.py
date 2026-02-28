def transform(grid):
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    # Sort by count descending
    sorted_vals = sorted(counts.items(), key=lambda x: -x[1])
    
    max_count = sorted_vals[0][1]
    num_cols = len(sorted_vals)
    
    result = []
    for row_idx in range(max_count):
        row = []
        for val, cnt in sorted_vals:
            if row_idx < cnt:
                row.append(val)
            else:
                row.append(0)
        result.append(row)
    return result
