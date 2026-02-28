from collections import Counter

def transform(grid):
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    # Keep the top 2 most frequent values, replace others with 7
    top2 = set(v for v, _ in counts.most_common(2))
    return [[v if v in top2 else 7 for v in row] for row in grid]
