import math

def transform(grid):
    H = len(grid)
    W = len(grid[0])
    
    # Find color and count of non-zero cells in first row
    row0 = grid[0]
    color = next(v for v in row0 if v != 0)
    N = sum(1 for v in row0 if v != 0)
    
    # Generate sequence: groups [1,2,...,N,...,2,1] each followed by 0
    seq = []
    groups = list(range(1, N+1)) + list(range(N-1, 0, -1))
    for g in groups:
        seq.extend([color] * g)
        seq.append(0)
    
    # Compute output height
    total = len(seq)
    H_out = math.ceil(total / W)
    
    # Pad with 0s
    while len(seq) < H_out * W:
        seq.append(0)
    
    # Reshape into H_out x W
    out = []
    for r in range(H_out):
        out.append(seq[r*W:(r+1)*W])
    
    return out
