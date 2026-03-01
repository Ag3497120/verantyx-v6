
def transform(grid):
    rows = len(grid)
    cols = len(grid[0])
    n_blocks = rows // 3
    blocks = []
    for i in range(n_blocks):
        block = [grid[3*i+r] for r in range(3)]
        # find color of this block
        color = 0
        for row in block:
            for v in row:
                if v != 0:
                    color = v
                    break
            if color: break
        blocks.append((color, block))
    # priority: determine by counting non-zero cells (more = higher priority)
    blocks_sorted = sorted(blocks, key=lambda x: sum(v!=0 for row in x[1] for v in row), reverse=True)
    # but empirically priority is 5>4>8>2, so use that if colors match
    priority_order = [5, 4, 8, 2]
    def get_priority(color):
        try:
            return priority_order.index(color)
        except:
            return 99
    blocks_sorted = sorted(blocks, key=lambda x: get_priority(x[0]))
    result = []
    for r in range(3):
        row = []
        for c in range(cols):
            val = 0
            for color, block in blocks_sorted:
                if block[r][c] != 0:
                    val = block[r][c]
                    break
            row.append(val)
        result.append(row)
    return result
