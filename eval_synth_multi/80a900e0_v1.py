def transform(grid):
    rows, cols = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    bg = {0, 1}
    
    # Find connected components of non-bg cells (8-connected)
    visited = set()
    diamonds = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in bg or (r, c) in visited:
                continue
            comp = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if (cr, cc) in visited: continue
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols: continue
                if grid[cr][cc] in bg: continue
                visited.add((cr, cc))
                comp.append((cr, cc))
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        stack.append((cr+dr, cc+dc))
            diamonds.append(comp)
    
    for comp in diamonds:
        color_counts = {}
        for r, c in comp:
            color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1
        body_color = max(color_counts, key=color_counts.get)
        
        center_r = sum(r for r, c in comp) / len(comp)
        center_c = sum(c for r, c in comp) / len(comp)
        
        tip_colors = set(color_counts.keys()) - {body_color}
        for tip_color in tip_colors:
            cells = [(r, c) for r, c in comp if grid[r][c] == tip_color]
            if len(cells) < 2:
                continue
            
            anti_groups = {}
            main_groups = {}
            for r, c in cells:
                anti_groups.setdefault(r + c, []).append((r, c))
                main_groups.setdefault(r - c, []).append((r, c))
            
            max_anti = max(len(v) for v in anti_groups.values())
            max_main = max(len(v) for v in main_groups.values())
            
            def extend_ray(tip_r, tip_c, dr, dc, color):
                cr, cc = tip_r + dr, tip_c + dc
                while 0 <= cr < rows and 0 <= cc < cols:
                    if out[cr][cc] in bg:
                        out[cr][cc] = color
                    cr += dr
                    cc += dc
            
            if max_main >= max_anti:
                # Edge along main diagonal, extend on anti-diagonals
                anti_vals = sorted(anti_groups.keys())
                extend_vals = {anti_vals[0], anti_vals[-1]} if len(anti_vals) > 1 else set(anti_vals)
                
                for val in extend_vals:
                    tip_on_line = anti_groups.get(val, [])
                    for tip_r, tip_c in tip_on_line:
                        # Extend in direction away from center along anti-diag
                        for dr, dc in [(-1, 1), (1, -1)]:
                            test_r, test_c = tip_r + dr, tip_c + dc
                            d_from = abs(tip_r - center_r) + abs(tip_c - center_c)
                            d_to = abs(test_r - center_r) + abs(test_c - center_c)
                            if d_to > d_from:
                                extend_ray(tip_r, tip_c, dr, dc, tip_color)
                                break
            else:
                # Edge along anti-diagonal, extend on main diagonals
                main_vals = sorted(main_groups.keys())
                extend_vals = {main_vals[0], main_vals[-1]} if len(main_vals) > 1 else set(main_vals)
                
                for val in extend_vals:
                    tip_on_line = main_groups.get(val, [])
                    for tip_r, tip_c in tip_on_line:
                        for dr, dc in [(-1, -1), (1, 1)]:
                            test_r, test_c = tip_r + dr, tip_c + dc
                            d_from = abs(tip_r - center_r) + abs(tip_c - center_c)
                            d_to = abs(test_r - center_r) + abs(test_c - center_c)
                            if d_to > d_from:
                                extend_ray(tip_r, tip_c, dr, dc, tip_color)
                                break
    
    return out
