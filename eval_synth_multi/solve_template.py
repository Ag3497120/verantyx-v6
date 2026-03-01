import json, sys
import numpy as np

def analyze_and_solve(task_path):
    with open(task_path) as f:
        task = json.load(f)
    
    for idx, ex in enumerate(task['train']):
        grid = np.array(ex['input'])
        expected = np.array(ex['output'])
        
        # Find 8-rectangle
        eights = np.argwhere(grid == 8)
        if len(eights) == 0:
            print(f"Train {idx}: No 8s found")
            continue
        r_min, c_min = eights.min(axis=0)
        r_max, c_max = eights.max(axis=0)
        
        H, W = grid.shape
        print(f"Train {idx}: grid {H}x{W}, 8s at rows {r_min}-{r_max}, cols {c_min}-{c_max}")
        print(f"  Expected output: {expected.shape}")
        
        # Try different symmetry axes
        for R in range(H+1, H+H):
            # Check row symmetry: r -> R-r
            valid = True
            for r in range(r_min, r_max+1):
                mr = R - r
                if mr < 0 or mr >= H:
                    valid = False
                    break
                if np.any(grid[mr, c_min:c_max+1] == 8):
                    valid = False  # mirror also has 8s, can't use row symmetry alone
                    break
            if valid:
                result = grid[np.array([R-r for r in range(r_min, r_max+1)]), c_min:c_max+1]
                if np.array_equal(result, expected):
                    print(f"  ROW symmetry R={R} works!")
                    break
        
        for C in range(W+1, W+W):
            # Check col symmetry: c -> C-c
            valid = True
            for c in range(c_min, c_max+1):
                mc = C - c
                if mc < 0 or mc >= W:
                    valid = False
                    break
                if np.any(grid[r_min:r_max+1, mc] == 8):
                    valid = False
                    break
            if valid:
                result = grid[r_min:r_max+1, np.array([C-c for c in range(c_min, c_max+1)])]
                if np.array_equal(result, expected):
                    print(f"  COL symmetry C={C} works!")
                    break

analyze_and_solve(sys.argv[1])
