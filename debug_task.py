#!/usr/bin/env python3
"""Debug specific task to understand pattern"""

import json
import numpy as np
from pathlib import Path

def show_task(task_id):
    task_path = Path(f"/tmp/arc-agi-2/data/training/{task_id}.json")
    with open(task_path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    for idx, pair in enumerate(data["train"]):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])

        print(f"\nPair {idx}:")
        print(f"Input ({inp.shape}):")
        print(inp)
        print(f"\nOutput ({out.shape}):")
        print(out)

        # Analysis
        ih, iw = inp.shape
        oh, ow = out.shape

        if oh % ih == 0 and ow % iw == 0:
            scale_h = oh // ih
            scale_w = ow // iw
            print(f"\nScale: {scale_h}x{scale_w}")

            if scale_h == scale_w:
                K = scale_h
                print(f"Block size: {K}x{K}")

                # Show what each input cell maps to
                for r in range(min(3, ih)):
                    for c in range(min(3, iw)):
                        inp_val = inp[r, c]
                        block = out[r*K:(r+1)*K, c*K:(c+1)*K]
                        print(f"  inp[{r},{c}]={inp_val} -> block={block.flatten().tolist()}")

# Test scale task
print("SCALE TASK c3e719e8:")
show_task('c3e719e8')

print("\n\nSCALE TASK f0afb749:")
show_task('f0afb749')

print("\n\nLINE RAY TASK 94414823:")
show_task('94414823')
