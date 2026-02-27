# Failure Analysis — Verantyx V62 (228/1000)

## Overview
- **Solved**: 228/1000 (22.8%)
- **Unsolved**: 772/1000
- **Avg time**: 0.65s/task
- **Total eval time**: 648s

## Score by Verification Level

| ver | Total | Solved | Rate |
|-----|-------|--------|------|
| 0 | 453 | 0 | 0.0% |
| 1 | 134 | 44 | 32.8% |
| 2 | 121 | 76 | 62.8% |
| 3 | 43 | 19 | 44.2% |
| 4 | 9 | 8 | 88.9% |
| 5 | 170 | 20 | 11.8% |
| 6 | 25 | 21 | 84.0% |
| 7 | 20 | 17 | 85.0% |
| 8 | 11 | 10 | 90.9% |
| 9 | 7 | 7 | 100.0% |
| 10 | 1 | 1 | 100.0% |
| 11 | 2 | 1 | 50.0% |
| 13 | 1 | 1 | 100.0% |
| 14 | 1 | 1 | 100.0% |
| 17 | 1 | 1 | 100.0% |
| 19 | 1 | 1 | 100.0% |

## Unsolved Tasks by Category (LLM-classified)

| Category | Count | % of Unsolved |
|----------|-------|--------------|
| unclassified | 319 | 41.3% |
| composition_2step | 125 | 16.2% |
| extract_object | 78 | 10.1% |
| gravity | 49 | 6.3% |
| pattern_stamp | 47 | 6.1% |
| neighborhood_rule | 44 | 5.7% |
| flood_fill | 31 | 4.0% |
| crop_extract | 22 | 2.8% |
| conditional_transform | 18 | 2.3% |
| line_draw | 10 | 1.3% |
| tiling | 7 | 0.9% |
| upscale_downscale | 6 | 0.8% |
| symmetry | 5 | 0.6% |
| object_recolor | 4 | 0.5% |
| color_map | 3 | 0.4% |

## Top Solving Rules

| Rule | Count |
|------|-------|
| neighborhood_rule | 45 |
| cross:puzzle:global_color_map | 5 |
| cross:tile:tile_2x2_transform | 4 |
| composite(dedup_rows+dedup_cols) | 4 |
| cross:puzzle:connect_same_color_lines | 3 |
| cross:puzzle:mirror_h_alt_tile | 3 |
| cross:extract_largest | 3 |
| composite(crop_bbox+tile_to_output) | 3 |
| corners_mirror | 3 |
| cross:puzzle:cell_to_copy_block | 2 |
| fill_enclosed | 2 |
| extract_smallest_region | 2 |
| cross:grow:self_stamp | 2 |
| cross:ptree:Seq(Apply(extract_majority_bbox) → Apply(extract_summary:lar | 2 |
| cross:puzzle:sep_h_xor_mark3 | 2 |

---

*This report is generated automatically for each eval run.*
*Sponsor-exclusive: detailed per-task reasoning traces available on request.*
