# Failure Analysis — Verantyx v65 (228/1000)

Generated: 2026-02-27 18:59 JST

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

## Failure Reasons

| Reason | Count | % |
|--------|-------|----|
| no_rule_matched | 453 | 58.7% |
| verification_gap | 184 | 23.8% |
| partial_match | 90 | 11.7% |
| incomplete_generalization | 45 | 5.8% |

## Unsolved by Category (LLM-classified)

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
| separator_split | 2 | 0.3% |
| counting_output | 1 | 0.1% |
| overlay_merge | 1 | 0.1% |

## Top 20 Solving Rules

| Rule | Count |
|------|-------|
| `neighborhood_rule` | 45 |
| `cross:puzzle:global_color_map` | 5 |
| `cross:tile:tile_2x2_transform` | 4 |
| `composite(dedup_rows+dedup_cols)` | 4 |
| `cross:puzzle:connect_same_color_lines` | 3 |
| `cross:puzzle:mirror_h_alt_tile` | 3 |
| `cross:extract_largest` | 3 |
| `composite(crop_bbox+tile_to_output)` | 3 |
| `corners_mirror` | 3 |
| `cross:puzzle:cell_to_copy_block` | 2 |
| `fill_enclosed` | 2 |
| `extract_smallest_region` | 2 |
| `cross:grow:self_stamp` | 2 |
| `cross:ptree:Seq(Apply(extract_majority_bbox) → Apply(extract_summary:lar` | 2 |
| `cross:puzzle:sep_h_xor_mark3` | 2 |
| `cross:puzzle:rotate_180_content` | 2 |
| `cross:sym_h` | 2 |
| `cross:puzzle:sep_h_or_mark3` | 2 |
| `cross:puzzle:upscale_2x` | 2 |
| `cross:puzzle:ray_extend_diag` | 2 |

## Unsolved Task IDs by Category

### color_map (3 tasks)
`b74ca5d1`, `bd14c3bf`, `d406998b`

### composition_2step (125 tasks)
`017c7c7b`, `05a7bcf2`, `0a1d4ef5`, `0d87d2a6`, `0f63c0b9`, `10fcaaa3`, `15660dd6`, `178fcbfb`, `17b866bd`, `184a9768`, `1990f7a8`, `1be83260`, `1c02dbbe`, `20981f0e`, `20fb2937`, `212895b5`, `234bbc79`, `278e5215`, `281123b4`, `305b1341` ... and 105 more

### conditional_transform (18 tasks)
`09c534e7`, `1da012fc`, `2204b7a8`, `2281f1f4`, `22a4bbc2`, `33b52de3`, `5b526a93`, `7d419a02`, `855e0971`, `868de0fa`, `941d9a10`, `97239e3d`, `ce9e57f2`, `db7260a4`, `df8cc377`, `e4888269`, `e681b708`, `fea12743`

### counting_output (1 tasks)
`1fad071e`

### crop_extract (22 tasks)
`1190e5a7`, `1a6449f1`, `25c199f5`, `2ccd9fef`, `337b420f`, `505fff84`, `50a16a69`, `5ecac7f7`, `6d1d5c90`, `79cce52d`, `8a6d367c`, `a6953f00`, `abbfd121`, `ba1aa698`, `bbb1b8b6`, `ca8de6ea`, `caa06a1f`, `cf98881b`, `e50d258f`, `e6721834` ... and 2 more

### extract_object (78 tasks)
`12997ef3`, `137eaa0f`, `19bb5feb`, `2037f2c7`, `20818e16`, `22425bda`, `2f0c5170`, `351d6448`, `3de23699`, `3f7978a0`, `4290ef0e`, `42f14c03`, `48d8fb45`, `4be741c5`, `50aad11f`, `5289ad53`, `57edb29d`, `5ad4f10b`, `5daaa586`, `652646ff` ... and 58 more

### flood_fill (31 tasks)
`0e671a1a`, `13713586`, `1d0a4b61`, `1e32b0e9`, `256b0a75`, `2dd70a9a`, `447fd412`, `484b58aa`, `538b439f`, `57aa92db`, `62b74c02`, `673ef223`, `692cd3b6`, `6ffe8f07`, `753ea09b`, `78e78cff`, `7e576d6e`, `928ad970`, `95755ff2`, `96a8c0cd` ... and 11 more

### gravity (49 tasks)
`05f2a901`, `0e206a2e`, `11dc524f`, `13f06aa5`, `17829a00`, `17b80ad2`, `22208ba4`, `29700607`, `2b01abd0`, `2c608aff`, `2c737e39`, `3d6c6e23`, `423a55dc`, `4acc7107`, `4df5b0ae`, `50c07299`, `5623160b`, `56dc2b01`, `6ca952ad`, `705a3229` ... and 29 more

### line_draw (10 tasks)
`3490cc26`, `508bd3b6`, `55059096`, `58e15b12`, `60a26a3e`, `7e2bad24`, `992798f6`, `ac605cbb`, `d06dbe63`, `d43fd935`

### neighborhood_rule (44 tasks)
`1190bc91`, `1478ab18`, `1c56ad9f`, `1efba499`, `264363fd`, `28e73c20`, `292dd178`, `2faf500b`, `3bdb4ada`, `42918530`, `42a15761`, `516b51b7`, `5ad8a7c0`, `689c358e`, `694f12f3`, `69889d6e`, `6aa20dc0`, `760b3cac`, `7acdf6d3`, `7df24a62` ... and 24 more

### object_recolor (4 tasks)
`103eff5b`, `44d8ac46`, `551d5bf1`, `e9ac8c9e`

### overlay_merge (1 tasks)
`e41c6fd3`

### pattern_stamp (47 tasks)
`045e512c`, `0a938d79`, `11e1fe23`, `1d398264`, `1f0c79e5`, `22233c11`, `2685904e`, `2697da3f`, `2b9ef948`, `33067df9`, `363442ee`, `3e980e27`, `3f23242b`, `40f6cd08`, `5af49b42`, `5c2c9af4`, `6d58a25d`, `72207abc`, `85fa5666`, `8719f442` ... and 27 more

### separator_split (2 tasks)
`d47aa2ff`, `e78887d1`

### symmetry (5 tasks)
`47c1f68c`, `4938f0c2`, `4c5c2cf0`, `f18ec8cc`, `f5c89df1`

### tiling (7 tasks)
`12422b43`, `58c02a16`, `695367ec`, `91413438`, `963e52fc`, `d8c310e9`, `eb281b96`

### unclassified (319 tasks)
`009d5c81`, `00dbd492`, `0607ce86`, `09629e4f`, `0962bcdd`, `0becf7df`, `0ca9ddb6`, `12eac192`, `137f0df0`, `14754a24`, `14b8e18c`, `150deff5`, `15113be4`, `15663ba9`, `17cae0c1`, `18286ef8`, `182e5d0f`, `18419cfa`, `18447a8d`, `1a07d186` ... and 299 more

### upscale_downscale (6 tasks)
`4522001f`, `53b68214`, `973e499e`, `cad67732`, `d749d46f`, `e3f79277`


---
*Auto-generated. Sponsor-exclusive detailed per-task reasoning traces available.*
