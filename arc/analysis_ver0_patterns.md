# ver=0 Pattern Analysis (30 task sample)

## Cluster A: "Object Move/Shift to Center" (64a7c07e)
- Object in grid → move to center/specific position
- 64a7c07e: 2x2 block moves right by 2, 3x3 frame moves right by 3
- Rule: move object by its own width (or some function of position)

## Cluster B: "L-shaped / Path Fill" (2dd70a9a, e5790162)
- Two anchor points with different colors → draw L-shaped path between them
- Path color = one anchor's color, direction determined by relative position

## Cluster C: "Separator-based Template Projection" (2685904e, f2829549, 6430c8c4)
- Grid split by separator into zones
- Template zone contains colors; marker zone defines count/shape
- Project template into empty zone based on marker info
- 2685904e: row of 8s defines count N → template colors with N occurrences projected up
- f2829549: separator col=1; left/right halves compared → XOR-like output in new color
- 6430c8c4: separator row; top/bottom halves → XOR of shapes → output

## Cluster D: "Recursive Concentric Frames" (3f23242b, a3f84088)
- Single dot → expand outward with concentric rectangles of different colors
- 3f23242b: dot at (4,4) → rings of colors 5,2,8 expanding outward

## Cluster E: "Region/Object → Summary Grid" (67636eac, 846bdb03, 94133066, a3325580)
- Multiple objects in grid → extract/arrange as summary
- 67636eac: scattered dots → group by color → each color's shape becomes a 3x3 pattern
- 846bdb03: objects in large grid → bbox of non-zero region
- a3325580: objects with different shapes → height of each object repeated

## Cluster F: "Tile with Conditional Placement" (15696249, 9c1e755f)  
- Input pattern → tile to fill larger area, with conditional transform
- 15696249: 3x3 → tile in 9x9, but only first row occupied (conditional by row content)
- 9c1e755f: header defines fill region; body rows/template rows fill the rectangle

## Cluster G: "Color Cycle / Rotation" (f18ec8cc, ba26e723, 652646ff)
- Colors in grid are cyclically rotated or shifted
- f18ec8cc: blocks rearranged in cyclic order
- ba26e723: every 3rd column of 4→6 (periodic substitution)
- 652646ff: extract diamond shapes, each filled with different color

## Cluster H: "Symmetry Completion" (045e512c, 8ee62060, 7d7772cc)
- Partial pattern → complete by symmetry (rotational, reflective)
- 045e512c: sparse dots → fill symmetric pattern around them

## Cluster I: "Line Drawing Between Anchors" (696d4842, 58e15b12, 762cd429)
- Anchor points → draw lines/rays from them
- 696d4842: colored dots → draw rays in specific directions
- 762cd429: 3 colored dots → fill rectangular regions

## Cluster J: "Conditional Cell Swap" (e4888269, 7e2bad24, 9f27f097)
- Same-size grids, few cells change
- Rule depends on neighbor relationships or position in object
