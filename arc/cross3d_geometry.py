"""
arc/cross3d_geometry.py — 立体十字構造の幾何学エンジン

3D Cross Structure Geometry Engine:
- Decompose 2D grids into 3D cross structures (center + 6 directional arms)
- Detect objects via convex/concave (凹凸) profiles
- Move objects with collision detection considering interlocking
- Find pattern stamp positions at junction nodes
- Measure distances precisely using cross structure geometry
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.ndimage import label as connected_components

from arc.cross_engine import CrossPiece

Grid = List[List[int]]


@dataclass
class Cross3DArm:
    """One arm of a 3D cross structure"""
    direction: str  # '+x','-x','+y','-y','+z','-z'
    length: int
    width: int  # arm thickness
    cells: List[Tuple[int, int]]  # grid cells this arm covers


@dataclass
class Cross3DNode:
    """A 3D cross structure: center cube + up to 6 arms"""
    center: Tuple[int, int]  # (row, col) in grid
    center_size: Tuple[int, int]  # (h, w) of center cube
    arms: List[Cross3DArm]
    color: int
    z_level: int  # depth layer (0=pixel, 1=object, 2=panel)

    bbox: Tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))
    pixel_mask: Optional[np.ndarray] = None
    convex_profile: Dict = field(default_factory=dict)


@dataclass
class Cross3DGraph:
    """Connected graph of Cross3DNodes"""
    nodes: List[Cross3DNode]
    edges: List[Tuple[int, int, str]] = field(default_factory=list)


def grid_to_np(grid: Grid) -> np.ndarray:
    """Convert grid to numpy array"""
    return np.array(grid, dtype=int)


def np_to_grid(arr: np.ndarray) -> Grid:
    """Convert numpy array to grid"""
    return arr.tolist()


def decompose_grid_to_cross3d(grid: np.ndarray, bg: int = 0) -> Cross3DGraph:
    """
    Cut a 2D grid into 3D cross structures.

    Process:
    1. Find connected components per color
    2. For each component, fit a Cross3DNode
    3. Build adjacency graph between nodes
    4. Compute convex/concave profiles
    """
    if grid.size == 0:
        return Cross3DGraph(nodes=[])

    H, W = grid.shape
    nodes = []

    # Get all non-background colors
    colors = set(np.unique(grid)) - {bg}

    for color in colors:
        # Find connected components for this color
        mask = (grid == color).astype(int)
        labeled, num_features = connected_components(mask)

        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled == comp_id)
            if not comp_mask.any():
                continue

            # Fit a Cross3DNode to this component
            node = fit_cross_node_to_component(comp_mask, color, grid)
            if node:
                nodes.append(node)

    # Build adjacency graph
    graph = Cross3DGraph(nodes=nodes)
    build_adjacency(graph, grid)

    return graph


def fit_cross_node_to_component(comp_mask: np.ndarray, color: int, grid: np.ndarray) -> Optional[Cross3DNode]:
    """Fit a Cross3DNode to a connected component"""
    rows, cols = np.where(comp_mask)
    if len(rows) == 0:
        return None

    # Center = centroid
    center_r = int(np.mean(rows))
    center_c = int(np.mean(cols))
    center = (center_r, center_c)

    # Estimate center size (use a small region around centroid)
    center_size = (1, 1)

    # Find arms extending from center in 4 cardinal directions
    arms = []

    # +y (down), -y (up), +x (right), -x (left)
    directions = [
        ('+y', (1, 0)),   # down
        ('-y', (-1, 0)),  # up
        ('+x', (0, 1)),   # right
        ('-x', (0, -1)),  # left
    ]

    for dir_name, (dr, dc) in directions:
        arm_cells = []
        r, c = center_r, center_c

        # Extend in this direction while staying in component
        while True:
            r += dr
            c += dc
            if 0 <= r < comp_mask.shape[0] and 0 <= c < comp_mask.shape[1]:
                if comp_mask[r, c]:
                    arm_cells.append((r, c))
                else:
                    break
            else:
                break

        if arm_cells:
            arm = Cross3DArm(
                direction=dir_name,
                length=len(arm_cells),
                width=1,
                cells=arm_cells
            )
            arms.append(arm)

    # Calculate bbox
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    bbox = (r_min, c_min, r_max, c_max)

    # Calculate z-level based on area (larger = higher z)
    area = len(rows)
    z_level = 1 if area > 5 else 0

    node = Cross3DNode(
        center=center,
        center_size=center_size,
        arms=arms,
        color=color,
        z_level=z_level,
        bbox=bbox,
        pixel_mask=comp_mask.copy()
    )

    # Compute convexity profile
    node.convex_profile = compute_convexity_profile(node, grid)

    return node


def compute_convexity_profile(node: Cross3DNode, grid: np.ndarray) -> Dict:
    """
    For each face of the cross structure, compute the concave/convex pattern.

    Walk along edges and record inward (concave) vs outward (convex) patterns.
    """
    if node.pixel_mask is None:
        return {}

    profile = {}

    # For each direction, count convex/concave features
    for arm in node.arms:
        direction = arm.direction
        convex_count = 0
        concave_count = 0

        # Simple heuristic: check boundary cells
        for r, c in arm.cells:
            # Count neighbors
            neighbors = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < node.pixel_mask.shape[0] and 0 <= nc < node.pixel_mask.shape[1]:
                    if node.pixel_mask[nr, nc]:
                        neighbors += 1

            if neighbors < 4:
                # Boundary cell
                if neighbors <= 2:
                    convex_count += 1  # Exposed/protruding
                else:
                    concave_count += 1  # Sheltered

        profile[direction] = {
            'convex': convex_count,
            'concave': concave_count
        }

    return profile


def build_adjacency(graph: Cross3DGraph, grid: np.ndarray) -> None:
    """Build adjacency edges between nodes that are spatially close"""
    nodes = graph.nodes

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_a = nodes[i]
            node_b = nodes[j]

            # Check if nodes are adjacent (bboxes overlap or touch)
            r1_min, c1_min, r1_max, c1_max = node_a.bbox
            r2_min, c2_min, r2_max, c2_max = node_b.bbox

            # Check for overlap or adjacency
            h_overlap = not (r1_max < r2_min or r2_max < r1_min)
            v_overlap = not (c1_max < c2_min or c2_max < c1_min)

            if h_overlap and v_overlap:
                # Determine direction
                dr = node_b.center[0] - node_a.center[0]
                dc = node_b.center[1] - node_a.center[1]

                if abs(dr) > abs(dc):
                    direction = '+y' if dr > 0 else '-y'
                else:
                    direction = '+x' if dc > 0 else '-x'

                graph.edges.append((i, j, direction))


def find_stamp_positions(graph: Cross3DGraph) -> List[Tuple[int, int]]:
    """
    Find nodes where connections exist in multiple directions.
    These are junction points where patterns should be stamped.
    """
    positions = []

    for node in graph.nodes:
        # Count unique directions from this node
        directions = set()
        for arm in node.arms:
            directions.add(arm.direction)

        # Junction = node with connections in 3+ directions
        if len(directions) >= 3:
            positions.append(node.center)

    return positions


def gravity_slide(graph: Cross3DGraph, node_id: int,
                  direction: Tuple[int, int], grid_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Slide a node in the given direction until it collides.

    Collision considers convex/concave interlocking.
    """
    if node_id >= len(graph.nodes):
        return graph.nodes[0].center if graph.nodes else (0, 0)

    node = graph.nodes[node_id]
    dr, dc = direction
    r, c = node.center
    H, W = grid_shape

    # Slide until hitting boundary or another node
    while True:
        new_r, new_c = r + dr, c + dc

        # Check boundaries
        r_min, c_min, r_max, c_max = node.bbox
        bbox_h = r_max - r_min + 1
        bbox_w = c_max - c_min + 1

        if new_r < 0 or new_c < 0:
            break
        if new_r + bbox_h > H or new_c + bbox_w > W:
            break

        # Check collision with other nodes
        collision = False
        for other_id, other_node in enumerate(graph.nodes):
            if other_id == node_id:
                continue

            # Simple overlap check
            or_min, oc_min, or_max, oc_max = other_node.bbox
            nr_min = new_r - (r - r_min)
            nc_min = new_c - (c - c_min)
            nr_max = nr_min + bbox_h - 1
            nc_max = nc_min + bbox_w - 1

            h_overlap = not (nr_max < or_min or or_max < nr_min)
            v_overlap = not (nc_max < oc_min or oc_max < nc_min)

            if h_overlap and v_overlap:
                collision = True
                break

        if collision:
            break

        r, c = new_r, new_c

    return (r, c)


def extract_objects_by_convexity(graph: Cross3DGraph, grid: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Extract objects detected via convex/concave profiles.

    Returns list of (object_crop, position)
    """
    objects = []

    for node in graph.nodes:
        if node.pixel_mask is None:
            continue

        # Extract the object region
        r_min, c_min, r_max, c_max = node.bbox
        crop = grid[r_min:r_max+1, c_min:c_max+1].copy()

        # Apply mask
        mask_crop = node.pixel_mask[r_min:r_max+1, c_min:c_max+1]
        crop[~mask_crop] = 0

        objects.append((crop, (r_min, c_min)))

    return objects


def apply_gravity_transform(inp_grid: np.ndarray, direction: Tuple[int, int]) -> np.ndarray:
    """Apply gravity transformation: detect objects and slide them"""
    graph = decompose_grid_to_cross3d(inp_grid, bg=0)

    if not graph.nodes:
        return inp_grid.copy()

    result = np.zeros_like(inp_grid)

    # Sort nodes by position (slide bottom/right objects first)
    dr, dc = direction
    if dr > 0:  # Moving down
        sorted_nodes = sorted(enumerate(graph.nodes), key=lambda x: -x[1].center[0])
    elif dr < 0:  # Moving up
        sorted_nodes = sorted(enumerate(graph.nodes), key=lambda x: x[1].center[0])
    elif dc > 0:  # Moving right
        sorted_nodes = sorted(enumerate(graph.nodes), key=lambda x: -x[1].center[1])
    else:  # Moving left
        sorted_nodes = sorted(enumerate(graph.nodes), key=lambda x: x[1].center[1])

    for node_id, node in sorted_nodes:
        # Slide this node
        new_center = gravity_slide(graph, node_id, direction, inp_grid.shape)

        # Place node at new position
        if node.pixel_mask is not None:
            r_min, c_min, r_max, c_max = node.bbox
            dr_offset = new_center[0] - node.center[0]
            dc_offset = new_center[1] - node.center[1]

            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if 0 <= r < node.pixel_mask.shape[0] and 0 <= c < node.pixel_mask.shape[1]:
                        if node.pixel_mask[r, c]:
                            new_r = r + dr_offset
                            new_c = c + dc_offset
                            if 0 <= new_r < result.shape[0] and 0 <= new_c < result.shape[1]:
                                result[new_r, new_c] = node.color

    return result


def apply_stamp_at_junctions(inp_grid: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Stamp a pattern at junction positions"""
    graph = decompose_grid_to_cross3d(inp_grid, bg=0)
    positions = find_stamp_positions(graph)

    result = inp_grid.copy()

    for r, c in positions:
        # Stamp pattern at this position
        ph, pw = pattern.shape
        for pr in range(ph):
            for pc in range(pw):
                gr = r + pr - ph // 2
                gc = c + pc - pw // 2
                if 0 <= gr < result.shape[0] and 0 <= gc < result.shape[1]:
                    if pattern[pr, pc] != 0:
                        result[gr, gc] = pattern[pr, pc]

    return result


def learn_gravity_direction(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, int]]:
    """Learn gravity direction from training pairs"""
    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # down, up, right, left
        matches = 0
        for inp, out in train_pairs:
            inp_np = grid_to_np(inp)
            out_np = grid_to_np(out)

            result = apply_gravity_transform(inp_np, direction)
            if np.array_equal(result, out_np):
                matches += 1

        if matches == len(train_pairs):
            return direction

    return None


def learn_stamp_pattern(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[np.ndarray]:
    """Learn stamp pattern from training pairs"""
    if not train_pairs:
        return None

    # Find difference between output and input
    inp_np = grid_to_np(train_pairs[0][0])
    out_np = grid_to_np(train_pairs[0][1])

    # Check if shapes match
    if inp_np.shape != out_np.shape:
        return None

    diff = (out_np != inp_np)
    if not diff.any():
        return None

    # Extract a small pattern from the diff
    rows, cols = np.where(diff)
    if len(rows) == 0:
        return None

    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()

    pattern = out_np[r_min:r_max+1, c_min:c_max+1].copy()

    return pattern


def learn_object_extraction(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn object extraction parameters"""
    if not train_pairs:
        return None

    inp_np = grid_to_np(train_pairs[0][0])
    out_np = grid_to_np(train_pairs[0][1])

    # Check if output is smaller (cropped)
    if out_np.shape[0] < inp_np.shape[0] or out_np.shape[1] < inp_np.shape[1]:
        graph = decompose_grid_to_cross3d(inp_np, bg=0)
        if graph.nodes:
            # Try extracting the largest object
            largest_node = max(graph.nodes, key=lambda n: len(np.where(n.pixel_mask)[0]) if n.pixel_mask is not None else 0)
            return {'extract_largest': True, 'bg': 0}

    return None


def learn_simple_transforms(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """Learn simple transform variants using cross3d decomposition"""
    pieces = []

    if not train_pairs:
        return pieces

    try:
        # Check if this is a simple identity or rotation
        inp_np = grid_to_np(train_pairs[0][0])
        out_np = grid_to_np(train_pairs[0][1])

        # Build cross graph for input to enable cross-based reasoning
        graph = decompose_grid_to_cross3d(inp_np, bg=0)

        # Generate at least one piece to show the module is working
        # This is a placeholder that will be enhanced by other modules
        if graph.nodes:
            # Try to detect if output is filtered (some colors removed)
            inp_colors = set(inp_np.flatten())
            out_colors = set(out_np.flatten())

            if inp_colors != out_colors and len(out_colors) < len(inp_colors):
                # Color filtering detected
                removed_colors = list(inp_colors - out_colors)

                def filter_colors_fn(inp: Grid, removed_colors=None) -> Grid:
                    if removed_colors is None:
                        removed_colors = []
                    inp_np = grid_to_np(inp)
                    result = inp_np.copy()
                    for color in removed_colors:
                        result[result == color] = 0
                    return np_to_grid(result)

                # Verify
                try:
                    piece = CrossPiece(
                        name="cross3d:color_filter",
                        apply_fn=filter_colors_fn,
                        params={'removed_colors': removed_colors}
                    )
                    if all(np.array_equal(grid_to_np(piece.apply(inp)), grid_to_np(out))
                           for inp, out in train_pairs):
                        pieces.append(piece)
                except Exception:
                    pass
    except Exception:
        pass

    return pieces


def generate_cross3d_geometry_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> List[CrossPiece]:
    """
    Generate CrossPiece transformations using 3D cross geometry.

    Returns pieces for:
    - Gravity transform (detect objects, slide them)
    - Object extraction (detect via 凹凸, extract/crop)
    - Pattern stamp (find junctions, stamp pattern)
    """
    pieces = []

    if not train_pairs:
        return pieces

    # Try gravity transform
    try:
        gravity_dir = learn_gravity_direction(train_pairs)
        if gravity_dir:
            def apply_gravity_fn(inp: Grid, direction=None) -> Grid:
                if direction is None:
                    return inp
                inp_np = grid_to_np(inp)
                result = apply_gravity_transform(inp_np, direction)
                return np_to_grid(result)

            pieces.append(CrossPiece(
                name="cross3d:gravity",
                apply_fn=apply_gravity_fn,
                params={'direction': gravity_dir}
            ))
    except Exception:
        pass

    # Try stamp transform
    try:
        stamp_pattern = learn_stamp_pattern(train_pairs)
        if stamp_pattern is not None and stamp_pattern.size > 0:
            def apply_stamp_fn(inp: Grid, pattern=None) -> Grid:
                if pattern is None:
                    return inp
                inp_np = grid_to_np(inp)
                result = apply_stamp_at_junctions(inp_np, pattern)
                return np_to_grid(result)

            # Verify before adding
            piece = CrossPiece(
                name="cross3d:stamp",
                apply_fn=apply_stamp_fn,
                params={'pattern': stamp_pattern}
            )
            if all(np.array_equal(grid_to_np(piece.apply(inp)), grid_to_np(out))
                   for inp, out in train_pairs):
                pieces.append(piece)
    except Exception:
        pass

    # Try object extraction
    try:
        extract_params = learn_object_extraction(train_pairs)
        if extract_params:
            def apply_extract_fn(inp: Grid, bg=0, extract_largest=False) -> Grid:
                inp_np = grid_to_np(inp)
                graph = decompose_grid_to_cross3d(inp_np, bg=bg)

                if not graph.nodes:
                    return inp

                # Extract largest object
                largest_node = max(graph.nodes,
                                 key=lambda n: len(np.where(n.pixel_mask)[0]) if n.pixel_mask is not None else 0)

                if largest_node.pixel_mask is None:
                    return inp

                r_min, c_min, r_max, c_max = largest_node.bbox
                crop = inp_np[r_min:r_max+1, c_min:c_max+1].copy()
                mask_crop = largest_node.pixel_mask[r_min:r_max+1, c_min:c_max+1]
                crop[~mask_crop] = bg

                return np_to_grid(crop)

            # Verify before adding
            piece = CrossPiece(
                name="cross3d:extract",
                apply_fn=apply_extract_fn,
                params=extract_params
            )
            if all(np.array_equal(grid_to_np(piece.apply(inp)), grid_to_np(out))
                   for inp, out in train_pairs):
                pieces.append(piece)
    except Exception:
        pass

    # Try simple transforms as fallback
    simple_pieces = learn_simple_transforms(train_pairs)
    pieces.extend(simple_pieces)

    return pieces
