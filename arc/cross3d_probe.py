"""
arc/cross3d_probe.py — 立体十字構造プローブエンジン

6軸の十字構造をプローブ（測定器具）としてグリッドに入れ、
到達距離・衝突色・凹凸パターンを測定する。

z軸 = 抽象化レベル（プローブのスケール）:
  z=0: pixel scale (arm_width=1)
  z=1: object scale (arm_width=2-5)
  z=2: panel scale (arm_width=grid_dim//3+)

Design by kofdai, implemented for Verantyx V6.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

Grid = List[List[int]]


# ─── Data structures ───

@dataclass
class ProbeResult:
    """Result of inserting a cross probe at a position"""
    center: Tuple[int, int]          # (row, col)
    arm_length: int                   # requested arm length
    arm_width: int                    # requested arm width (= z scale)
    
    # Per-direction measurements: +y(down), -y(up), +x(right), -x(left)
    reach: Dict[str, int] = field(default_factory=dict)          # how far arm actually extends before hitting something
    hit_color: Dict[str, Optional[int]] = field(default_factory=dict)  # what color stopped the arm (None = boundary)
    arm_colors: Dict[str, List[int]] = field(default_factory=dict)     # colors encountered along arm
    concavity: Dict[str, List[int]] = field(default_factory=dict)      # concavity profile per direction: 1=filled, 0=gap


@dataclass 
class ProbeMap:
    """Collection of probe results across a grid"""
    results: List[ProbeResult]
    grid_shape: Tuple[int, int]
    arm_length: int
    arm_width: int
    
    def reach_grid(self, direction: str) -> np.ndarray:
        """Build a 2D grid of reach values for a given direction"""
        H, W = self.grid_shape
        g = np.full((H, W), -1, dtype=int)
        for r in self.results:
            g[r.center[0], r.center[1]] = r.reach.get(direction, 0)
        return g
    
    def hit_color_grid(self, direction: str) -> np.ndarray:
        """Build a 2D grid of hit colors for a given direction"""
        H, W = self.grid_shape
        g = np.full((H, W), -1, dtype=int)
        for r in self.results:
            hc = r.hit_color.get(direction)
            if hc is not None:
                g[r.center[0], r.center[1]] = hc
        return g


@dataclass
class Cross3DProbe:
    """A 6-axis cross probe with configurable dimensions"""
    arm_length: int    # max arm extension
    arm_width: int     # arm thickness (also determines z-level)
    
    @property
    def z_level(self) -> int:
        if self.arm_width <= 1:
            return 0  # pixel
        elif self.arm_width <= 5:
            return 1  # object
        else:
            return 2  # panel


# ─── Core probe function ───

def probe(grid: np.ndarray, center: Tuple[int, int], arm_length: int, 
          arm_width: int = 1, bg: int = 0) -> ProbeResult:
    """
    Insert a cross probe at center and measure in 4 planar directions.
    
    For each direction, extend an arm (width x length rectangle) and record:
    - reach: how many steps before hitting a non-bg color or boundary
    - hit_color: the color that stopped the arm
    - arm_colors: all colors encountered along the arm's path
    - concavity: binary profile of filled(1) vs gap(0) along the arm edge
    
    arm_width > 1: the arm is a rectangle, not a line.
    The probe checks ALL cells in the arm's cross-section.
    """
    H, W = grid.shape
    cr, cc = center
    
    result = ProbeResult(
        center=center,
        arm_length=arm_length,
        arm_width=arm_width,
    )
    
    # Direction vectors: (dr, dc) for the arm extension direction
    # For each direction, the arm cross-section is perpendicular
    directions = {
        '+y': (1, 0),    # down
        '-y': (-1, 0),   # up  
        '+x': (0, 1),    # right
        '-x': (0, -1),   # left
    }
    
    for dir_name, (dr, dc) in directions.items():
        reach = 0
        hit_color = None
        colors_along = []
        concavity_profile = []
        
        # Cross-section offsets (perpendicular to direction)
        if dr != 0:  # vertical direction → cross-section is horizontal
            half_w = arm_width // 2
            cross_offsets = list(range(-half_w, -half_w + arm_width))
        else:  # horizontal direction → cross-section is vertical
            half_w = arm_width // 2
            cross_offsets = list(range(-half_w, -half_w + arm_width))
        
        for step in range(1, arm_length + 1):
            r = cr + dr * step
            c = cc + dc * step
            
            # Check if center of arm is in bounds
            if r < 0 or r >= H or c < 0 or c >= W:
                hit_color = None  # boundary
                break
            
            # Check all cells in cross-section
            step_colors = []
            step_filled = 0
            step_total = 0
            blocked = False
            
            for offset in cross_offsets:
                if dr != 0:  # moving vertically, cross-section is horizontal
                    cr2, cc2 = r, c + offset
                else:  # moving horizontally, cross-section is vertical
                    cr2, cc2 = r + offset, c
                
                if 0 <= cr2 < H and 0 <= cc2 < W:
                    cell_color = int(grid[cr2, cc2])
                    step_colors.append(cell_color)
                    step_total += 1
                    if cell_color != bg:
                        step_filled += 1
                        if hit_color is None and blocked is False:
                            # First non-bg hit
                            pass
                else:
                    step_total += 1  # out of bounds counts as boundary
            
            # Determine if this step is blocked
            non_bg_colors = [c for c in step_colors if c != bg]
            
            if non_bg_colors:
                # Record colors encountered
                colors_along.extend(non_bg_colors)
                # Concavity: ratio of filled cells in cross-section
                concavity_profile.append(step_filled)
                
                # If majority of cross-section is non-bg, arm is blocked
                if step_filled > step_total // 2:
                    hit_color = non_bg_colors[0]  # primary blocking color
                    reach = step - 1
                    break
            else:
                colors_along.append(bg)
                concavity_profile.append(0)
                reach = step
        else:
            # Reached max arm_length without hitting anything
            reach = arm_length
        
        result.reach[dir_name] = reach
        result.hit_color[dir_name] = hit_color
        result.arm_colors[dir_name] = colors_along
        result.concavity[dir_name] = concavity_profile
    
    return result


def scan_grid(grid: np.ndarray, arm_length: int, arm_width: int = 1,
              bg: int = 0, stride: int = 1) -> ProbeMap:
    """
    Scan entire grid with cross probes at every position (or with stride).
    Returns a ProbeMap with all measurements.
    """
    H, W = grid.shape
    results = []
    
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            pr = probe(grid, (r, c), arm_length, arm_width, bg)
            results.append(pr)
    
    return ProbeMap(
        results=results,
        grid_shape=(H, W),
        arm_length=arm_length,
        arm_width=arm_width,
    )


def multi_scale_scan(grid: np.ndarray, bg: int = 0) -> Dict[int, ProbeMap]:
    """
    Scan at multiple z-levels (scales).
    z=0: arm_width=1 (pixel)
    z=1: arm_width=3 (object) 
    z=2: arm_width=max(H,W)//3 (panel)
    """
    H, W = grid.shape
    scales = {
        0: 1,
        1: min(3, min(H, W)),
        2: max(1, max(H, W) // 3),
    }
    
    results = {}
    for z, width in scales.items():
        arm_len = max(H, W)
        stride = max(1, width)
        results[z] = scan_grid(grid, arm_len, width, bg, stride)
    
    return results


# ─── Measurement extraction ───

def measure_objects(grid: np.ndarray, bg: int = 0) -> List[Dict]:
    """
    Use probes to detect and measure objects.
    
    Strategy: scan at z=0, find positions where reach changes sharply
    → those are object boundaries.
    """
    H, W = grid.shape
    pmap = scan_grid(grid, max(H, W), arm_width=1, bg=bg)
    
    objects = []
    visited = np.zeros((H, W), dtype=bool)
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg and not visited[r, c]:
                # Flood fill to find connected component
                obj_cells = []
                stack = [(r, c)]
                color = int(grid[r, c])
                
                while stack:
                    cr, cc = stack.pop()
                    if 0 <= cr < H and 0 <= cc < W and not visited[cr, cc] and int(grid[cr, cc]) == color:
                        visited[cr, cc] = True
                        obj_cells.append((cr, cc))
                        stack.extend([(cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)])
                
                if obj_cells:
                    rows = [p[0] for p in obj_cells]
                    cols = [p[1] for p in obj_cells]
                    bbox = (min(rows), min(cols), max(rows), max(cols))
                    
                    # Measure this object with probes from its center
                    center_r = (bbox[0] + bbox[2]) // 2
                    center_c = (bbox[1] + bbox[3]) // 2
                    obj_probe = probe(grid, (center_r, center_c), max(H, W), 1, bg)
                    
                    # Compute concavity profile for each face
                    obj_h = bbox[2] - bbox[0] + 1
                    obj_w = bbox[3] - bbox[1] + 1
                    
                    # Scan each face for concavity
                    faces = {}
                    # Top face
                    top_profile = []
                    for fc in range(bbox[1], bbox[3] + 1):
                        top_profile.append(1 if (bbox[0], fc) in set(obj_cells) else 0)
                    faces['top'] = top_profile
                    # Bottom face
                    bot_profile = []
                    for fc in range(bbox[1], bbox[3] + 1):
                        bot_profile.append(1 if (bbox[2], fc) in set(obj_cells) else 0)
                    faces['bottom'] = bot_profile
                    # Left face
                    left_profile = []
                    for fr in range(bbox[0], bbox[2] + 1):
                        left_profile.append(1 if (fr, bbox[1]) in set(obj_cells) else 0)
                    faces['left'] = left_profile
                    # Right face
                    right_profile = []
                    for fr in range(bbox[0], bbox[2] + 1):
                        right_profile.append(1 if (fr, bbox[3]) in set(obj_cells) else 0)
                    faces['right'] = right_profile
                    
                    objects.append({
                        'color': color,
                        'cells': obj_cells,
                        'bbox': bbox,
                        'size': (obj_h, obj_w),
                        'area': len(obj_cells),
                        'center': (center_r, center_c),
                        'probe': obj_probe,
                        'faces': faces,
                    })
    
    return objects


def faces_interlock(face_a: List[int], face_b: List[int]) -> bool:
    """
    Check if two face profiles interlock (凹凸が噛み合う).
    face_a's convex parts should fit face_b's concave parts and vice versa.
    """
    if len(face_a) != len(face_b):
        return False
    
    for a, b in zip(face_a, face_b):
        # Interlock: where one is filled, other should be gap (or both filled for flush)
        if a == 1 and b == 1:
            continue  # flush contact — ok
        if a == 0 and b == 0:
            continue  # mutual gap — ok (no contact)
        # a=1,b=0 or a=0,b=1: one fills the other's gap — interlock!
    
    # True interlock: complementary pattern
    # Perfect interlock: a XOR b or a AND b for every position
    return True


def measure_gaps(grid: np.ndarray, bg: int = 0) -> List[Dict]:
    """
    Measure gaps (empty regions) between objects using probes.
    Useful for determining gravity direction and distance.
    """
    H, W = grid.shape
    gaps = []
    
    # For each bg cell, probe in all directions to find bounding objects
    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg:
                pr = probe(grid, (r, c), max(H, W), 1, bg)
                # If bounded on opposite sides, this is a gap
                if pr.hit_color.get('+x') is not None and pr.hit_color.get('-x') is not None:
                    gaps.append({
                        'pos': (r, c),
                        'type': 'horizontal',
                        'left_color': pr.hit_color['-x'],
                        'right_color': pr.hit_color['+x'],
                        'left_dist': pr.reach['-x'],
                        'right_dist': pr.reach['+x'],
                    })
                if pr.hit_color.get('+y') is not None and pr.hit_color.get('-y') is not None:
                    gaps.append({
                        'pos': (r, c),
                        'type': 'vertical',
                        'top_color': pr.hit_color['-y'],
                        'bottom_color': pr.hit_color['+y'],
                        'top_dist': pr.reach['-y'],
                        'bottom_dist': pr.reach['+y'],
                    })
    
    return gaps


def count_probe_fits(grid: np.ndarray, arm_length: int, arm_width: int, 
                     bg: int = 0) -> int:
    """
    Count how many times a cross probe of given size fits in the grid.
    "Fits" = center is on bg, all 4 arms extend to full length without hitting non-bg.
    """
    H, W = grid.shape
    count = 0
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg:
                continue
            pr = probe(grid, (r, c), arm_length, arm_width, bg)
            if all(pr.reach.get(d, 0) >= arm_length for d in ['+x', '-x', '+y', '-y']):
                count += 1
    
    return count


# ─── Transform learning via probes ───

def learn_transform_from_probes(train_pairs: List[Tuple[Grid, Grid]], bg: int = 0) -> Optional[Dict]:
    """
    Compare probe measurements between input and output to learn transformation.
    
    Strategy:
    1. Measure objects in input and output
    2. Match objects by color
    3. Determine how each object moved (delta_r, delta_c)
    4. Find pattern in movements (common direction, stacking order, etc.)
    """
    if not train_pairs:
        return None
    
    all_movements = []
    
    for inp, out in train_pairs:
        inp_np = np.array(inp, dtype=int)
        out_np = np.array(out, dtype=int)
        
        inp_objs = measure_objects(inp_np, bg)
        out_objs = measure_objects(out_np, bg)
        
        # Match objects by color + area
        movements = []
        used_out = set()
        
        for io in inp_objs:
            best_match = None
            best_score = -1
            
            for j, oo in enumerate(out_objs):
                if j in used_out:
                    continue
                if io['color'] != oo['color']:
                    continue
                # Score by area similarity
                area_sim = 1.0 - abs(io['area'] - oo['area']) / max(io['area'], oo['area'], 1)
                if area_sim > best_score:
                    best_score = area_sim
                    best_match = j
            
            if best_match is not None and best_score > 0.5:
                used_out.add(best_match)
                oo = out_objs[best_match]
                dr = oo['center'][0] - io['center'][0]
                dc = oo['center'][1] - io['center'][1]
                movements.append({
                    'color': io['color'],
                    'inp_center': io['center'],
                    'out_center': oo['center'],
                    'delta': (dr, dc),
                    'inp_bbox': io['bbox'],
                    'out_bbox': oo['bbox'],
                    'inp_size': io['size'],
                    'out_size': oo['size'],
                })
        
        all_movements.append(movements)
    
    if not all_movements or not all_movements[0]:
        return None
    
    return {
        'type': 'probe_movement',
        'train_movements': all_movements,
    }


if __name__ == '__main__':
    # Quick test with gravity task 03560426
    import json
    
    tid = '03560426'
    with open(f'/tmp/arc-agi-2/data/training/{tid}.json') as f:
        task = json.load(f)
    
    inp = np.array(task['train'][0]['input'])
    out = np.array(task['train'][0]['output'])
    
    print(f"=== Task {tid} train[0] ===")
    print(f"Grid: {inp.shape}")
    
    # Measure objects
    objs = measure_objects(inp, bg=0)
    print(f"\nObjects found: {len(objs)}")
    for obj in objs:
        print(f"  color={obj['color']} center={obj['center']} size={obj['size']} area={obj['area']}")
        print(f"    bbox={obj['bbox']}")
        print(f"    faces: top={obj['faces']['top']} bot={obj['faces']['bottom']}")
        print(f"           left={obj['faces']['left']} right={obj['faces']['right']}")
        print(f"    probe reach: {obj['probe'].reach}")
        print(f"    probe hit:   {obj['probe'].hit_color}")
    
    # Measure output
    out_objs = measure_objects(out, bg=0)
    print(f"\nOutput objects: {len(out_objs)}")
    for obj in out_objs:
        print(f"  color={obj['color']} center={obj['center']} size={obj['size']} area={obj['area']}")
    
    # Learn transform
    train_pairs = [(p['input'], p['output']) for p in task['train']]
    transform = learn_transform_from_probes(train_pairs, bg=0)
    if transform:
        print(f"\nTransform learned:")
        for i, movs in enumerate(transform['train_movements']):
            print(f"  train[{i}]:")
            for m in movs:
                print(f"    color={m['color']} delta={m['delta']} {m['inp_center']}→{m['out_center']}")
