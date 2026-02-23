"""
arc/grid_partition.py — Grid partition detection and panel-based transforms

Cross-Structure approach:
  Piece 1: Grid → panels (separator detection)
  Piece 2: Panel relationship (overlay, AND, OR, XOR, template fill)
  Piece 3: Panels → output grid (reassembly)
"""

from typing import List, Tuple, Optional, Dict
from arc.grid import Grid, grid_shape, grid_eq, most_common_color


def detect_separators(g: Grid) -> Optional[Dict]:
    """Detect separator lines (full row/col of single non-bg color)"""
    h, w = grid_shape(g)
    bg = most_common_color(g)
    
    sep_rows = []
    sep_color = None
    for r in range(h):
        vals = set(g[r])
        if len(vals) == 1 and g[r][0] != bg:
            sep_rows.append(r)
            sep_color = g[r][0]
    
    sep_cols = []
    for c in range(w):
        vals = set(g[r][c] for r in range(h))
        if len(vals) == 1 and g[0][c] != bg:
            sep_cols.append(c)
            if sep_color is None:
                sep_color = g[0][c]
    
    if not sep_rows and not sep_cols:
        return None
    
    # Build panel boundaries
    row_bounds = []
    prev = 0
    for r in sorted(sep_rows):
        if r > prev:
            row_bounds.append((prev, r - 1))
        prev = r + 1
    if prev < h:
        row_bounds.append((prev, h - 1))
    
    col_bounds = []
    prev = 0
    for c in sorted(sep_cols):
        if c > prev:
            col_bounds.append((prev, c - 1))
        prev = c + 1
    if prev < w:
        col_bounds.append((prev, w - 1))
    
    if not row_bounds:
        row_bounds = [(0, h - 1)]
    if not col_bounds:
        col_bounds = [(0, w - 1)]
    
    # Check all panels are same size
    panel_heights = set(r2 - r1 + 1 for r1, r2 in row_bounds)
    panel_widths = set(c2 - c1 + 1 for c1, c2 in col_bounds)
    
    if len(panel_heights) != 1 or len(panel_widths) != 1:
        return None  # Non-uniform panels
    
    return {
        'sep_rows': sorted(sep_rows),
        'sep_cols': sorted(sep_cols),
        'sep_color': sep_color,
        'row_bounds': row_bounds,
        'col_bounds': col_bounds,
        'panel_h': panel_heights.pop(),
        'panel_w': panel_widths.pop(),
        'n_panels_r': len(row_bounds),
        'n_panels_c': len(col_bounds),
        'bg': bg,
    }


def extract_panels(g: Grid, partition: Dict) -> List[List[Grid]]:
    """Extract panels as 2D array of grids"""
    panels = []
    for r1, r2 in partition['row_bounds']:
        row = []
        for c1, c2 in partition['col_bounds']:
            panel = [g[r][c1:c2+1] for r in range(r1, r2+1)]
            row.append(panel)
        panels.append(row)
    return panels


def reassemble(panels: List[List[Grid]], partition: Dict) -> Grid:
    """Reassemble panels back into a grid with separators"""
    sep_color = partition['sep_color']
    bg = partition['bg']
    h_total = sum(r2 - r1 + 1 for r1, r2 in partition['row_bounds'])
    h_total += len(partition['sep_rows'])
    w_total = sum(c2 - c1 + 1 for c1, c2 in partition['col_bounds'])
    w_total += len(partition['sep_cols'])
    
    result = [[bg] * w_total for _ in range(h_total)]
    
    # Place separator rows
    for sr in partition['sep_rows']:
        for c in range(w_total):
            result[sr][c] = sep_color
    
    # Place separator cols
    for sc in partition['sep_cols']:
        for r in range(h_total):
            result[r][sc] = sep_color
    
    # Place panels
    for pi, (r1, r2) in enumerate(partition['row_bounds']):
        for pj, (c1, c2) in enumerate(partition['col_bounds']):
            panel = panels[pi][pj]
            ph, pw = grid_shape(panel)
            for r in range(ph):
                for c in range(pw):
                    result[r1 + r][c1 + c] = panel[r][c]
    
    return result


def panel_nonbg_count(panel: Grid, bg: int) -> int:
    return sum(1 for row in panel for v in row if v != bg)


def panel_overlay(base: Grid, overlay: Grid, bg: int) -> Grid:
    """Overlay non-bg cells of overlay onto base"""
    h, w = grid_shape(base)
    result = [row[:] for row in base]
    for r in range(h):
        for c in range(w):
            if overlay[r][c] != bg:
                result[r][c] = overlay[r][c]
    return result


def panel_and(a: Grid, b: Grid, bg: int, fill: int) -> Grid:
    """AND: keep cells where both are non-bg"""
    h, w = grid_shape(a)
    return [[fill if a[r][c] != bg and b[r][c] != bg else bg
             for c in range(w)] for r in range(h)]


def panel_or(a: Grid, b: Grid, bg: int) -> Grid:
    """OR: keep cells where either is non-bg"""
    h, w = grid_shape(a)
    return [[a[r][c] if a[r][c] != bg else b[r][c] for c in range(w)]
            for r in range(h)]


def panel_xor(a: Grid, b: Grid, bg: int, fill: int) -> Grid:
    """XOR: keep cells where exactly one is non-bg"""
    h, w = grid_shape(a)
    return [[fill if (a[r][c] != bg) != (b[r][c] != bg) else bg
             for c in range(w)] for r in range(h)]


def learn_panel_transform(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Dict]:
    """Learn a panel-based transformation rule.
    
    Tries multiple strategies:
    1. Template overlay: one panel is the template, fill others
    2. Panel boolean ops: AND/OR/XOR of panel pairs → output panel
    3. Panel selection: output is one specific panel
    4. Panel-wise same transform: each panel transformed independently
    """
    # Check all inputs have same partition structure
    partitions = []
    for inp, out in train_pairs:
        p = detect_separators(inp)
        if p is None:
            return None
        partitions.append(p)
    
    # All must have same structure
    p0 = partitions[0]
    for p in partitions[1:]:
        if p['n_panels_r'] != p0['n_panels_r'] or p['n_panels_c'] != p0['n_panels_c']:
            return None
    
    nr, nc = p0['n_panels_r'], p0['n_panels_c']
    
    # Check if output has same partition
    out_partitions = []
    for inp, out in train_pairs:
        p_out = detect_separators(out)
        if p_out is None:
            # Output might not have separators (shrink to one panel)
            oh, ow = grid_shape(out)
            if oh == p0['panel_h'] and ow == p0['panel_w']:
                out_partitions.append('single_panel')
            else:
                return None
        else:
            if p_out['n_panels_r'] != nr or p_out['n_panels_c'] != nc:
                return None
            out_partitions.append(p_out)
    
    bg = p0['bg']
    
    # === Strategy: Output is single panel (shrink) ===
    if all(x == 'single_panel' for x in out_partitions):
        return _learn_panel_reduce(train_pairs, partitions, bg)
    
    # === Strategy: Same-partition output ===
    return _learn_panel_same_partition(train_pairs, partitions, out_partitions, bg)


def _learn_panel_reduce(train_pairs, partitions, bg):
    """Output is one panel — learn which panel or what combination"""
    nr, nc = partitions[0]['n_panels_r'], partitions[0]['n_panels_c']
    
    # Try: output = specific panel (i, j)
    for pi in range(nr):
        for pj in range(nc):
            consistent = True
            for (inp, out), part in zip(train_pairs, partitions):
                panels = extract_panels(inp, part)
                if not grid_eq(panels[pi][pj], out):
                    consistent = False
                    break
            if consistent:
                return {'type': 'select_panel', 'pi': pi, 'pj': pj}
    
    # Try: output = OR of all panels
    consistent = True
    for (inp, out), part in zip(train_pairs, partitions):
        panels = extract_panels(inp, part)
        result = panels[0][0]
        for pi in range(nr):
            for pj in range(nc):
                if pi == 0 and pj == 0:
                    continue
                result = panel_or(result, panels[pi][pj], bg)
        if not grid_eq(result, out):
            consistent = False
            break
    if consistent:
        return {'type': 'reduce_or'}
    
    # Try: output = AND of all panels (with most common non-bg fill)
    for fill_color in range(10):
        consistent = True
        for (inp, out), part in zip(train_pairs, partitions):
            panels = extract_panels(inp, part)
            result = panels[0][0]
            for pi in range(nr):
                for pj in range(nc):
                    if pi == 0 and pj == 0:
                        continue
                    result = panel_and(result, panels[pi][pj], bg, fill_color)
            if not grid_eq(result, out):
                consistent = False
                break
        if consistent:
            return {'type': 'reduce_and', 'fill': fill_color}
    
    # Try: output = XOR of all panels
    for fill_color in range(10):
        consistent = True
        for (inp, out), part in zip(train_pairs, partitions):
            panels = extract_panels(inp, part)
            ph, pw = grid_shape(panels[0][0])
            # Binary representation: nonbg count per cell
            counts = [[0]*pw for _ in range(ph)]
            for pi in range(nr):
                for pj in range(nc):
                    for r in range(ph):
                        for c in range(pw):
                            if panels[pi][pj][r][c] != bg:
                                counts[r][c] += 1
            # XOR = odd count
            result = [[fill_color if counts[r][c] % 2 == 1 else bg
                        for c in range(pw)] for r in range(ph)]
            if not grid_eq(result, out):
                consistent = False
                break
        if consistent:
            return {'type': 'reduce_xor', 'fill': fill_color}
    
    return None


def _learn_panel_same_partition(train_pairs, partitions, out_partitions, bg):
    """Output has same partition — learn per-panel or cross-panel transform"""
    nr, nc = partitions[0]['n_panels_r'], partitions[0]['n_panels_c']
    
    # Try: template panel overlay
    # Find which panel is the "template" (densest non-bg content)
    for ti in range(nr):
        for tj in range(nc):
            consistent = True
            for (inp, out), p_in, p_out in zip(train_pairs, partitions, out_partitions):
                if p_out == 'single_panel':
                    consistent = False
                    break
                in_panels = extract_panels(inp, p_in)
                out_panels = extract_panels(out, p_out)
                template = in_panels[ti][tj]
                
                for pi in range(nr):
                    for pj in range(nc):
                        if pi == ti and pj == tj:
                            # Template panel: should remain the same in output
                            if not grid_eq(in_panels[pi][pj], out_panels[pi][pj]):
                                consistent = False
                                break
                        else:
                            # Other panels: should be overlay of template + panel content
                            expected = panel_overlay(template, in_panels[pi][pj], bg)
                            if not grid_eq(expected, out_panels[pi][pj]):
                                consistent = False
                                break
                    if not consistent:
                        break
                if not consistent:
                    break
            
            if consistent:
                return {'type': 'template_overlay', 'ti': ti, 'tj': tj}
    
    # Try: broadcast template to all panels (exact copy)
    for ti in range(nr):
        for tj in range(nc):
            consistent = True
            for (inp, out), p_in, p_out in zip(train_pairs, partitions, out_partitions):
                if p_out == 'single_panel':
                    consistent = False; break
                in_panels = extract_panels(inp, p_in)
                out_panels = extract_panels(out, p_out)
                template = in_panels[ti][tj]
                
                for pi in range(nr):
                    for pj in range(nc):
                        if not grid_eq(template, out_panels[pi][pj]):
                            consistent = False; break
                    if not consistent: break
                if not consistent: break
            
            if consistent:
                return {'type': 'broadcast_template', 'ti': ti, 'tj': tj}
    
    # Try: template recolored by each panel's unique color
    for ti in range(nr):
        for tj in range(nc):
            consistent = True
            for (inp, out), p_in, p_out in zip(train_pairs, partitions, out_partitions):
                if p_out == 'single_panel':
                    consistent = False; break
                in_panels = extract_panels(inp, p_in)
                out_panels = extract_panels(out, p_out)
                template = in_panels[ti][tj]
                ph, pw = grid_shape(template)
                
                template_color = None
                for r in range(ph):
                    for c in range(pw):
                        if template[r][c] != bg:
                            template_color = template[r][c]
                            break
                    if template_color is not None: break
                
                if template_color is None:
                    consistent = False; break
                
                for pi in range(nr):
                    for pj in range(nc):
                        out_p = out_panels[pi][pj]
                        # Find panel's output color
                        panel_color = None
                        for r in range(ph):
                            for c in range(pw):
                                if out_p[r][c] != bg:
                                    panel_color = out_p[r][c]
                                    break
                            if panel_color is not None: break
                        
                        if panel_color is None:
                            continue
                        
                        # Check: output panel = template recolored to panel_color
                        for r in range(ph):
                            for c in range(pw):
                                expected = panel_color if template[r][c] != bg else bg
                                if out_p[r][c] != expected:
                                    consistent = False; break
                            if not consistent: break
                        if not consistent: break
                    if not consistent: break
                if not consistent: break
            
            if consistent:
                return {'type': 'template_recolor', 'ti': ti, 'tj': tj}
    
    # Try: each output panel = template with panel's input content overlaid
    # (different from template_overlay: template fills where panel is bg)
    for ti in range(nr):
        for tj in range(nc):
            consistent = True
            for (inp, out), p_in, p_out in zip(train_pairs, partitions, out_partitions):
                if p_out == 'single_panel':
                    consistent = False; break
                in_panels = extract_panels(inp, p_in)
                out_panels = extract_panels(out, p_out)
                template = in_panels[ti][tj]
                
                for pi in range(nr):
                    for pj in range(nc):
                        # Expected: overlay panel content on top of template
                        expected = panel_overlay(template, in_panels[pi][pj], bg)
                        if not grid_eq(expected, out_panels[pi][pj]):
                            # Also try: panel content on template, then template colors where both non-bg
                            consistent = False; break
                    if not consistent: break
                if not consistent: break
            
            if consistent:
                return {'type': 'template_overlay', 'ti': ti, 'tj': tj}
    
    return None


def apply_panel_transform(inp: Grid, rule: Dict) -> Optional[Grid]:
    """Apply learned panel transform"""
    part = detect_separators(inp)
    if part is None:
        return None
    
    bg = part['bg']
    nr, nc = part['n_panels_r'], part['n_panels_c']
    panels = extract_panels(inp, part)
    
    rtype = rule['type']
    
    if rtype == 'select_panel':
        pi, pj = rule['pi'], rule['pj']
        if pi < nr and pj < nc:
            return panels[pi][pj]
        return None
    
    elif rtype == 'reduce_or':
        result = panels[0][0]
        for pi in range(nr):
            for pj in range(nc):
                if pi == 0 and pj == 0:
                    continue
                result = panel_or(result, panels[pi][pj], bg)
        return result
    
    elif rtype == 'reduce_and':
        result = panels[0][0]
        for pi in range(nr):
            for pj in range(nc):
                if pi == 0 and pj == 0:
                    continue
                result = panel_and(result, panels[pi][pj], bg, rule['fill'])
        return result
    
    elif rtype == 'reduce_xor':
        ph, pw = grid_shape(panels[0][0])
        counts = [[0]*pw for _ in range(ph)]
        for pi in range(nr):
            for pj in range(nc):
                for r in range(ph):
                    for c in range(pw):
                        if panels[pi][pj][r][c] != bg:
                            counts[r][c] += 1
        return [[rule['fill'] if counts[r][c] % 2 == 1 else bg
                 for c in range(pw)] for r in range(ph)]
    
    elif rtype == 'broadcast_template':
        ti, tj = rule['ti'], rule['tj']
        if ti >= nr or tj >= nc:
            return None
        template = panels[ti][tj]
        out_panels = [[template for _ in range(nc)] for _ in range(nr)]
        return reassemble(out_panels, part)
    
    elif rtype == 'template_recolor':
        ti, tj = rule['ti'], rule['tj']
        if ti >= nr or tj >= nc:
            return None
        template = panels[ti][tj]
        ph, pw = grid_shape(template)
        
        out_panels = [[None]*nc for _ in range(nr)]
        for pi in range(nr):
            for pj in range(nc):
                # Find this panel's dominant non-bg color
                panel = panels[pi][pj]
                colors = {}
                for r in range(ph):
                    for c in range(pw):
                        v = panel[r][c]
                        if v != bg:
                            colors[v] = colors.get(v, 0) + 1
                
                if colors:
                    panel_color = max(colors, key=colors.get)
                else:
                    # No content — use template color
                    for r in range(ph):
                        for c in range(pw):
                            if template[r][c] != bg:
                                panel_color = template[r][c]
                                break
                        else:
                            continue
                        break
                    else:
                        panel_color = bg
                
                out_panels[pi][pj] = [[panel_color if template[r][c] != bg else bg
                                        for c in range(pw)] for r in range(ph)]
        
        return reassemble(out_panels, part)
    
    elif rtype == 'template_overlay':
        ti, tj = rule['ti'], rule['tj']
        if ti >= nr or tj >= nc:
            return None
        template = panels[ti][tj]
        
        out_panels = [[None]*nc for _ in range(nr)]
        for pi in range(nr):
            for pj in range(nc):
                if pi == ti and pj == tj:
                    out_panels[pi][pj] = panels[pi][pj]
                else:
                    out_panels[pi][pj] = panel_overlay(template, panels[pi][pj], bg)
        
        return reassemble(out_panels, part)
    
    return None
