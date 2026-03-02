"""
arc/hint_generator.py — Verantyx分析結果をLLMヒントに変換

Cross Engineの分析結果（構造特徴、partial match、検出パターン）を
自然言語ヒントに変換してsynthプロンプトに注入する。

Usage:
  from arc.hint_generator import generate_hints
  hints = generate_hints(task_data)
  # → {"structural": "...", "partial_matches": "...", "recommendations": "..."}
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
from arc.grid import Grid, grid_shape, grid_eq, most_common_color, grid_colors


def analyze_structure(train_pairs: List[Tuple[Grid, Grid]]) -> Dict[str, str]:
    """Phase 1: 構造特徴を分析してヒント文生成"""
    hints = []
    
    input_sizes = [grid_shape(inp) for inp, _ in train_pairs]
    output_sizes = [grid_shape(out) for _, out in train_pairs]
    
    # Size relationship
    same_size = all(is_ == os_ for is_, os_ in zip(input_sizes, output_sizes))
    if same_size:
        hints.append("The output grid is ALWAYS the same size as the input grid. This is an in-place transformation.")
    else:
        ratios_h = [os_[0] / max(is_[0], 1) for is_, os_ in zip(input_sizes, output_sizes)]
        ratios_w = [os_[1] / max(is_[1], 1) for is_, os_ in zip(input_sizes, output_sizes)]
        
        if all(r == ratios_h[0] for r in ratios_h) and all(r == ratios_w[0] for r in ratios_w):
            if ratios_h[0] > 1 or ratios_w[0] > 1:
                hints.append(f"Output is SCALED UP: height x{ratios_h[0]:.1f}, width x{ratios_w[0]:.1f}. Think: tiling, upscaling, repetition, border expansion.")
            elif ratios_h[0] < 1 or ratios_w[0] < 1:
                hints.append(f"Output is SMALLER: height x{ratios_h[0]:.1f}, width x{ratios_w[0]:.1f}. Think: cropping, extraction, summarization, downscaling.")
        
        const_output = len(set(output_sizes)) == 1
        if const_output:
            oh, ow = output_sizes[0]
            hints.append(f"Output is ALWAYS {oh}x{ow} regardless of input size. This suggests extraction or counting to a fixed template.")
    
    # Color analysis
    for i, (inp, out) in enumerate(train_pairs):
        inp_colors = grid_colors(inp)
        out_colors = grid_colors(out)
        new_colors = out_colors - inp_colors
        removed_colors = inp_colors - out_colors
        
        if i == 0:
            if new_colors:
                hints.append(f"New colors appear in output that aren't in input: {new_colors}. The rule CREATES new color values.")
            if removed_colors:
                hints.append(f"Some input colors are removed in output: {removed_colors}. The rule FILTERS or REPLACES colors.")
            if inp_colors == out_colors:
                hints.append("Input and output use the SAME set of colors. The rule rearranges/transforms without adding/removing colors.")
    
    # Symmetry detection
    for inp, out in train_pairs:
        g = np.array(inp)
        h, w = g.shape
        
        syms = []
        if np.array_equal(g, g[::-1, :]):
            syms.append("vertically symmetric")
        if np.array_equal(g, g[:, ::-1]):
            syms.append("horizontally symmetric")
        if h == w:
            if np.array_equal(g, g.T):
                syms.append("diagonally symmetric")
            if np.array_equal(g, np.rot90(g)):
                syms.append("rotationally symmetric (90°)")
        
        if syms:
            hints.append(f"Input grid has symmetry: {', '.join(syms)}. The transformation may exploit or create symmetry.")
        break  # Only check first example
    
    # Check if output is a simple transform of input
    for inp, out in train_pairs[:1]:
        gi = np.array(inp)
        go = np.array(out)
        if gi.shape == go.shape:
            diff_count = int(np.sum(gi != go))
            total = gi.size
            pct = diff_count / total * 100
            if pct < 10:
                hints.append(f"Only {pct:.0f}% of cells change. This is a MINIMAL local edit — look for local rules (neighborhood, flood fill).")
            elif pct > 90:
                hints.append(f"{pct:.0f}% of cells change. This is a GLOBAL transformation — think recoloring, tiling, complete restructure.")
        break
    
    # Background detection
    for inp, out in train_pairs[:1]:
        inp_flat = [c for row in inp for c in row]
        bg = Counter(inp_flat).most_common(1)[0][0]
        bg_pct = Counter(inp_flat)[bg] / len(inp_flat) * 100
        if bg_pct > 70:
            hints.append(f"Background color is {bg} ({bg_pct:.0f}% of input). Objects are non-{bg} cells.")
    
    # Object analysis
    for inp, out in train_pairs[:1]:
        g = np.array(inp)
        bg = Counter(g.flatten()).most_common(1)[0][0]
        from scipy.ndimage import label as scipy_label
        binary = (g != bg).astype(int)
        if binary.any():
            labeled, n_obj = scipy_label(binary)
            if n_obj == 1:
                hints.append("Input contains a SINGLE connected object. Think: extraction, transformation of that object.")
            elif 2 <= n_obj <= 5:
                hints.append(f"Input contains {n_obj} separate objects. Think: object matching, sorting, combining, or per-object operations.")
            elif n_obj > 5:
                hints.append(f"Input contains {n_obj} objects — many small objects. Think: counting, pattern detection, cellular automaton.")
            
            # Object sizes
            obj_sizes = [int(np.sum(labeled == i)) for i in range(1, n_obj + 1)]
            if len(set(obj_sizes)) == 1 and n_obj > 1:
                hints.append(f"All {n_obj} objects are the same size ({obj_sizes[0]} cells). Uniform objects suggest sorting by color or position.")
            elif n_obj > 1:
                hints.append(f"Object sizes vary: {sorted(obj_sizes)}. Different sizes may determine the transformation rule.")
        break
    
    return {"structural": " ".join(hints)}


def analyze_partial_matches(train_pairs: List[Tuple[Grid, Grid]]) -> Dict[str, str]:
    """Phase 2: Cross Engine partial match分析"""
    from arc.cross_engine import _generate_cross_pieces, CrossSimulator
    
    pieces = _generate_cross_pieces(train_pairs)
    sim = CrossSimulator()
    
    partial_results = []
    for piece in pieces:
        score = sim.partial_verify(piece, train_pairs)
        if score > 0.3:  # 30%以上マッチ
            partial_results.append((piece.name, score))
    
    partial_results.sort(key=lambda x: -x[1])
    
    if not partial_results:
        return {"partial_matches": ""}
    
    lines = ["PARTIAL MATCHES from Verantyx analysis (these transformations partially work):"]
    for name, score in partial_results[:5]:
        pct = score * 100
        lines.append(f"  - '{name}' matches {pct:.0f}% of output cells. The correct solution may be similar to or build upon this.")
    
    if partial_results[0][1] > 0.7:
        lines.append(f"NOTE: '{partial_results[0][0]}' is very close ({partial_results[0][1]*100:.0f}%). "
                     "The answer likely requires a small modification of this approach.")
    
    return {"partial_matches": "\n".join(lines)}


def analyze_quick_checks(train_pairs: List[Tuple[Grid, Grid]]) -> Dict[str, str]:
    """Phase 3: 高速チェック（flip, rotate, color map等）"""
    hints = []
    
    inp0, out0 = train_pairs[0]
    gi = np.array(inp0)
    go = np.array(out0)
    
    checks_passed = []
    
    # Simple transforms
    if gi.shape == go.shape:
        if np.array_equal(gi[::-1, :], go):
            checks_passed.append("vertical flip")
        if np.array_equal(gi[:, ::-1], go):
            checks_passed.append("horizontal flip")
        if np.array_equal(np.rot90(gi, 2), go):
            checks_passed.append("180° rotation")
        if gi.shape[0] == gi.shape[1]:
            if np.array_equal(np.rot90(gi, 1), go):
                checks_passed.append("90° CW rotation")
            if np.array_equal(np.rot90(gi, 3), go):
                checks_passed.append("90° CCW rotation")
            if np.array_equal(gi.T, go):
                checks_passed.append("transpose")
    
    # Color mapping check
    if gi.shape == go.shape:
        color_map = {}
        is_color_map = True
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                ic, oc = int(gi[r, c]), int(go[r, c])
                if ic in color_map:
                    if color_map[ic] != oc:
                        is_color_map = False
                        break
                else:
                    color_map[ic] = oc
            if not is_color_map:
                break
        
        if is_color_map and color_map:
            non_identity = {k: v for k, v in color_map.items() if k != v}
            if non_identity:
                checks_passed.append(f"color remapping: {non_identity}")
    
    if checks_passed:
        # Verify across all examples
        for check in checks_passed:
            hints.append(f"DETECTED (example 0): {check}. Verify this holds for all examples.")
    else:
        hints.append("No simple single-step transform detected (not a flip, rotation, or color map). The rule is more complex.")
    
    return {"quick_checks": " ".join(hints)}


def generate_hints(task_data: dict, include_partial: bool = True) -> str:
    """
    メインエントリ: ARCタスク → ヒントテキスト

    Returns:
        ヒントテキスト（synthプロンプトに注入する用）
    """
    train_pairs = [(ex['input'], ex['output']) for ex in task_data['train']]
    
    all_hints = {}
    
    # Phase 1: Structure analysis (fast, always run)
    all_hints.update(analyze_structure(train_pairs))
    
    # Phase 3: Quick checks (fast)
    all_hints.update(analyze_quick_checks(train_pairs))
    
    # Phase 2: Partial matches (slower, optional)
    if include_partial:
        try:
            all_hints.update(analyze_partial_matches(train_pairs))
        except Exception:
            all_hints["partial_matches"] = ""
    
    # Combine into final hint text
    sections = []
    
    if all_hints.get("structural"):
        sections.append(f"=== STRUCTURAL ANALYSIS ===\n{all_hints['structural']}")
    
    if all_hints.get("quick_checks"):
        sections.append(f"=== QUICK CHECKS ===\n{all_hints['quick_checks']}")
    
    if all_hints.get("partial_matches"):
        sections.append(all_hints["partial_matches"])
    
    return "\n\n".join(sections)


# ── CLI test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m arc.hint_generator <task.json> [--no-partial]")
        sys.exit(1)
    
    path = sys.argv[1]
    include_partial = "--no-partial" not in sys.argv
    
    with open(path) as f:
        task_data = json.load(f)
    
    hints = generate_hints(task_data, include_partial=include_partial)
    print(hints)
