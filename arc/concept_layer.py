"""
arc/concept_layer.py — 概念層: ConceptSignature + MoEルーティング

言語不要の不変量表現（ConceptSignature）を用いて、
各タスクの「概念的特徴」を抽出し、適切なモジュールにルーティングする。

概念:
  - structural: 対称性、パネル、フレーム、セパレータ
  - object: オブジェクト数、サイズ分布、形状タイプ
  - color: 色数、色分布、マーカー色
  - spatial: 密度、境界、位置関係
  - transform: 入出力のサイズ関係、差分パターン

MoEルーティング:
  - ConceptSignatureに基づいてモジュールを選択的に起動
  - 各モジュールのゲート関数が概念スコアを評価
  - top-kモジュールのみ実行（高速化）
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from scipy import ndimage

Grid = List[List[int]]


# ============================================================
# ConceptSignature: 言語不要の不変量表現
# ============================================================

@dataclass
class ConceptSignature:
    """タスクの概念的特徴ベクトル"""
    # Structural
    has_symmetry_h: float = 0.0
    has_symmetry_v: float = 0.0
    has_symmetry_rot: float = 0.0
    has_panels: float = 0.0  # セパレータで分割
    has_frame: float = 0.0
    has_separator: float = 0.0

    # Object
    n_objects: float = 0.0
    n_unique_shapes: float = 0.0
    has_uniform_objects: float = 0.0  # 全obj同サイズ
    has_nested_objects: float = 0.0  # 入れ子
    has_isolated_pixels: float = 0.0
    max_object_size: float = 0.0

    # Color
    n_colors: float = 0.0
    has_marker_color: float = 0.0  # 1つだけ異なる色
    has_color_gradient: float = 0.0
    color_distribution_entropy: float = 0.0

    # Spatial
    grid_density: float = 0.0
    has_enclosed_regions: float = 0.0
    has_border_objects: float = 0.0
    has_diagonal_pattern: float = 0.0

    # Transform (入出力関係)
    same_shape: float = 0.0
    output_smaller: float = 0.0
    output_larger: float = 0.0
    output_is_subgrid: float = 0.0
    is_recolor_only: float = 0.0
    cells_changed_ratio: float = 0.0
    shape_ratio: float = 0.0  # out_area / in_area

    def to_vector(self) -> np.ndarray:
        """特徴ベクトルに変換"""
        return np.array([getattr(self, f) for f in self.__dataclass_fields__])

    @property
    def field_names(self) -> List[str]:
        return list(self.__dataclass_fields__.keys())


def compute_concept_signature(train_pairs: List[Tuple[Grid, Grid]]) -> ConceptSignature:
    """訓練ペアからConceptSignatureを計算"""
    from arc.grid import most_common_color

    sig = ConceptSignature()

    if not train_pairs:
        return sig

    # Aggregate over all train pairs
    sigs = []
    for inp, out in train_pairs:
        s = _compute_single_pair_sig(inp, out)
        sigs.append(s)

    # Average
    for f in sig.__dataclass_fields__:
        vals = [getattr(s, f) for s in sigs]
        setattr(sig, f, float(np.mean(vals)))

    return sig


def _compute_single_pair_sig(inp: Grid, out: Grid) -> ConceptSignature:
    """単一ペアの概念シグネチャ"""
    from arc.grid import most_common_color

    sig = ConceptSignature()
    inp_arr = np.array(inp)
    out_arr = np.array(out)
    ih, iw = inp_arr.shape
    oh, ow = out_arr.shape
    bg = most_common_color(inp)

    # Structural
    sig.has_symmetry_h = 1.0 if np.array_equal(inp_arr, inp_arr[:, ::-1]) else 0.0
    sig.has_symmetry_v = 1.0 if np.array_equal(inp_arr, inp_arr[::-1, :]) else 0.0
    if ih == iw:
        sig.has_symmetry_rot = 1.0 if np.array_equal(inp_arr, np.rot90(inp_arr)) else 0.0

    # Separator detection
    for r in range(ih):
        if len(set(int(v) for v in inp_arr[r])) == 1 and int(inp_arr[r][0]) != bg:
            sig.has_separator = 1.0
            sig.has_panels = 1.0
            break
    for c in range(iw):
        if len(set(int(v) for v in inp_arr[:, c])) == 1 and int(inp_arr[0][c]) != bg:
            sig.has_separator = 1.0
            sig.has_panels = 1.0
            break

    # Frame
    border_colors = set()
    for r in range(ih):
        border_colors.add(int(inp_arr[r][0]))
        border_colors.add(int(inp_arr[r][iw - 1]))
    for c in range(iw):
        border_colors.add(int(inp_arr[0][c]))
        border_colors.add(int(inp_arr[ih - 1][c]))
    if len(border_colors) == 1 and list(border_colors)[0] != bg:
        sig.has_frame = 1.0

    # Object
    labeled, n_obj = ndimage.label(inp_arr != bg)
    sig.n_objects = float(n_obj)
    sizes = []
    shapes = set()
    for oid in range(1, n_obj + 1):
        mask = labeled == oid
        rows, cols = np.where(mask)
        sz = len(rows)
        sizes.append(sz)
        if sz == 1:
            sig.has_isolated_pixels = 1.0
        r0, c0 = rows.min(), cols.min()
        local = tuple(tuple(int(mask[r, c]) for c in range(c0, cols.max() + 1))
                       for r in range(r0, rows.max() + 1))
        shapes.add(local)

    sig.n_unique_shapes = float(len(shapes))
    sig.has_uniform_objects = 1.0 if len(set(sizes)) == 1 and n_obj > 1 else 0.0
    sig.max_object_size = float(max(sizes)) if sizes else 0.0

    # Nested objects (check if any bg region is enclosed)
    bg_mask = inp_arr == bg
    bg_labeled, bg_n = ndimage.label(bg_mask)
    border_bg = set()
    for r in range(ih):
        for c in range(iw):
            if (r == 0 or r == ih - 1 or c == 0 or c == iw - 1) and bg_mask[r, c]:
                border_bg.add(bg_labeled[r, c])
    sig.has_nested_objects = 1.0 if bg_n > len(border_bg) else 0.0
    sig.has_enclosed_regions = sig.has_nested_objects

    # Color
    non_bg = [int(v) for v in inp_arr.flatten() if v != bg]
    unique_colors = set(non_bg)
    sig.n_colors = float(len(unique_colors))
    if non_bg:
        counts = np.bincount(non_bg, minlength=10)
        probs = counts[counts > 0] / counts[counts > 0].sum()
        sig.color_distribution_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    # Marker color: one color appears much less than others
    if len(unique_colors) >= 2:
        color_counts = {c: non_bg.count(c) for c in unique_colors}
        min_count = min(color_counts.values())
        max_count = max(color_counts.values())
        if min_count < max_count * 0.1:
            sig.has_marker_color = 1.0

    # Spatial
    sig.grid_density = float(np.sum(inp_arr != bg)) / inp_arr.size
    border_objs = set()
    for r in range(ih):
        for c in range(iw):
            if (r == 0 or r == ih - 1 or c == 0 or c == iw - 1) and inp_arr[r, c] != bg:
                if labeled[r, c] > 0:
                    border_objs.add(labeled[r, c])
    sig.has_border_objects = 1.0 if border_objs else 0.0

    # Diagonal pattern
    diag_count = sum(1 for i in range(min(ih, iw)) if inp_arr[i, i] != bg)
    sig.has_diagonal_pattern = 1.0 if diag_count > min(ih, iw) * 0.5 else 0.0

    # Transform
    sig.same_shape = 1.0 if (ih, iw) == (oh, ow) else 0.0
    sig.output_smaller = 1.0 if oh * ow < ih * iw else 0.0
    sig.output_larger = 1.0 if oh * ow > ih * iw else 0.0
    sig.shape_ratio = float(oh * ow) / max(ih * iw, 1)

    if (ih, iw) == (oh, ow):
        diff = inp_arr != out_arr
        sig.cells_changed_ratio = float(np.sum(diff)) / max(inp_arr.size, 1)

        # Is it recolor only? (non-bg mask stays the same)
        inp_mask = inp_arr != bg
        out_bg = most_common_color(out)
        out_mask = out_arr != out_bg
        sig.is_recolor_only = 1.0 if np.array_equal(inp_mask, out_mask) else 0.0

    return sig


# ============================================================
# MoE Router: 概念スコアベースのモジュール選択
# ============================================================

@dataclass
class ModuleGate:
    """モジュールのゲート関数: 概念シグネチャからスコアを計算"""
    name: str
    weights: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0

    def score(self, sig: ConceptSignature) -> float:
        """ゲートスコア = Σ(weight_i * feature_i)"""
        s = 0.0
        for feat, w in self.weights.items():
            s += w * getattr(sig, feat, 0.0)
        return s

    def should_activate(self, sig: ConceptSignature) -> bool:
        return self.score(sig) > self.threshold


# Pre-defined gates for each module
MODULE_GATES = {
    'neighborhood_rule': ModuleGate(
        name='neighborhood_rule',
        weights={'same_shape': 2.0, 'cells_changed_ratio': 1.0, 'n_objects': -0.1},
        threshold=1.5,
    ),
    'tile_transform': ModuleGate(
        name='tile_transform',
        weights={'output_larger': 2.0, 'has_uniform_objects': 1.0},
        threshold=1.0,
    ),
    'extract_crop': ModuleGate(
        name='extract_crop',
        weights={'output_smaller': 2.0, 'n_objects': 0.3},
        threshold=1.5,
    ),
    'per_object': ModuleGate(
        name='per_object',
        weights={'same_shape': 1.0, 'n_objects': 0.5, 'is_recolor_only': 2.0},
        threshold=1.0,
    ),
    'symmetry': ModuleGate(
        name='symmetry',
        weights={'has_symmetry_h': 2.0, 'has_symmetry_v': 2.0, 'same_shape': 1.0},
        threshold=1.5,
    ),
    'panel_ops': ModuleGate(
        name='panel_ops',
        weights={'has_panels': 3.0, 'has_separator': 2.0},
        threshold=2.0,
    ),
    'puzzle_lang': ModuleGate(
        name='puzzle_lang',
        weights={},  # Always activate (highest priority)
        threshold=-1.0,
    ),
    'gravity': ModuleGate(
        name='gravity',
        weights={'same_shape': 1.0, 'has_isolated_pixels': 1.0, 'n_objects': 0.3},
        threshold=1.0,
    ),
    'holes_to_color': ModuleGate(
        name='holes_to_color',
        weights={'same_shape': 1.0, 'has_enclosed_regions': 3.0, 'is_recolor_only': 1.0},
        threshold=2.0,
    ),
    'composition': ModuleGate(
        name='composition',
        weights={},  # Always try as fallback
        threshold=-1.0,
    ),
    'program_tree': ModuleGate(
        name='program_tree',
        weights={},  # Always try (CEGIS-based)
        threshold=-1.0,
    ),
}


class MoERouter:
    """Mixture of Experts Router: 概念スコアに基づいてモジュールを選択"""

    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        self.gates = MODULE_GATES.copy()

    def route(self, sig: ConceptSignature) -> List[str]:
        """活性化するモジュール名をスコア降順で返す"""
        scored = [(name, gate.score(sig)) for name, gate in self.gates.items()
                  if gate.should_activate(sig)]
        scored.sort(key=lambda x: -x[1])
        return [name for name, score in scored[:self.top_k]]

    def route_with_scores(self, sig: ConceptSignature) -> List[Tuple[str, float]]:
        """スコア付きで返す"""
        scored = [(name, gate.score(sig)) for name, gate in self.gates.items()
                  if gate.should_activate(sig)]
        scored.sort(key=lambda x: -x[1])
        return scored[:self.top_k]


# ============================================================
# Concept Miner: 新規概念の自動発見
# ============================================================

class ConceptMiner:
    """
    訓練ペアから新しい概念を発見する。
    
    解けたタスクのConceptSignatureをクラスタリングし、
    共通パターンを新しい概念として抽出する。
    """

    def __init__(self):
        self.known_concepts: List[Tuple[str, ConceptSignature]] = []

    def register(self, name: str, sig: ConceptSignature):
        """解けたタスクの概念を登録"""
        self.known_concepts.append((name, sig))

    def find_similar(self, sig: ConceptSignature, top_k: int = 5) -> List[Tuple[str, float]]:
        """最も類似した既知概念を返す"""
        if not self.known_concepts:
            return []

        target = sig.to_vector()
        results = []
        for name, known_sig in self.known_concepts:
            known_vec = known_sig.to_vector()
            # Cosine similarity
            dot = np.dot(target, known_vec)
            norm = np.linalg.norm(target) * np.linalg.norm(known_vec)
            sim = float(dot / max(norm, 1e-10))
            results.append((name, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]
