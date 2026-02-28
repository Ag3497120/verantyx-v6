"""
arc/cross_parallel_engine.py — 並列Cross層エンジン (kofdai設計)

kofdaiの6大構想を実装:
1. 並列cross層: 複数サイズのcross構造を並列接続、接続面の節点数で測定
2. 段層的空間推論: 画像認識的な階層的特徴抽出
3. 3×3ベースcross → 動的ピース生成
4. 入れ子cross構造: 指数的空間拡大
5. 3D位置認識: crossによるグリッド上の3次元座標
6. オブジェクト→cross層マッピング: フローグラフ描画

核心思想: 局所近傍(NB)ではなく、crossの接続構造を通じた
動的な空間推論により、オブジェクトレベルの文脈を捉える。
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field

Grid = List[List[int]]


# ============================================================
# 1. Cross Node & Layer — 並列Cross層の基本単位
# ============================================================

@dataclass
class CrossNode:
    """Cross構造の1ノード (中心 + 4方向腕)"""
    row: int
    col: int
    size: int  # cross半径 (1=3x3, 2=5x5, ...)
    color: int  # このノードの中心色
    arm_colors: Dict[str, List[int]] = field(default_factory=dict)  # 方向→色列
    arm_lengths: Dict[str, int] = field(default_factory=dict)  # 実効腕長
    junction_count: int = 0  # 他crossとの接続面の節点数
    z_coord: float = 0.0  # 3D空間でのZ座標

    @property
    def feature_vector(self) -> tuple:
        """このノードの特徴ベクトル (ハッシュ可能)"""
        arms = []
        for d in ['up', 'down', 'left', 'right']:
            ac = self.arm_colors.get(d, [])
            arms.append(tuple(ac))
        return (self.color, self.size, self.junction_count, tuple(arms))


@dataclass
class CrossLayer:
    """同一サイズのcrossノードで構成される1層"""
    size: int  # cross半径
    nodes: List[CrossNode] = field(default_factory=list)
    node_map: Dict[Tuple[int, int], CrossNode] = field(default_factory=dict)
    
    def get_node(self, r: int, c: int) -> Optional[CrossNode]:
        return self.node_map.get((r, c))


@dataclass 
class NestedCross:
    """入れ子cross構造 — 大きいcrossの中に小さいcrossが入る"""
    outer: CrossNode
    inner_nodes: List[CrossNode] = field(default_factory=list)
    depth: int = 0  # 入れ子深さ


# ============================================================
# 2. Grid → CrossLayer マッピング
# ============================================================

def _extract_arm(grid: np.ndarray, r: int, c: int, dr: int, dc: int, 
                 max_len: int) -> Tuple[List[int], int]:
    """指定方向の腕を抽出。色列と実効長を返す"""
    h, w = grid.shape
    colors = []
    for step in range(1, max_len + 1):
        nr, nc = r + dr * step, c + dc * step
        if 0 <= nr < h and 0 <= nc < w:
            colors.append(int(grid[nr, nc]))
        else:
            break
    return colors, len(colors)


def build_cross_layer(grid: np.ndarray, size: int, bg: int = 0) -> CrossLayer:
    """グリッド全体にcrossノードを配置して1層を構築"""
    h, w = grid.shape
    layer = CrossLayer(size=size)
    
    directions = {
        'up': (-1, 0), 'down': (1, 0),
        'left': (0, -1), 'right': (0, 1)
    }
    
    for r in range(h):
        for c in range(w):
            node = CrossNode(row=r, col=c, size=size, color=int(grid[r, c]))
            
            for dname, (dr, dc) in directions.items():
                colors, length = _extract_arm(grid, r, c, dr, dc, size)
                node.arm_colors[dname] = colors
                node.arm_lengths[dname] = length
            
            # Z座標: 非bg色の近傍密度に基づく深さ
            local = grid[max(0, r-size):r+size+1, max(0, c-size):c+size+1]
            nonbg_ratio = np.sum(local != bg) / local.size if local.size > 0 else 0
            node.z_coord = nonbg_ratio
            
            layer.nodes.append(node)
            layer.node_map[(r, c)] = node
    
    return layer


def build_parallel_layers(grid: np.ndarray, sizes: List[int] = None,
                          bg: int = 0) -> List[CrossLayer]:
    """複数サイズの並列cross層を構築 (構想1)"""
    if sizes is None:
        h, w = grid.shape
        # 自動サイズ選択: 1, 2, min(h,w)//3
        sizes = [1, 2]
        third = min(h, w) // 3
        if third > 2:
            sizes.append(third)
    
    layers = []
    for s in sizes:
        layer = build_cross_layer(grid, s, bg)
        layers.append(layer)
    
    # 層間接続の節点数を計算 (構想1: 接続面の節点数)
    _compute_inter_layer_junctions(layers)
    
    return layers


def _compute_inter_layer_junctions(layers: List[CrossLayer]):
    """層間の接続面ノード数を計算"""
    if len(layers) < 2:
        return
    
    for i in range(len(layers) - 1):
        small_layer = layers[i]
        big_layer = layers[i + 1]
        
        for node in small_layer.nodes:
            pos = (node.row, node.col)
            big_node = big_layer.get_node(*pos)
            if big_node is not None:
                # 接続: 小さいcrossの特徴が大きいcrossの腕に含まれる数
                junction = 0
                for d in ['up', 'down', 'left', 'right']:
                    if node.arm_lengths.get(d, 0) > 0 and big_node.arm_lengths.get(d, 0) > 0:
                        junction += 1
                node.junction_count = junction
                big_node.junction_count = max(big_node.junction_count, junction)


# ============================================================
# 3. 段層的空間推論 (Hierarchical Spatial Reasoning)
# ============================================================

@dataclass
class SpatialFeature:
    """段層的に抽出された空間特徴"""
    row: int
    col: int
    level: int  # 0=pixel, 1=local, 2=region, 3=global
    features: Dict[str, Any] = field(default_factory=dict)


def extract_hierarchical_features(grid: np.ndarray, layers: List[CrossLayer],
                                   bg: int = 0) -> Dict[Tuple[int, int], List[SpatialFeature]]:
    """
    画像認識的な階層特徴抽出 (構想2)
    Level 0: ピクセル色
    Level 1: 3×3局所パターン
    Level 2: cross層の腕パターン（region）
    Level 3: グローバル統計
    """
    h, w = grid.shape
    features = {}
    
    # Global stats (Level 3)
    color_counts = Counter(int(v) for v in grid.flatten())
    total_cells = h * w
    global_feat = {
        'color_dist': {c: cnt / total_cells for c, cnt in color_counts.items()},
        'dominant': color_counts.most_common(1)[0][0],
        'n_colors': len(color_counts),
    }
    
    # Connected components for object context
    labeled, n_objs = _label_objects(grid, bg)
    obj_sizes = Counter(int(v) for v in labeled.flatten() if v > 0)
    
    for r in range(h):
        for c in range(w):
            cell_features = []
            
            # Level 0: pixel
            sf0 = SpatialFeature(r, c, 0, {
                'color': int(grid[r, c]),
                'is_bg': int(grid[r, c]) == bg,
                'obj_id': int(labeled[r, c]),
            })
            cell_features.append(sf0)
            
            # Level 1: local 3x3
            patch = grid[max(0, r-1):r+2, max(0, c-1):c+2]
            local_colors = Counter(int(v) for v in patch.flatten())
            n4_nonbg = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                          if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr, c+dc] != bg)
            sf1 = SpatialFeature(r, c, 1, {
                'n4_nonbg': n4_nonbg,
                'local_colors': len(local_colors),
                'is_edge': r == 0 or r == h-1 or c == 0 or c == w-1,
                'local_dominant': local_colors.most_common(1)[0][0],
            })
            cell_features.append(sf1)
            
            # Level 2: cross-layer features
            for li, layer in enumerate(layers):
                node = layer.get_node(r, c)
                if node is not None:
                    sf2 = SpatialFeature(r, c, 2, {
                        'layer_size': layer.size,
                        'arm_pattern': node.feature_vector,
                        'z_coord': node.z_coord,
                        'junction': node.junction_count,
                    })
                    cell_features.append(sf2)
            
            # Level 3: global context
            obj_id = int(labeled[r, c])
            sf3 = SpatialFeature(r, c, 3, {
                'global': global_feat,
                'obj_size': obj_sizes.get(obj_id, 0),
                'rel_pos': (r / max(h-1, 1), c / max(w-1, 1)),
            })
            cell_features.append(sf3)
            
            features[(r, c)] = cell_features
    
    return features


def _label_objects(grid: np.ndarray, bg: int) -> Tuple[np.ndarray, int]:
    """Connected components (4-connected, excluding bg)"""
    from scipy import ndimage
    nonbg = (grid != bg).astype(int)
    labeled, n = ndimage.label(nonbg)
    return labeled, n


# ============================================================
# 4. 3×3ベースcross → 動的ピース生成 (構想3)
# ============================================================

@dataclass
class CrossPattern:
    """3×3ベースcrossパターン — 学習された変換ルール"""
    input_pattern: tuple  # (center, up, down, left, right) の抽象パターン
    output_color: int     # 出力色
    context: Dict = field(default_factory=dict)  # 追加の文脈条件


def learn_cross_based_rules(train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                            layers_per_pair: List[List[CrossLayer]],
                            bg: int = 0) -> List[CrossPattern]:
    """
    3×3ベースcrossパターンから変換ルールを学習 (構想3)
    
    各セルについて:
    - 入力のcross特徴 → 出力色 のマッピングを収集
    - 一貫したパターンのみルールとして採用
    """
    pattern_to_outputs = defaultdict(list)
    
    for pair_idx, (inp, out) in enumerate(train_pairs):
        h, w = inp.shape
        layers = layers_per_pair[pair_idx]
        
        if len(layers) == 0:
            continue
        
        base_layer = layers[0]  # size=1のベース層
        
        for r in range(h):
            for c in range(w):
                if h != out.shape[0] or w != out.shape[1]:
                    break  # same_size only
                
                node = base_layer.get_node(r, c)
                if node is None:
                    continue
                
                # cross特徴をパターンキーに
                center = node.color
                up = node.arm_colors.get('up', [bg])[0] if node.arm_colors.get('up') else -1
                down = node.arm_colors.get('down', [bg])[0] if node.arm_colors.get('down') else -1
                left = node.arm_colors.get('left', [bg])[0] if node.arm_colors.get('left') else -1
                right = node.arm_colors.get('right', [bg])[0] if node.arm_colors.get('right') else -1
                
                # 抽象化: bg=0, self=1, other=2+
                abs_pattern = _abstract_cross_pattern(center, up, down, left, right, bg)
                
                out_color = int(out[r, c])
                pattern_to_outputs[abs_pattern].append(out_color)
    
    # 一貫したパターンのみ採用
    rules = []
    for pattern, outputs in pattern_to_outputs.items():
        if len(outputs) == 0:
            continue
        counter = Counter(outputs)
        most_common, count = counter.most_common(1)[0]
        if count == len(outputs):  # 100% consistent
            rules.append(CrossPattern(
                input_pattern=pattern,
                output_color=most_common,
            ))
    
    return rules


def _abstract_cross_pattern(center: int, up: int, down: int, left: int, right: int,
                            bg: int) -> tuple:
    """cross腕の色を抽象化"""
    def _abs(c, center_c):
        if c == -1:
            return -1  # OOB
        if c == bg:
            return 0  # background
        if c == center_c:
            return 1  # same as center
        return c + 10  # actual color (shifted to avoid collision)
    
    return (
        0 if center == bg else 1,  # center: bg or not
        _abs(up, center), _abs(down, center),
        _abs(left, center), _abs(right, center)
    )


# ============================================================
# 5. 入れ子cross構造 (構想4) — 指数的空間拡大
# ============================================================

def build_nested_crosses(grid: np.ndarray, layers: List[CrossLayer],
                         bg: int = 0, max_depth: int = 3) -> List[NestedCross]:
    """
    入れ子cross構造の構築 (構想4)
    大きいcrossの腕の中に小さいcrossが入る階層的構造
    
    同じ構造の複製/変化した構造を空間的に配置
    """
    h, w = grid.shape
    nested = []
    
    if len(layers) < 2:
        return nested
    
    outer_layer = layers[-1]  # 最大サイズの層
    inner_layer = layers[0]   # 最小サイズの層
    
    # 各外側crossノードについて、内部の小crossを収集
    for outer_node in outer_layer.nodes:
        r, c = outer_node.row, outer_node.col
        size = outer_layer.size
        
        # 外側crossの領域内の内側ノードを収集
        inner_nodes = []
        for dr in range(-size, size + 1):
            for dc in range(-size, size + 1):
                nr, nc = r + dr, c + dc
                inner_node = inner_layer.get_node(nr, nc)
                if inner_node is not None:
                    inner_nodes.append(inner_node)
        
        if inner_nodes:
            nc = NestedCross(
                outer=outer_node,
                inner_nodes=inner_nodes,
                depth=0
            )
            nested.append(nc)
    
    return nested


# ============================================================
# 6. 3D位置認識 (構想5) — crossによるグリッド座標の3次元化
# ============================================================

@dataclass
class Position3D:
    """crossに基づく3D座標"""
    x: float  # 水平位置
    y: float  # 垂直位置
    z: float  # 深さ (cross密度/オブジェクト所属)
    obj_id: int = -1
    layer_feature: tuple = ()


def compute_3d_positions(grid: np.ndarray, layers: List[CrossLayer],
                         bg: int = 0) -> Dict[Tuple[int, int], Position3D]:
    """
    crossの接続構造から3D位置を計算 (構想5)
    
    X, Y: グリッド座標（正規化）
    Z: cross接続密度 + オブジェクト帰属
    """
    h, w = grid.shape
    positions = {}
    
    labeled, n_objs = _label_objects(grid, bg)
    
    # オブジェクトの重心を計算
    obj_centroids = {}
    for obj_id in range(1, n_objs + 1):
        rows, cols = np.where(labeled == obj_id)
        if len(rows) > 0:
            obj_centroids[obj_id] = (rows.mean(), cols.mean())
    
    for r in range(h):
        for c in range(w):
            # X, Y: 正規化座標
            x = c / max(w - 1, 1)
            y = r / max(h - 1, 1)
            
            # Z: cross層の接続密度の加重平均
            z = 0.0
            for layer in layers:
                node = layer.get_node(r, c)
                if node:
                    z += node.z_coord * layer.size
            z /= sum(l.size for l in layers) if layers else 1
            
            obj_id = int(labeled[r, c])
            
            # 層特徴: 各層のarm pattern
            layer_feat = []
            for layer in layers:
                node = layer.get_node(r, c)
                if node:
                    layer_feat.append(node.junction_count)
            
            positions[(r, c)] = Position3D(
                x=x, y=y, z=z,
                obj_id=obj_id,
                layer_feature=tuple(layer_feat)
            )
    
    return positions


# ============================================================
# 7. フローグラフ描画 (構想6)
# ============================================================

@dataclass
class FlowEdge:
    """フローグラフの辺"""
    src: Tuple[int, int]
    dst: Tuple[int, int]
    weight: float
    flow_type: str  # 'color_spread', 'gravity', 'symmetry', 'proximity'


def build_flow_graph(grid: np.ndarray, layers: List[CrossLayer],
                     positions: Dict[Tuple[int, int], Position3D],
                     bg: int = 0) -> List[FlowEdge]:
    """
    オブジェクト特性からcross層上にフローグラフを描画 (構想6)
    
    フローの種類:
    - color_spread: 同色セル間の伝播
    - gravity: 非bgセルの重力方向
    - symmetry: 対称位置への参照
    - proximity: 最近接オブジェクトへの参照
    """
    h, w = grid.shape
    edges = []
    
    labeled, n_objs = _label_objects(grid, bg)
    
    # Color spread flow: 同色非bgの隣接セル
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg:
                continue
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == grid[r, c]:
                    edges.append(FlowEdge(
                        src=(r, c), dst=(nr, nc),
                        weight=1.0, flow_type='color_spread'
                    ))
    
    # Symmetry flow: 中心対称
    cr, cc = h / 2, w / 2
    for r in range(h):
        for c in range(w):
            sr, sc = int(2 * cr - r - 1), int(2 * cc - c - 1)
            if 0 <= sr < h and 0 <= sc < w and (sr, sc) != (r, c):
                if grid[r, c] != bg or grid[sr, sc] != bg:
                    edges.append(FlowEdge(
                        src=(r, c), dst=(sr, sc),
                        weight=0.5, flow_type='symmetry'
                    ))
    
    return edges


# ============================================================
# 8. 統合ソルバー — 全構想を組み合わせたルール学習・適用
# ============================================================

def learn_parallel_cross_rule(train_pairs: List[Tuple[Grid, Grid]], 
                              bg: int = 0) -> Optional[Dict]:
    """
    並列cross層を使った変換ルールの学習
    
    各trainペアから:
    1. 並列cross層構築 (構想1)
    2. 階層的特徴抽出 (構想2)
    3. 3×3ベースcrossパターン学習 (構想3)
    4. 3D位置コンテキスト (構想5)
    5. フローグラフ (構想6)
    
    これらを組み合わせて変換ルールを学習
    """
    # same_size check
    for inp, out in train_pairs:
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
    
    np_pairs = [(np.array(i), np.array(o)) for i, o in train_pairs]
    
    # Step 1: 並列cross層構築
    all_layers = []
    for inp, _ in np_pairs:
        layers = build_parallel_layers(inp, bg=bg)
        all_layers.append(layers)
    
    # Step 2: 3×3ベースcrossルール学習 (構想3)
    cross_rules = learn_cross_based_rules(np_pairs, all_layers, bg)
    
    if not cross_rules:
        return None
    
    # Step 3: 拡張ルール — 3D位置 + 階層特徴を使ったコンテキスト付きルール
    context_rules = _learn_context_enriched_rules(np_pairs, all_layers, bg)
    
    # Step 4: フローベースルール
    flow_rules = _learn_flow_rules(np_pairs, all_layers, bg)
    
    # ルールの検証: 全trainペアで正しいか
    rule_set = {
        'cross_rules': cross_rules,
        'context_rules': context_rules,
        'flow_rules': flow_rules,
        'bg': bg,
    }
    
    # 検証
    for inp, out in np_pairs:
        result = _apply_rule_set(inp, rule_set)
        if result is None or not np.array_equal(result, out):
            # ルールセットが不完全 — コンテキストルールで補完を試みる
            rule_set = _try_multi_layer_rules(np_pairs, all_layers, bg)
            if rule_set is None:
                return None
            # 再検証
            valid = True
            for inp2, out2 in np_pairs:
                result2 = _apply_rule_set(inp2, rule_set)
                if result2 is None or not np.array_equal(result2, out2):
                    valid = False
                    break
            if not valid:
                return None
            break
    
    return rule_set


def _learn_context_enriched_rules(np_pairs, all_layers, bg):
    """3D位置 + 階層特徴を使ったコンテキスト付きルール"""
    # key: (color, obj_size_bucket, rel_position_bucket, layer_features) → output_color
    pattern_map = defaultdict(list)
    
    for pair_idx, (inp, out) in enumerate(np_pairs):
        h, w = inp.shape
        layers = all_layers[pair_idx]
        positions = compute_3d_positions(inp, layers, bg)
        
        labeled, n_objs = _label_objects(inp, bg)
        # Object size buckets
        obj_sizes = {}
        for oid in range(1, n_objs + 1):
            obj_sizes[oid] = int(np.sum(labeled == oid))
        
        for r in range(h):
            for c in range(w):
                pos = positions.get((r, c))
                if pos is None:
                    continue
                
                # Size bucket: 0=bg, 1=small(<5), 2=medium(<20), 3=large
                obj_size = obj_sizes.get(pos.obj_id, 0)
                if obj_size == 0:
                    size_bucket = 0
                elif obj_size < 5:
                    size_bucket = 1
                elif obj_size < 20:
                    size_bucket = 2
                else:
                    size_bucket = 3
                
                # Position bucket: quadrant
                qr = 0 if pos.y < 0.5 else 1
                qc = 0 if pos.x < 0.5 else 1
                pos_bucket = qr * 2 + qc
                
                # Z bucket
                z_bucket = 0 if pos.z < 0.3 else (1 if pos.z < 0.7 else 2)
                
                key = (int(inp[r, c]), size_bucket, pos_bucket, z_bucket,
                       pos.layer_feature)
                pattern_map[key].append(int(out[r, c]))
    
    # Filter consistent rules
    rules = {}
    for key, outputs in pattern_map.items():
        counter = Counter(outputs)
        most_common, count = counter.most_common(1)[0]
        if count == len(outputs):
            rules[key] = most_common
    
    return rules


def _learn_flow_rules(np_pairs, all_layers, bg):
    """フローグラフに基づくルール学習"""
    # Flow-based: 各セルの出力色が、フロー接続先の入力色に依存
    flow_patterns = defaultdict(list)
    
    for pair_idx, (inp, out) in enumerate(np_pairs):
        h, w = inp.shape
        layers = all_layers[pair_idx]
        positions = compute_3d_positions(inp, layers, bg)
        flow_edges = build_flow_graph(inp, layers, positions, bg)
        
        # Build adjacency
        adj = defaultdict(list)
        for edge in flow_edges:
            adj[edge.src].append((edge.dst, edge.flow_type, edge.weight))
        
        for r in range(h):
            for c in range(w):
                in_color = int(inp[r, c])
                out_color = int(out[r, c])
                
                if in_color == out_color:
                    continue  # No change — skip
                
                # Flow context: what colors flow into this cell?
                neighbors = adj.get((r, c), [])
                flow_colors = sorted(set(int(inp[nr][nc]) for (nr, nc), _, _ in neighbors 
                                         if 0 <= nr < h and 0 <= nc < w))
                
                key = (in_color, tuple(flow_colors))
                flow_patterns[key].append(out_color)
    
    rules = {}
    for key, outputs in flow_patterns.items():
        counter = Counter(outputs)
        most_common, count = counter.most_common(1)[0]
        if count == len(outputs):
            rules[key] = most_common
    
    return rules


def _try_multi_layer_rules(np_pairs, all_layers, bg):
    """多層crossルール — より複雑な特徴の組み合わせ"""
    # Full feature key: (color, n4_pattern, cross_arm_abstract, obj_membership, position)
    pattern_map = defaultdict(list)
    
    for pair_idx, (inp, out) in enumerate(np_pairs):
        h, w = inp.shape
        layers = all_layers[pair_idx]
        
        labeled, n_objs = _label_objects(inp, bg)
        
        # Object properties
        obj_props = {}
        for oid in range(1, n_objs + 1):
            mask = (labeled == oid)
            area = int(mask.sum())
            rows, cols = np.where(mask)
            bbox_h = int(rows.max() - rows.min() + 1)
            bbox_w = int(cols.max() - cols.min() + 1)
            dominant = Counter(int(inp[r, c]) for r, c in zip(rows, cols)).most_common(1)[0][0]
            touches_border = bool(rows.min() == 0 or rows.max() == h-1 or 
                                  cols.min() == 0 or cols.max() == w-1)
            obj_props[oid] = (area, bbox_h, bbox_w, dominant, touches_border)
        
        for r in range(h):
            for c in range(w):
                color = int(inp[r, c])
                
                # N4 abstract pattern
                n4 = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        nc_val = int(inp[nr, nc])
                        if nc_val == bg:
                            n4.append(0)
                        elif nc_val == color:
                            n4.append(1)
                        else:
                            n4.append(2)
                    else:
                        n4.append(-1)
                n4_key = tuple(n4)
                
                # Object membership
                oid = int(labeled[r, c])
                if oid > 0 and oid in obj_props:
                    area, bh, bw, dom, tb = obj_props[oid]
                    # Abstract object features
                    area_bucket = 0 if area < 3 else (1 if area < 10 else (2 if area < 30 else 3))
                    aspect = 0 if bh == bw else (1 if bh > bw else 2)
                    obj_key = (area_bucket, aspect, tb)
                else:
                    obj_key = (-1, -1, False)
                
                # Cross arm abstract (size=1)
                if layers:
                    node = layers[0].get_node(r, c)
                    if node:
                        arm_abs = []
                        for d in ['up', 'down', 'left', 'right']:
                            ac = node.arm_colors.get(d, [])
                            if not ac:
                                arm_abs.append(-1)
                            elif ac[0] == bg:
                                arm_abs.append(0)
                            elif ac[0] == color:
                                arm_abs.append(1)
                            else:
                                arm_abs.append(2)
                        cross_key = tuple(arm_abs)
                    else:
                        cross_key = (0, 0, 0, 0)
                else:
                    cross_key = (0, 0, 0, 0)
                
                # Position features
                is_border = r == 0 or r == h-1 or c == 0 or c == w-1
                
                # Full key
                full_key = (color, n4_key, cross_key, obj_key, is_border)
                pattern_map[full_key].append(int(out[r, c]))
    
    # Filter consistent
    rules = {}
    for key, outputs in pattern_map.items():
        counter = Counter(outputs)
        most_common, count = counter.most_common(1)[0]
        if count == len(outputs):
            rules[key] = most_common
    
    if not rules:
        return None
    
    return {
        'multi_layer_rules': rules,
        'bg': bg,
        'type': 'multi_layer',
    }


def _apply_rule_set(grid: np.ndarray, rule_set: Dict) -> Optional[np.ndarray]:
    """ルールセットを適用してグリッドを変換"""
    bg = rule_set.get('bg', 0)
    h, w = grid.shape
    result = grid.copy()
    
    if 'multi_layer_rules' in rule_set:
        return _apply_multi_layer_rules(grid, rule_set)
    
    # Build layers
    layers = build_parallel_layers(grid, bg=bg)
    positions = compute_3d_positions(grid, layers, bg)
    
    labeled, n_objs = _label_objects(grid, bg)
    obj_sizes = {}
    for oid in range(1, n_objs + 1):
        obj_sizes[oid] = int(np.sum(labeled == oid))
    
    # Try cross rules first
    cross_rules = rule_set.get('cross_rules', [])
    context_rules = rule_set.get('context_rules', {})
    flow_rules = rule_set.get('flow_rules', {})
    
    # Build flow graph
    flow_edges = build_flow_graph(grid, layers, positions, bg)
    adj = defaultdict(list)
    for edge in flow_edges:
        adj[edge.src].append((edge.dst, edge.flow_type, edge.weight))
    
    applied = np.zeros((h, w), dtype=bool)
    
    # Apply context rules (most specific)
    for r in range(h):
        for c in range(w):
            pos = positions.get((r, c))
            if pos is None:
                continue
            
            obj_size = obj_sizes.get(pos.obj_id, 0)
            if obj_size == 0:
                size_bucket = 0
            elif obj_size < 5:
                size_bucket = 1
            elif obj_size < 20:
                size_bucket = 2
            else:
                size_bucket = 3
            
            qr = 0 if pos.y < 0.5 else 1
            qc = 0 if pos.x < 0.5 else 1
            pos_bucket = qr * 2 + qc
            z_bucket = 0 if pos.z < 0.3 else (1 if pos.z < 0.7 else 2)
            
            key = (int(grid[r, c]), size_bucket, pos_bucket, z_bucket,
                   pos.layer_feature)
            if key in context_rules:
                result[r, c] = context_rules[key]
                applied[r, c] = True
    
    # Apply cross pattern rules (for unapplied cells)
    for r in range(h):
        for c in range(w):
            if applied[r, c]:
                continue
            
            center = int(grid[r, c])
            node = layers[0].get_node(r, c) if layers else None
            if node is None:
                continue
            
            up = node.arm_colors.get('up', [bg])[0] if node.arm_colors.get('up') else -1
            down = node.arm_colors.get('down', [bg])[0] if node.arm_colors.get('down') else -1
            left = node.arm_colors.get('left', [bg])[0] if node.arm_colors.get('left') else -1
            right = node.arm_colors.get('right', [bg])[0] if node.arm_colors.get('right') else -1
            
            abs_pattern = _abstract_cross_pattern(center, up, down, left, right, bg)
            
            for rule in cross_rules:
                if rule.input_pattern == abs_pattern:
                    result[r, c] = rule.output_color
                    applied[r, c] = True
                    break
    
    # Apply flow rules (for remaining cells that changed)
    for r in range(h):
        for c in range(w):
            if applied[r, c]:
                continue
            
            in_color = int(grid[r, c])
            neighbors = adj.get((r, c), [])
            flow_colors = sorted(set(int(grid[nr][nc]) for (nr, nc), _, _ in neighbors
                                     if 0 <= nr < h and 0 <= nc < w))
            key = (in_color, tuple(flow_colors))
            if key in flow_rules:
                result[r, c] = flow_rules[key]
                applied[r, c] = True
    
    return result


def _apply_multi_layer_rules(grid: np.ndarray, rule_set: Dict) -> Optional[np.ndarray]:
    """多層crossルールの適用"""
    bg = rule_set.get('bg', 0)
    rules = rule_set.get('multi_layer_rules', {})
    h, w = grid.shape
    result = grid.copy()
    
    layers = build_parallel_layers(grid, bg=bg)
    labeled, n_objs = _label_objects(grid, bg)
    
    obj_props = {}
    for oid in range(1, n_objs + 1):
        mask = (labeled == oid)
        area = int(mask.sum())
        rows, cols = np.where(mask)
        bbox_h = int(rows.max() - rows.min() + 1)
        bbox_w = int(cols.max() - cols.min() + 1)
        dominant = Counter(int(grid[r, c]) for r, c in zip(rows, cols)).most_common(1)[0][0]
        touches_border = bool(rows.min() == 0 or rows.max() == h-1 or
                              cols.min() == 0 or cols.max() == w-1)
        obj_props[oid] = (area, bbox_h, bbox_w, dominant, touches_border)
    
    all_applied = True
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            
            n4 = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nc_val = int(grid[nr, nc])
                    if nc_val == bg:
                        n4.append(0)
                    elif nc_val == color:
                        n4.append(1)
                    else:
                        n4.append(2)
                else:
                    n4.append(-1)
            n4_key = tuple(n4)
            
            oid = int(labeled[r, c])
            if oid > 0 and oid in obj_props:
                area, bh, bw, dom, tb = obj_props[oid]
                area_bucket = 0 if area < 3 else (1 if area < 10 else (2 if area < 30 else 3))
                aspect = 0 if bh == bw else (1 if bh > bw else 2)
                obj_key = (area_bucket, aspect, tb)
            else:
                obj_key = (-1, -1, False)
            
            if layers:
                node = layers[0].get_node(r, c)
                if node:
                    arm_abs = []
                    for d in ['up', 'down', 'left', 'right']:
                        ac = node.arm_colors.get(d, [])
                        if not ac:
                            arm_abs.append(-1)
                        elif ac[0] == bg:
                            arm_abs.append(0)
                        elif ac[0] == color:
                            arm_abs.append(1)
                        else:
                            arm_abs.append(2)
                    cross_key = tuple(arm_abs)
                else:
                    cross_key = (0, 0, 0, 0)
            else:
                cross_key = (0, 0, 0, 0)
            
            is_border = r == 0 or r == h-1 or c == 0 or c == w-1
            
            full_key = (color, n4_key, cross_key, obj_key, is_border)
            if full_key in rules:
                result[r, c] = rules[full_key]
            else:
                all_applied = False
    
    if not all_applied:
        # Fallback: 未知パターンのセルは入力色を維持
        # ただし、変更セル/未変更セルの比率が適切な場合のみ
        changed = np.sum(result != grid)
        total = h * w
        if changed == 0:
            return None  # 何も変わらない = ルール無効
        return result
    
    return result


# ============================================================
# 9. CrossPiece生成 — cross_engine統合用
# ============================================================

def _loo_validate(train_pairs, bg, rule_set) -> bool:
    """Leave-one-out validation: learn on N-1 pairs, test on held-out pair"""
    if len(train_pairs) < 2:
        return True  # Need at least 2 for LOO
    
    abstraction = rule_set.get('abstraction')
    
    for hold_idx in range(len(train_pairs)):
        subset = [p for i, p in enumerate(train_pairs) if i != hold_idx]
        held_inp, held_out = train_pairs[hold_idx]
        
        if abstraction:
            sub_rule = _learn_abstracted_cross_rule(subset, bg, abstraction)
        else:
            sub_rule = learn_parallel_cross_rule(subset, bg)
        
        if sub_rule is None:
            return False
        
        result = apply_parallel_cross(held_inp, sub_rule)
        if result is None:
            return False
        
        from arc.grid import grid_eq
        if not grid_eq(result, held_out):
            return False
    
    return True


def apply_parallel_cross(inp: Grid, rule_set: Dict) -> Optional[Grid]:
    """CrossPiece用のapply関数"""
    grid = np.array(inp)
    abstraction = rule_set.get('abstraction')
    if abstraction:
        result = _apply_abstracted_rule(grid, rule_set, abstraction)
    else:
        result = _apply_rule_set(grid, rule_set)
    if result is None:
        return None
    return result.tolist()


def generate_parallel_cross_pieces(train_pairs: List[Tuple[Grid, Grid]]) -> list:
    """cross_engineに統合するためのCrossPiece生成"""
    from arc.cross_engine import CrossPiece
    from arc.grid import most_common_color
    
    pieces = []
    
    if not train_pairs:
        return pieces
    
    bg = most_common_color(train_pairs[0][0])
    
    # Strategy 1: Full parallel cross rule (with LOO validation)
    rule_set = learn_parallel_cross_rule(train_pairs, bg)
    if rule_set is not None and _loo_validate(train_pairs, bg, rule_set):
        pieces.append(CrossPiece(
            'parallel_cross',
            lambda inp, _rs=rule_set: apply_parallel_cross(inp, _rs)
        ))
    
    # Strategy 2: Object-aware cross (異なる抽象度レベル)
    abstractions = [
        'obj_color', 'obj_topology', 'obj_relative',
        'obj_interior_dist', 'obj_size_relative', 'obj_neighbor_colors',
        'cross_arm_count', 'obj_enclosure',
    ]
    for abstraction in abstractions:
        rule_set = _learn_abstracted_cross_rule(train_pairs, bg, abstraction)
        if rule_set is not None and _loo_validate(train_pairs, bg, rule_set):
            pieces.append(CrossPiece(
                f'parallel_cross:{abstraction}',
                lambda inp, _rs=rule_set: apply_parallel_cross(inp, _rs)
            ))
    
    return pieces


def _learn_abstracted_cross_rule(train_pairs, bg, abstraction):
    """異なる抽象度レベルでのcrossルール学習"""
    # same_size check
    for inp, out in train_pairs:
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
    
    np_pairs = [(np.array(i), np.array(o)) for i, o in train_pairs]
    pattern_map = defaultdict(list)
    
    for inp, out in np_pairs:
        h, w = inp.shape
        labeled, n_objs = _label_objects(inp, bg)
        
        # Object properties
        obj_props = {}
        for oid in range(1, n_objs + 1):
            mask = (labeled == oid)
            area = int(mask.sum())
            rows, cols = np.where(mask)
            obj_props[oid] = {
                'area': area,
                'color': Counter(int(inp[r, c]) for r, c in zip(rows, cols)).most_common(1)[0][0],
                'bbox': (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
                'centroid': (rows.mean(), cols.mean()),
                'border': bool(rows.min() == 0 or rows.max() == h-1 or 
                               cols.min() == 0 or cols.max() == w-1),
            }
        
        for r in range(h):
            for c in range(w):
                color = int(inp[r, c])
                oid = int(labeled[r, c])
                
                # N4 abstract
                n4 = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        nv = int(inp[nr, nc])
                        n4.append(0 if nv == bg else (1 if nv == color else 2))
                    else:
                        n4.append(-1)
                
                if abstraction == 'obj_color':
                    # Key: (color, n4, obj_color, obj_area_bucket)
                    if oid > 0 and oid in obj_props:
                        oc = obj_props[oid]['color']
                        oa = obj_props[oid]['area']
                        ab = 0 if oa < 5 else (1 if oa < 20 else 2)
                        key = (color, tuple(n4), oc, ab)
                    else:
                        key = (color, tuple(n4), -1, -1)
                
                elif abstraction == 'obj_topology':
                    # Key: (is_bg, n4_count_nonbg, is_border_cell, in_object, obj_border)
                    n4_nonbg = sum(1 for x in n4 if x > 0)
                    is_border = r == 0 or r == h-1 or c == 0 or c == w-1
                    if oid > 0 and oid in obj_props:
                        obj_b = obj_props[oid]['border']
                        # Interior vs edge of object
                        in_same_obj = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                          if 0 <= r+dr < h and 0 <= c+dc < w 
                                          and labeled[r+dr, c+dc] == oid)
                        obj_interior = in_same_obj == sum(1 for x in n4 if x >= 0)
                        key = (color != bg, n4_nonbg, is_border, True, obj_b, obj_interior)
                    else:
                        key = (color != bg, n4_nonbg, is_border, False, False, False)
                
                elif abstraction == 'obj_relative':
                    if oid > 0 and oid in obj_props:
                        cr, cc = obj_props[oid]['centroid']
                        rel_r = 0 if r < cr - 0.5 else (2 if r > cr + 0.5 else 1)
                        rel_c = 0 if c < cc - 0.5 else (2 if c > cc + 0.5 else 1)
                        key = (color, rel_r, rel_c, tuple(n4))
                    else:
                        key = (color, -1, -1, tuple(n4))
                
                elif abstraction == 'obj_interior_dist':
                    if oid > 0 and oid in obj_props:
                        same_obj_n4 = sum(1 for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]
                                          if 0 <= r+dr2 < h and 0 <= c+dc2 < w 
                                          and labeled[r+dr2, c+dc2] == oid)
                        total_n4 = sum(1 for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]
                                       if 0 <= r+dr2 < h and 0 <= c+dc2 < w)
                        is_interior = same_obj_n4 == total_n4
                        is_edge_of_obj = same_obj_n4 < total_n4
                        key = (color, is_interior, is_edge_of_obj, tuple(n4))
                    else:
                        key = (color, False, False, tuple(n4))
                
                elif abstraction == 'obj_size_relative':
                    if oid > 0 and oid in obj_props:
                        area = obj_props[oid]['area']
                        grid_area = h * w
                        ratio = area / grid_area
                        size_cat = 0 if ratio < 0.05 else (1 if ratio < 0.15 else (2 if ratio < 0.4 else 3))
                        key = (color, size_cat, tuple(n4))
                    else:
                        key = (color, -1, tuple(n4))
                
                elif abstraction == 'obj_neighbor_colors':
                    neighbor_obj_colors = set()
                    for dr2 in range(-2, 3):
                        for dc2 in range(-2, 3):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            nr2, nc2 = r + dr2, c + dc2
                            if 0 <= nr2 < h and 0 <= nc2 < w:
                                noid = int(labeled[nr2, nc2])
                                if noid > 0 and noid != oid and noid in obj_props:
                                    neighbor_obj_colors.add(obj_props[noid]['color'])
                    key = (color, tuple(sorted(neighbor_obj_colors)), tuple(n4))
                
                elif abstraction == 'cross_arm_count':
                    arm_counts = []
                    for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                        count = 0
                        for step in range(1, 4):
                            nr2, nc2 = r + dr2*step, c + dc2*step
                            if 0 <= nr2 < h and 0 <= nc2 < w and inp[nr2, nc2] != bg:
                                count += 1
                            else:
                                break
                        arm_counts.append(count)
                    key = (color, tuple(arm_counts))
                
                elif abstraction == 'obj_enclosure':
                    is_bg_cell = (color == bg)
                    if is_bg_cell:
                        all_nonbg_n4 = all(x > 0 for x in n4 if x >= 0)
                        surrounding_oids = set()
                        for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr2, nc2 = r + dr2, c + dc2
                            if 0 <= nr2 < h and 0 <= nc2 < w:
                                noid = int(labeled[nr2, nc2])
                                if noid > 0:
                                    surrounding_oids.add(noid)
                        key = (True, all_nonbg_n4, len(surrounding_oids), tuple(n4))
                    else:
                        key = (False, False, 0, tuple(n4))
                
                else:
                    continue
                
                pattern_map[key].append(int(out[r, c]))
    
    # Filter consistent
    rules = {}
    for key, outputs in pattern_map.items():
        counter = Counter(outputs)
        most_common, count = counter.most_common(1)[0]
        if count == len(outputs):
            rules[key] = most_common
    
    if not rules:
        return None
    
    # Verify on train
    rule_set = {
        'multi_layer_rules': rules,
        'bg': bg,
        'type': 'multi_layer',
        'abstraction': abstraction,
    }
    
    for inp, out in np_pairs:
        result = _apply_abstracted_rule(inp, rule_set, abstraction)
        if result is None or not np.array_equal(result, out):
            return None
    
    return rule_set


def _apply_abstracted_rule(grid: np.ndarray, rule_set: Dict, abstraction: str) -> Optional[np.ndarray]:
    """抽象化ルールの適用"""
    bg = rule_set.get('bg', 0)
    rules = rule_set.get('multi_layer_rules', {})
    h, w = grid.shape
    result = grid.copy()
    
    labeled, n_objs = _label_objects(grid, bg)
    
    obj_props = {}
    for oid in range(1, n_objs + 1):
        mask = (labeled == oid)
        area = int(mask.sum())
        rows, cols = np.where(mask)
        obj_props[oid] = {
            'area': area,
            'color': Counter(int(grid[r, c]) for r, c in zip(rows, cols)).most_common(1)[0][0],
            'centroid': (rows.mean(), cols.mean()),
            'border': bool(rows.min() == 0 or rows.max() == h-1 or
                           cols.min() == 0 or cols.max() == w-1),
        }
    
    all_applied = True
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            oid = int(labeled[r, c])
            
            n4 = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nv = int(grid[nr, nc])
                    n4.append(0 if nv == bg else (1 if nv == color else 2))
                else:
                    n4.append(-1)
            
            if abstraction == 'obj_color':
                if oid > 0 and oid in obj_props:
                    oc = obj_props[oid]['color']
                    oa = obj_props[oid]['area']
                    ab = 0 if oa < 5 else (1 if oa < 20 else 2)
                    key = (color, tuple(n4), oc, ab)
                else:
                    key = (color, tuple(n4), -1, -1)
            
            elif abstraction == 'obj_topology':
                n4_nonbg = sum(1 for x in n4 if x > 0)
                is_border = r == 0 or r == h-1 or c == 0 or c == w-1
                if oid > 0 and oid in obj_props:
                    obj_b = obj_props[oid]['border']
                    in_same_obj = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                      if 0 <= r+dr < h and 0 <= c+dc < w 
                                      and labeled[r+dr, c+dc] == oid)
                    obj_interior = in_same_obj == sum(1 for x in n4 if x >= 0)
                    key = (color != bg, n4_nonbg, is_border, True, obj_b, obj_interior)
                else:
                    key = (color != bg, n4_nonbg, is_border, False, False, False)
            
            elif abstraction == 'obj_relative':
                if oid > 0 and oid in obj_props:
                    cr, cc = obj_props[oid]['centroid']
                    rel_r = 0 if r < cr - 0.5 else (2 if r > cr + 0.5 else 1)
                    rel_c = 0 if c < cc - 0.5 else (2 if c > cc + 0.5 else 1)
                    key = (color, rel_r, rel_c, tuple(n4))
                else:
                    key = (color, -1, -1, tuple(n4))
            
            elif abstraction == 'obj_interior_dist':
                if oid > 0 and oid in obj_props:
                    same_obj_n4 = sum(1 for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]
                                      if 0 <= r+dr2 < h and 0 <= c+dc2 < w 
                                      and labeled[r+dr2, c+dc2] == oid)
                    total_n4 = sum(1 for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]
                                   if 0 <= r+dr2 < h and 0 <= c+dc2 < w)
                    is_interior = same_obj_n4 == total_n4
                    is_edge_of_obj = same_obj_n4 < total_n4
                    key = (color, is_interior, is_edge_of_obj, tuple(n4))
                else:
                    key = (color, False, False, tuple(n4))
            
            elif abstraction == 'obj_size_relative':
                if oid > 0 and oid in obj_props:
                    area = obj_props[oid]['area']
                    grid_area = h * w
                    ratio = area / grid_area
                    size_cat = 0 if ratio < 0.05 else (1 if ratio < 0.15 else (2 if ratio < 0.4 else 3))
                    key = (color, size_cat, tuple(n4))
                else:
                    key = (color, -1, tuple(n4))
            
            elif abstraction == 'obj_neighbor_colors':
                neighbor_obj_colors = set()
                for dr2 in range(-2, 3):
                    for dc2 in range(-2, 3):
                        if dr2 == 0 and dc2 == 0:
                            continue
                        nr2, nc2 = r + dr2, c + dc2
                        if 0 <= nr2 < h and 0 <= nc2 < w:
                            noid = int(labeled[nr2, nc2])
                            if noid > 0 and noid != oid and noid in obj_props:
                                neighbor_obj_colors.add(obj_props[noid]['color'])
                key = (color, tuple(sorted(neighbor_obj_colors)), tuple(n4))
            
            elif abstraction == 'cross_arm_count':
                arm_counts = []
                for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                    count = 0
                    for step in range(1, 4):
                        nr2, nc2 = r + dr2*step, c + dc2*step
                        if 0 <= nr2 < h and 0 <= nc2 < w and grid[nr2, nc2] != bg:
                            count += 1
                        else:
                            break
                    arm_counts.append(count)
                key = (color, tuple(arm_counts))
            
            elif abstraction == 'obj_enclosure':
                is_bg_cell = (color == bg)
                if is_bg_cell:
                    all_nonbg_n4 = all(x > 0 for x in n4 if x >= 0)
                    surrounding_oids = set()
                    for dr2, dc2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr2, nc2 = r + dr2, c + dc2
                        if 0 <= nr2 < h and 0 <= nc2 < w:
                            noid = int(labeled[nr2, nc2])
                            if noid > 0:
                                surrounding_oids.add(noid)
                    key = (True, all_nonbg_n4, len(surrounding_oids), tuple(n4))
                else:
                    key = (False, False, 0, tuple(n4))
            
            else:
                continue
            
            if key in rules:
                result[r, c] = rules[key]
            else:
                all_applied = False
    
    if not all_applied:
        changed = np.sum(result != grid)
        if changed == 0:
            return None
        return result
    
    return result
