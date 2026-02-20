"""
Cross Structure Mapper

Expertを3次元Cross構造にマッピングして高効率探索を実現
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

from core.ir import Domain
from knowledge.expert_profiler import ExpertProfiler


class CrossStructureMapper:
    """
    Expertを3次元Cross構造にマッピング
    
    X軸: 抽象度 (0=具体的計算, 1=抽象的推論)
    Y軸: ドメイン (0=純粋数学, 1=応用)
    Z軸: 深さ (0=基礎, 1=研究レベル)
    """
    
    def __init__(self, profiler: ExpertProfiler):
        """
        Args:
            profiler: ExpertProfilerインスタンス
        """
        self.profiler = profiler
        self.cross_space = {}  # (layer, expert_id) -> (x, y, z)
        
        # ドメインの軸マッピング定義
        self._define_axis_mappings()
        
        print("[CROSS_MAPPER] Initialized")
    
    def _define_axis_mappings(self):
        """
        ドメインから座標軸へのマッピングを定義
        """
        # X軸: 抽象度
        self.abstract_domains = {
            Domain.LOGIC_PROPOSITIONAL: 1.0,
            Domain.LOGIC_MODAL: 1.0,
            Domain.LOGIC_FIRST_ORDER: 1.0,
            Domain.ALGEBRA: 0.7,
            Domain.NUMBER_THEORY: 0.6,
        }
        
        self.concrete_domains = {
            Domain.ARITHMETIC: 1.0,
            Domain.COMBINATORICS: 0.7,
            Domain.PROBABILITY: 0.5,
        }
        
        # Y軸: 応用度
        self.pure_math_domains = {
            Domain.NUMBER_THEORY: 1.0,
            Domain.ALGEBRA: 0.9,
            Domain.LOGIC_PROPOSITIONAL: 0.8,
            Domain.CALCULUS: 0.7,
        }
        
        self.applied_domains = {
            Domain.PHYSICS: 1.0,
            Domain.COMPUTER_SCIENCE: 0.9,
            Domain.CHEMISTRY: 0.8,
            Domain.GEOMETRY: 0.6,  # 応用幾何
        }
    
    def build_cross_structure(
        self,
        profiles: Dict[Tuple[int, int], Dict[Domain, float]]
    ):
        """
        全expertをCross構造にマッピング
        
        Args:
            profiles: プロファイル結果
                {(layer, expert_id): {Domain: score}}
        """
        print(f"[CROSS_MAPPER] Building cross structure for {len(profiles)} experts...")
        
        for (layer, expert_id), domain_scores in profiles.items():
            coords = self._compute_cross_coordinates(domain_scores, layer)
            self.cross_space[(layer, expert_id)] = coords
        
        print(f"[CROSS_MAPPER] Cross structure built: {len(self.cross_space)} experts mapped")
    
    def _compute_cross_coordinates(
        self,
        domain_scores: Dict[Domain, float],
        layer: int
    ) -> Tuple[float, float, float]:
        """
        ドメインスコアと層情報から3次元座標を計算
        
        Args:
            domain_scores: {Domain: score}
            layer: レイヤー番号 (0-60)
        
        Returns:
            (x, y, z) - 各軸0.0-1.0の範囲
        """
        # X軸: 抽象度
        x = self._compute_abstraction_score(domain_scores)
        
        # Y軸: 応用度
        y = self._compute_application_score(domain_scores)
        
        # Z軸: 深さ（レイヤー深度 + ドメイン深度）
        z = self._compute_depth_score(domain_scores, layer)
        
        return (x, y, z)
    
    def _compute_abstraction_score(self, domain_scores: Dict[Domain, float]) -> float:
        """
        抽象度スコアを計算
        
        Returns:
            0.0 (具体的) ~ 1.0 (抽象的)
        """
        abstract_score = 0.0
        concrete_score = 0.0
        
        # 抽象的ドメインのスコアを集計
        for domain, weight in self.abstract_domains.items():
            abstract_score += domain_scores.get(domain, 0.0) * weight
        
        # 具体的ドメインのスコアを集計
        for domain, weight in self.concrete_domains.items():
            concrete_score += domain_scores.get(domain, 0.0) * weight
        
        # 正規化: (-1, 1) → (0, 1)
        total = abstract_score + concrete_score
        if total > 0:
            raw_score = (abstract_score - concrete_score) / total
        else:
            raw_score = 0.0
        
        # [0, 1]にクリップ
        return np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0)
    
    def _compute_application_score(self, domain_scores: Dict[Domain, float]) -> float:
        """
        応用度スコアを計算
        
        Returns:
            0.0 (純粋数学) ~ 1.0 (応用)
        """
        pure_math_score = 0.0
        applied_score = 0.0
        
        # 純粋数学ドメインのスコアを集計
        for domain, weight in self.pure_math_domains.items():
            pure_math_score += domain_scores.get(domain, 0.0) * weight
        
        # 応用ドメインのスコアを集計
        for domain, weight in self.applied_domains.items():
            applied_score += domain_scores.get(domain, 0.0) * weight
        
        # 正規化
        total = pure_math_score + applied_score
        if total > 0:
            raw_score = (applied_score - pure_math_score) / total
        else:
            raw_score = 0.0
        
        return np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0)
    
    def _compute_depth_score(
        self,
        domain_scores: Dict[Domain, float],
        layer: int
    ) -> float:
        """
        知識の深さスコアを計算
        
        Returns:
            0.0 (基礎) ~ 1.0 (研究レベル)
        """
        # レイヤー深度の寄与（深い層ほど高度）
        layer_contribution = layer / 60.0
        
        # 高度なドメインの寄与
        advanced_domains = [
            Domain.NUMBER_THEORY,
            Domain.CALCULUS,
            Domain.LINEAR_ALGEBRA,
            Domain.LOGIC_MODAL
        ]
        
        domain_contribution = 0.0
        for domain in advanced_domains:
            domain_contribution += domain_scores.get(domain, 0.0)
        
        domain_contribution = min(domain_contribution, 1.0)
        
        # 加重平均（層60%, ドメイン40%）
        depth_score = 0.6 * layer_contribution + 0.4 * domain_contribution
        
        return np.clip(depth_score, 0.0, 1.0)
    
    def search_nearest_experts(
        self,
        query_coords: Tuple[float, float, float],
        k: int = 5,
        search_mode: str = "euclidean"
    ) -> List[Tuple[int, int, float]]:
        """
        Cross構造で近傍expertを探索
        
        Args:
            query_coords: クエリの座標 (x, y, z)
            k: 返すexpert数
            search_mode: 距離メトリック
                - "euclidean": ユークリッド距離
                - "manhattan": マンハッタン距離
                - "cross": Cross構造特化（交差点優先）
        
        Returns:
            [(layer, expert_id, distance), ...] - 距離昇順
        """
        if search_mode == "cross":
            return self._cross_structure_search(query_coords, k)
        
        distances = []
        query = np.array(query_coords)
        
        for (layer, expert_id), coords in self.cross_space.items():
            coord_array = np.array(coords)
            
            if search_mode == "euclidean":
                dist = np.linalg.norm(query - coord_array)
            elif search_mode == "manhattan":
                dist = np.sum(np.abs(query - coord_array))
            else:
                dist = np.linalg.norm(query - coord_array)
            
            distances.append((layer, expert_id, dist))
        
        # 距離昇順でソート
        distances.sort(key=lambda x: x[2])
        
        return distances[:k]
    
    def _cross_structure_search(
        self,
        query_coords: Tuple[float, float, float],
        k: int
    ) -> List[Tuple[int, int, float]]:
        """
        Cross構造特化の探索
        
        戦略:
        1. 各軸で最も近いexpertを見つける
        2. 軸の交差点に近いexpertに高スコア
        3. 立体十字の中心に近いexpertを優先
        """
        qx, qy, qz = query_coords
        scored_experts = []
        
        for (layer, expert_id), (x, y, z) in self.cross_space.items():
            # 各軸での距離
            dx = abs(x - qx)
            dy = abs(y - qy)
            dz = abs(z - qz)
            
            # Cross構造スコア（交差点に近いほど高スコア）
            # 2軸が近い場合にボーナス
            cross_score = 0.0
            
            # XY平面の交差
            if dx < 0.1 and dy < 0.1:
                cross_score += 2.0
            
            # XZ平面の交差
            if dx < 0.1 and dz < 0.1:
                cross_score += 2.0
            
            # YZ平面の交差
            if dy < 0.1 and dz < 0.1:
                cross_score += 2.0
            
            # 3軸すべてが近い場合（中心点）
            if dx < 0.1 and dy < 0.1 and dz < 0.1:
                cross_score += 5.0
            
            # 総合距離（cross_scoreで調整）
            euclidean_dist = np.sqrt(dx**2 + dy**2 + dz**2)
            adjusted_dist = euclidean_dist / (1.0 + cross_score)
            
            scored_experts.append((layer, expert_id, adjusted_dist))
        
        # スコア昇順でソート
        scored_experts.sort(key=lambda x: x[2])
        
        return scored_experts[:k]
    
    def visualize_cross_space(
        self,
        output_file: str = "cross_space_3d.json",
        sample_size: Optional[int] = None
    ):
        """
        Cross構造を可視化用にエクスポート
        
        Args:
            output_file: 出力ファイル
            sample_size: サンプルサイズ（Noneで全expert）
        """
        data = {
            "experts": [],
            "axes": {
                "x": {"name": "Abstraction", "min": 0.0, "max": 1.0},
                "y": {"name": "Application", "min": 0.0, "max": 1.0},
                "z": {"name": "Depth", "min": 0.0, "max": 1.0}
            }
        }
        
        items = list(self.cross_space.items())
        if sample_size:
            import random
            items = random.sample(items, min(sample_size, len(items)))
        
        for (layer, expert_id), (x, y, z) in items:
            data["experts"].append({
                "layer": layer,
                "expert_id": expert_id,
                "identifier": f"L{layer}E{expert_id}",
                "coords": {"x": float(x), "y": float(y), "z": float(z)}
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[CROSS_MAPPER] Visualization data saved to {output_file}")
    
    def save_cross_structure(self, filepath: str):
        """Cross構造を保存"""
        data = {}
        for (layer, expert_id), coords in self.cross_space.items():
            key = f"L{layer}E{expert_id}"
            data[key] = {
                "layer": layer,
                "expert_id": expert_id,
                "coords": [float(c) for c in coords]
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[CROSS_MAPPER] Cross structure saved to {filepath}")
    
    def load_cross_structure(self, filepath: str):
        """Cross構造を読み込み"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.cross_space = {}
        for key, value in data.items():
            layer = value["layer"]
            expert_id = value["expert_id"]
            coords = tuple(value["coords"])
            self.cross_space[(layer, expert_id)] = coords
        
        print(f"[CROSS_MAPPER] Loaded {len(self.cross_space)} experts from {filepath}")
