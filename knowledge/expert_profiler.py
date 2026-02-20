"""
Expert Profiler

Expertの重みを分析してドメイン特性を推定

Phase 6 Enhancement: Added real routing analysis support via ExpertRouterAnalyzer.
The previous approach used FAKE hardcoded domain signatures. The new approach:

1. (Legacy) Weight-statistics-based profiling (original method, kept for compat)
2. (NEW) Routing-analysis-based profiling via ExpertRouterAnalyzer
   - Runs actual probe queries through the model
   - Records which experts fire for math vs physics vs chemistry etc.
   - Builds REAL domain → expert_id mapping from actual activation patterns

Usage:
    # Legacy weight-based profiling (no model needed):
    profiler = ExpertProfiler(weight_loader=loader)
    scores = profiler.profile_expert(layer=42, expert_id=187)

    # NEW: Routing-analysis-based profiling (requires model or stub):
    profiler = ExpertProfiler(weight_loader=loader)
    profiler.calibrate_from_routing(stub=True)  # stub=False for real model
    scores = profiler.profile_expert_routing(layer=42, expert_id=187)
"""
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import cosine

from core.ir import Domain
from knowledge.weight_loader import DeepSeekWeightLoader


# Lazy import to avoid circular dependency
_ExpertRouterAnalyzer = None

def _get_router_analyzer():
    global _ExpertRouterAnalyzer
    if _ExpertRouterAnalyzer is None:
        from knowledge.expert_router_analyzer import ExpertRouterAnalyzer
        _ExpertRouterAnalyzer = ExpertRouterAnalyzer
    return _ExpertRouterAnalyzer


# Map between routing domain strings and IR Domain enum
ROUTING_DOMAIN_MAP = {
    "math": [Domain.ARITHMETIC, Domain.ALGEBRA, Domain.NUMBER_THEORY,
             Domain.CALCULUS, Domain.LINEAR_ALGEBRA, Domain.COMBINATORICS],
    "physics": [Domain.PHYSICS],
    "computer_science": [Domain.COMPUTER_SCIENCE],
    "chemistry": [],
    "biology": [],
    "history": [],
    "literature": [],
    "philosophy": [Domain.LOGIC_PROPOSITIONAL],
}

# Reverse map: IR Domain → routing domain string
IR_TO_ROUTING = {}
for routing_d, ir_domains in ROUTING_DOMAIN_MAP.items():
    for ir_d in ir_domains:
        IR_TO_ROUTING[ir_d] = routing_d


class ExpertProfiler:
    """
    Expertの知識領域をプロファイリング
    
    方法:
    1. 重み行列の統計的特性を抽出
    2. ドメインシグネチャと比較
    3. expertのドメイン特性スコアを計算
    """
    
    def __init__(
        self,
        weight_loader: DeepSeekWeightLoader,
        signature_file: Optional[str] = None
    ):
        """
        Args:
            weight_loader: WeightLoaderインスタンス
            signature_file: 事前計算済みシグネチャファイル（オプション）
        """
        self.weight_loader = weight_loader
        self.domain_signatures = {}
        
        if signature_file:
            self.load_signatures(signature_file)
        else:
            # デフォルトシグネチャ（仮の値）
            self._init_default_signatures()
        
        print(f"[EXPERT_PROFILER] Initialized with {len(self.domain_signatures)} domain signatures")
    
    def _init_default_signatures(self):
        """
        デフォルトのドメインシグネチャを初期化
        
        注: これは仮の値。実際にはプロファイリングで生成
        """
        # 各ドメインの特徴ベクトル（11次元）
        # [mean, std, skew, kurtosis, max_singular, sum_singular, condition, sparsity, rank_ratio, frobenius, spectral_entropy]
        
        self.domain_signatures = {
            Domain.ARITHMETIC: np.array([
                0.0, 0.05, 0.1, 3.0, 50.0, 200.0, 10.0, 0.1, 0.9, 100.0, 0.8
            ]),
            Domain.ALGEBRA: np.array([
                0.0, 0.06, -0.1, 2.8, 55.0, 220.0, 12.0, 0.15, 0.85, 110.0, 0.82
            ]),
            Domain.NUMBER_THEORY: np.array([
                0.0, 0.07, 0.2, 3.2, 60.0, 240.0, 15.0, 0.12, 0.88, 120.0, 0.85
            ]),
            Domain.CALCULUS: np.array([
                0.0, 0.08, 0.0, 2.5, 70.0, 280.0, 20.0, 0.08, 0.92, 140.0, 0.9
            ]),
            Domain.LINEAR_ALGEBRA: np.array([
                0.0, 0.05, -0.2, 2.9, 65.0, 260.0, 18.0, 0.1, 0.9, 130.0, 0.88
            ]),
            Domain.LOGIC_PROPOSITIONAL: np.array([
                0.0, 0.04, 0.3, 4.0, 45.0, 180.0, 8.0, 0.2, 0.8, 90.0, 0.75
            ]),
            Domain.PROBABILITY: np.array([
                0.0, 0.06, 0.1, 2.7, 58.0, 230.0, 13.0, 0.11, 0.87, 115.0, 0.83
            ]),
            Domain.GEOMETRY: np.array([
                0.0, 0.07, -0.1, 2.6, 62.0, 250.0, 16.0, 0.09, 0.91, 125.0, 0.86
            ]),
        }
    
    def profile_expert(
        self,
        layer: int,
        expert_id: int,
        components: List[str] = ["gate_proj", "up_proj"]
    ) -> Dict[Domain, float]:
        """
        Expertのドメイン特性スコアを計算
        
        Args:
            layer: レイヤー番号
            expert_id: Expert ID
            components: 分析する重みコンポーネント
        
        Returns:
            {Domain: score} - 各ドメインとの類似度スコア
        """
        # 複数コンポーネントの重みを結合
        weights = []
        for comp in components:
            W = self.weight_loader.load_expert_weights(layer, expert_id, comp)
            weights.append(W)
        
        # 平均特徴を計算
        features_list = [self._extract_weight_features(W) for W in weights]
        avg_features = np.mean(features_list, axis=0)
        
        # 各ドメインとの類似度を計算
        domain_scores = {}
        for domain, signature in self.domain_signatures.items():
            # コサイン類似度（1 - cosine距離）
            similarity = 1.0 - cosine(avg_features, signature)
            domain_scores[domain] = max(0.0, similarity)  # 負の値は0に
        
        return domain_scores
    
    def _extract_weight_features(self, W: np.ndarray) -> np.ndarray:
        """
        重み行列から特徴ベクトルを抽出
        
        特徴（11次元）:
        1-4: 基本統計量（mean, std, skew, kurtosis）
        5-7: スペクトル特性（max_singular, sum_singular, condition）
        8: スパース性
        9: ランク比率
        10: Frobeniusノルム
        11: スペクトルエントロピー
        
        Args:
            W: 重み行列 (shape: [in_dim, out_dim])
        
        Returns:
            特徴ベクトル (11次元)
        """
        features = []
        
        # 1-4: 基本統計量
        flat = W.flatten()
        features.append(np.mean(flat))
        features.append(np.std(flat))
        features.append(stats.skew(flat))
        features.append(stats.kurtosis(flat))
        
        # 5-7: スペクトル特性（SVD）
        try:
            # 計算コスト削減のためランダムSVD
            if W.shape[0] > 1000 or W.shape[1] > 1000:
                # 部分行列でSVD
                sub_W = W[:min(1000, W.shape[0]), :min(1000, W.shape[1])]
                U, S, Vt = np.linalg.svd(sub_W, full_matrices=False)
            else:
                U, S, Vt = np.linalg.svd(W, full_matrices=False)
            
            features.append(S[0] if len(S) > 0 else 0.0)  # 最大特異値
            features.append(np.sum(S))  # 特異値の和
            features.append(S[0] / (S[-1] + 1e-10) if len(S) > 0 else 1.0)  # 条件数
            
        except Exception as e:
            # SVD失敗時はデフォルト値
            features.extend([1.0, 1.0, 1.0])
        
        # 8: スパース性
        sparsity = np.sum(np.abs(W) < 1e-6) / W.size
        features.append(sparsity)
        
        # 9: ランク比率
        try:
            rank = np.linalg.matrix_rank(W)
            rank_ratio = rank / min(W.shape)
            features.append(rank_ratio)
        except:
            features.append(0.5)
        
        # 10: Frobeniusノルム
        frobenius = np.linalg.norm(W, 'fro')
        features.append(frobenius)
        
        # 11: スペクトルエントロピー
        if len(S) > 0:
            # 特異値を正規化して確率分布とみなす
            p = S / np.sum(S)
            p = p[p > 1e-10]  # ゼロ除去
            entropy = -np.sum(p * np.log(p + 1e-10))
            features.append(entropy)
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def batch_profile(
        self,
        expert_list: List[Tuple[int, int]],
        save_path: Optional[str] = None
    ) -> Dict[Tuple[int, int], Dict[Domain, float]]:
        """
        複数expertを一括プロファイリング
        
        Args:
            expert_list: [(layer, expert_id), ...]
            save_path: 結果保存先（オプション）
        
        Returns:
            {(layer, expert_id): {Domain: score}}
        """
        results = {}
        
        for i, (layer, expert_id) in enumerate(expert_list, 1):
            if i % 100 == 0:
                print(f"[EXPERT_PROFILER] Profiling {i}/{len(expert_list)}")
            
            try:
                scores = self.profile_expert(layer, expert_id)
                results[(layer, expert_id)] = scores
            except Exception as e:
                print(f"[EXPERT_PROFILER] Error profiling L{layer}E{expert_id}: {e}")
                results[(layer, expert_id)] = {}
        
        if save_path:
            self.save_profiles(results, save_path)
        
        return results
    
    def save_profiles(
        self,
        profiles: Dict[Tuple[int, int], Dict[Domain, float]],
        filepath: str
    ):
        """
        プロファイル結果を保存
        """
        import json
        
        # JSONシリアライズ可能な形式に変換
        data = {}
        for (layer, expert_id), scores in profiles.items():
            key = f"L{layer}E{expert_id}"
            data[key] = {
                "layer": layer,
                "expert_id": expert_id,
                "domain_scores": {d.value: float(s) for d, s in scores.items()}
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[EXPERT_PROFILER] Saved {len(profiles)} profiles to {filepath}")
    
    def load_signatures(self, filepath: str):
        """
        事前計算済みシグネチャをロード
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.domain_signatures = {}
        for domain_str, features in data.items():
            try:
                domain = Domain[domain_str.upper()]
                self.domain_signatures[domain] = np.array(features)
            except KeyError:
                print(f"[WARNING] Unknown domain: {domain_str}")
        
        print(f"[EXPERT_PROFILER] Loaded {len(self.domain_signatures)} signatures")
    
    def get_top_experts_for_domain(
        self,
        domain: Domain,
        profiles: Dict[Tuple[int, int], Dict[Domain, float]],
        k: int = 10
    ) -> List[Tuple[int, int, float]]:
        """
        特定ドメインに最も強いexpertを取得
        
        Args:
            domain: 対象ドメイン
            profiles: プロファイル結果
            k: 返すexpert数
        
        Returns:
            [(layer, expert_id, score), ...] - スコア降順
        """
        scored_experts = []
        
        for (layer, expert_id), scores in profiles.items():
            score = scores.get(domain, 0.0)
            scored_experts.append((layer, expert_id, score))
        
        # スコア降順でソート
        scored_experts.sort(key=lambda x: -x[2])
        
        return scored_experts[:k]

    # ──────────────────────────────────────────────────────────────
    # Phase 6 Enhancement: Routing-Analysis-Based Profiling
    # ──────────────────────────────────────────────────────────────

    def calibrate_from_routing(
        self,
        model=None,
        tokenizer=None,
        stub: bool = False,
        cache_path: Optional[str] = None,
        domains: Optional[List[str]] = None,
        probes_per_domain: int = 15,
    ) -> None:
        """
        Calibrate domain signatures using ACTUAL routing analysis.

        This replaces the fake hardcoded signatures with real data derived
        from running probe queries through the model and observing which
        experts actually activate for each domain.

        Args:
            model: Loaded DeepSeek V3 model (None for stub mode)
            tokenizer: Model tokenizer
            stub: If True, use synthetic routing data
            cache_path: Path to save/load routing results
            domains: Domains to analyze (None = all)
            probes_per_domain: Number of probe queries per domain
        """
        RouterAnalyzer = _get_router_analyzer()

        analyzer = RouterAnalyzer(
            model=model,
            tokenizer=tokenizer,
            stub=stub,
            cache_path=cache_path,
        )

        print("[EXPERT_PROFILER] Running routing-based calibration...")
        self._routing_result = analyzer.analyze_all_domains(
            domains=domains,
            probes_per_domain=probes_per_domain,
        )

        # Build expert → {Domain: score} mapping from routing
        self._routing_profiles: Dict[Tuple[int, int], Dict[Domain, float]] = {}

        for routing_domain, expert_profiles in self._routing_result.domain_experts.items():
            ir_domains = ROUTING_DOMAIN_MAP.get(routing_domain, [])

            for profile in expert_profiles:
                key = (profile.layer, profile.expert_id)
                if key not in self._routing_profiles:
                    self._routing_profiles[key] = {}

                for ir_domain in ir_domains:
                    existing = self._routing_profiles[key].get(ir_domain, 0.0)
                    score = profile.activation_frequency * profile.mean_activation_score
                    self._routing_profiles[key][ir_domain] = max(existing, score)

        print(f"[EXPERT_PROFILER] Calibration complete: "
              f"{len(self._routing_profiles)} expert profiles built")

    def profile_expert_routing(
        self,
        layer: int,
        expert_id: int,
    ) -> Dict[Domain, float]:
        """
        Get domain scores for an expert using ROUTING analysis (not weight stats).

        Requires calibrate_from_routing() to have been called first.

        Returns:
            {Domain: score} where score is based on actual activation frequency
        """
        if not hasattr(self, "_routing_profiles"):
            raise RuntimeError(
                "Routing profiles not available. "
                "Call calibrate_from_routing() first."
            )

        return self._routing_profiles.get((layer, expert_id), {})

    def get_top_routing_experts(
        self,
        domain: Domain,
        k: int = 20,
    ) -> List[Tuple[int, int, float]]:
        """
        Get top-K experts for a domain using routing analysis data.

        Returns:
            [(layer, expert_id, score), ...] sorted by score descending
        """
        if not hasattr(self, "_routing_profiles"):
            raise RuntimeError("Call calibrate_from_routing() first.")

        scored = []
        for (layer, expert_id), scores in self._routing_profiles.items():
            score = scores.get(domain, 0.0)
            if score > 0.0:
                scored.append((layer, expert_id, score))

        scored.sort(key=lambda x: -x[2])
        return scored[:k]

    def save_routing_profiles(self, filepath: str) -> None:
        """Save routing-based profiles to JSON."""
        if not hasattr(self, "_routing_profiles"):
            raise RuntimeError("Call calibrate_from_routing() first.")

        data = {}
        for (layer, expert_id), scores in self._routing_profiles.items():
            key = f"L{layer}E{expert_id}"
            data[key] = {
                "layer": layer,
                "expert_id": expert_id,
                "routing_scores": {
                    d.value: float(s) for d, s in scores.items()
                }
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[EXPERT_PROFILER] Saved {len(data)} routing profiles to {filepath}")

    def load_routing_profiles(self, filepath: str) -> None:
        """Load routing-based profiles from JSON."""
        with open(filepath) as f:
            data = json.load(f)

        self._routing_profiles = {}
        for key, entry in data.items():
            layer = entry["layer"]
            expert_id = entry["expert_id"]
            scores = {}
            for domain_str, score in entry.get("routing_scores", {}).items():
                try:
                    scores[Domain(domain_str)] = float(score)
                except (ValueError, KeyError):
                    pass
            self._routing_profiles[(layer, expert_id)] = scores

        print(f"[EXPERT_PROFILER] Loaded {len(self._routing_profiles)} routing profiles")
