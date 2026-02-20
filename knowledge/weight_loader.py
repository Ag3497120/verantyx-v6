"""
DeepSeek V3.2 Weight File Loader

ローカルのモデルファイル（safetensors）から重みを効率的にロード
"""
import os
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path


class DeepSeekWeightLoader:
    """
    DeepSeek V3.2のモデルファイルをロード
    
    対応形式:
    - safetensors（推奨）
    - PyTorch .bin（将来対応）
    
    DeepSeek V3構造:
    - 総パラメータ: ~671B
    - MoE: 256 experts × 61 layers
    - Hidden dim: 7168
    - FFN dim: 18432（per expert）
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: モデルディレクトリのパス
                例: "/path/to/DeepSeek-V3-Base"
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.config = self._load_config()
        self.num_layers = self.config.get("num_hidden_layers", 61)
        self.num_experts = self.config.get("n_routed_experts", 256)
        self.hidden_size = self.config.get("hidden_size", 7168)
        
        # キャッシュ（メモリ効率化）
        self.weight_cache = {}
        self.cache_limit = 10  # 最大10個のexpertをキャッシュ
        
        print(f"[WEIGHT_LOADER] Initialized: {self.num_layers} layers, {self.num_experts} experts")
    
    def _load_config(self) -> Dict:
        """
        config.jsonをロード
        """
        config_path = self.model_path / "config.json"
        
        if not config_path.exists():
            print(f"[WARNING] config.json not found, using defaults")
            return {
                "num_hidden_layers": 61,
                "n_routed_experts": 256,
                "hidden_size": 7168
            }
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_expert_weights(
        self,
        layer: int,
        expert_id: int,
        component: str = "gate_proj"
    ) -> np.ndarray:
        """
        特定expertの重み行列をロード
        
        Args:
            layer: レイヤー番号 (0-60)
            expert_id: Expert ID (0-255)
            component: 重みコンポーネント
                - "gate_proj": FFNのgate projection
                - "up_proj": FFNのup projection
                - "down_proj": FFNのdown projection
        
        Returns:
            重み行列 (numpy array)
        """
        cache_key = (layer, expert_id, component)
        
        # キャッシュチェック
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]
        
        # 重みキー（DeepSeek V3の命名規則）
        weight_key = f"model.layers.{layer}.mlp.experts.{expert_id}.{component}.weight"
        
        try:
            # safetensorsからロード
            weight = self._load_from_safetensors(weight_key)
            
            # キャッシュ管理
            if len(self.weight_cache) >= self.cache_limit:
                # LRU: 最古のエントリを削除
                oldest_key = next(iter(self.weight_cache))
                del self.weight_cache[oldest_key]
            
            self.weight_cache[cache_key] = weight
            
            return weight
            
        except Exception as e:
            print(f"[WEIGHT_LOADER] Error loading {weight_key}: {e}")
            # フォールバック: ゼロ行列
            return np.zeros((self.hidden_size, self.hidden_size))
    
    def _load_from_safetensors(self, weight_key: str) -> np.ndarray:
        """
        safetensorsから特定の重みをロード
        
        Args:
            weight_key: 重みのキー名
        
        Returns:
            numpy array
        """
        try:
            # safetensorsライブラリを使用（遅延インポート）
            from safetensors import safe_open
            
            # safetensorsファイルを探索
            safetensors_files = list(self.model_path.glob("*.safetensors"))
            
            if not safetensors_files:
                raise FileNotFoundError("No safetensors files found")
            
            # 各ファイルを試行（sharded model対応）
            for st_file in safetensors_files:
                try:
                    with safe_open(st_file, framework="numpy") as f:
                        # キーが存在するか確認
                        keys = f.keys()
                        if weight_key in keys:
                            tensor = f.get_tensor(weight_key)
                            return tensor
                except Exception as e:
                    # このファイルにはない、次へ
                    continue
            
            raise KeyError(f"Weight key not found: {weight_key}")
            
        except ImportError:
            print("[ERROR] safetensors library not installed. Install: pip install safetensors")
            raise
    
    def list_experts(self) -> List[Tuple[int, int]]:
        """
        全expertのリストを取得
        
        Returns:
            [(layer, expert_id), ...]
        """
        experts = []
        for layer in range(self.num_layers):
            for expert_id in range(self.num_experts):
                experts.append((layer, expert_id))
        return experts
    
    def get_expert_metadata(self, layer: int, expert_id: int) -> Dict:
        """
        Expertのメタデータを取得
        
        Args:
            layer: レイヤー番号
            expert_id: Expert ID
        
        Returns:
            メタデータ辞書
        """
        return {
            "layer": layer,
            "expert_id": expert_id,
            "total_experts_in_layer": self.num_experts,
            "hidden_size": self.hidden_size,
            "identifier": f"L{layer}E{expert_id}"
        }
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """
        メモリ使用量を推定
        
        Returns:
            {"per_expert_mb": float, "all_experts_gb": float}
        """
        # 1 expert = hidden_size × ffn_dim × 3 components × 4 bytes (float32)
        # 例: 7168 × 18432 × 3 × 4 = ~1.6GB per expert
        
        per_expert_bytes = self.hidden_size * 18432 * 3 * 4
        per_expert_mb = per_expert_bytes / (1024 ** 2)
        
        total_experts = self.num_layers * self.num_experts
        all_experts_gb = (per_expert_bytes * total_experts) / (1024 ** 3)
        
        return {
            "per_expert_mb": per_expert_mb,
            "all_experts_gb": all_experts_gb,
            "total_experts": total_experts
        }


class WeightLoaderStub(DeepSeekWeightLoader):
    """
    スタブ版WeightLoader
    
    実際のモデルファイルがない場合のテスト用
    """
    
    def __init__(self):
        """スタブ初期化（model_pathなし）"""
        self.model_path = Path("/stub/path")
        self.config = {
            "num_hidden_layers": 61,
            "n_routed_experts": 256,
            "hidden_size": 7168
        }
        self.num_layers = 61
        self.num_experts = 256
        self.hidden_size = 7168
        self.weight_cache = {}
        self.cache_limit = 10
        
        print("[WEIGHT_LOADER] Stub mode initialized")
    
    def load_expert_weights(
        self,
        layer: int,
        expert_id: int,
        component: str = "gate_proj"
    ) -> np.ndarray:
        """
        ランダムな重み行列を返す（スタブ）
        """
        # 再現性のためシード設定
        np.random.seed(layer * 1000 + expert_id)
        
        # ランダム行列（正規分布）
        weight = np.random.randn(self.hidden_size, 18432) * 0.02
        
        return weight
