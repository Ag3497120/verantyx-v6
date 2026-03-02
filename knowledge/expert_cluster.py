"""
expert_cluster.py — Expert クラスタリング + Cross Engine ルーティング強化

Step 1: 15104 Expert の SVD 方向ベクトルを k-means クラスタリング
Step 2: 問題テキスト → Expert 活性化 → クラスタ分布を返す
Step 3: クラスタ → ソルバー統計マップ（build_solver_map で構築）

使い方:
  from knowledge.expert_cluster import ExpertClusterRouter
  router = ExpertClusterRouter()
  cluster_dist = router.get_cluster_distribution("What color fills the grid?")
  recommendations = router.recommend_solvers("What color fills the grid?")
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DB_DIR = Path("/Users/motonishikoudai/avh_math/avh_math/db/moe_sparse_cross_600b_real")
CONCEPT_DIRS_PATH = DB_DIR / "concept_dirs.npy"
EMBED_TOKENS_PATH = DB_DIR / "embed_tokens.npy"
TOKENIZER_PATH = DB_DIR / "tokenizer.json"

CACHE_DIR = Path(__file__).parent / "expert_cluster_cache"
CLUSTER_LABELS_PATH = CACHE_DIR / "cluster_labels.npy"
CLUSTER_CENTERS_PATH = CACHE_DIR / "cluster_centers.npy"
SOLVER_MAP_PATH = CACHE_DIR / "solver_map.json"

N_CLUSTERS = 64


def _load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER_PATH))


def _embed_text(text: str, embed_tokens: np.ndarray, tokenizer) -> np.ndarray:
    enc = tokenizer.encode(text)
    ids = enc.ids[:512]
    if not ids:
        return np.zeros(embed_tokens.shape[1], dtype=np.float32)
    vecs = embed_tokens[ids]
    mean_vec = vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec


class ExpertClusterRouter:
    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self._concept_dirs = None
        self._embed_tokens = None
        self._tokenizer = None
        self._cluster_labels = None
        self._cluster_centers = None
        self._solver_map = None

    def _ensure_loaded(self):
        if self._concept_dirs is None:
            self._concept_dirs = np.load(str(CONCEPT_DIRS_PATH))
            self._embed_tokens = np.load(str(EMBED_TOKENS_PATH), mmap_mode='r')
            self._tokenizer = _load_tokenizer()

    def _ensure_clusters(self):
        if self._cluster_labels is not None:
            return
        if CLUSTER_LABELS_PATH.exists() and CLUSTER_CENTERS_PATH.exists():
            self._cluster_labels = np.load(str(CLUSTER_LABELS_PATH))
            self._cluster_centers = np.load(str(CLUSTER_CENTERS_PATH))
            return
        print(f"[ExpertCluster] Computing k-means with {self.n_clusters} clusters...")
        self._ensure_loaded()
        self._run_clustering()

    def _run_clustering(self):
        from sklearn.cluster import MiniBatchKMeans
        X = self._concept_dirs[:, 0, :].copy()
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X /= norms

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, batch_size=1024,
            max_iter=300, random_state=42, n_init=3,
        )
        self._cluster_labels = kmeans.fit_predict(X)
        self._cluster_centers = kmeans.cluster_centers_

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(CLUSTER_LABELS_PATH), self._cluster_labels)
        np.save(str(CLUSTER_CENTERS_PATH), self._cluster_centers)

        unique, counts = np.unique(self._cluster_labels, return_counts=True)
        print(f"[ExpertCluster] Saved {self.n_clusters} clusters. "
              f"Sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    def get_expert_activations(self, text: str) -> np.ndarray:
        """テキスト → Expert 活性化スコア (15104,)"""
        self._ensure_loaded()
        query_vec = _embed_text(text, self._embed_tokens, self._tokenizer)
        cd_flat = self._concept_dirs[:, 0, :]
        return cd_flat @ query_vec

    def get_cluster_distribution(self, text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """テキスト → 上位クラスタ活性化 [(cluster_id, mean_activation)]"""
        self._ensure_clusters()
        activations = self.get_expert_activations(text)
        cluster_scores = {}
        for cid in range(self.n_clusters):
            mask = self._cluster_labels == cid
            if mask.any():
                cluster_scores[cid] = float(activations[mask].mean())
        return sorted(cluster_scores.items(), key=lambda x: -x[1])[:top_k]

    def get_layer_cluster_distribution(self, text: str, top_k: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """階層別クラスタ分布 (shallow: layer 3-20, deep: layer 40-61)"""
        self._ensure_clusters()
        activations = self.get_expert_activations(text)
        result = {}
        for label, layer_range in [("shallow", (3, 21)), ("deep", (40, 62)), ("all", (3, 62))]:
            start_idx = (layer_range[0] - 3) * 256
            end_idx = min((layer_range[1] - 3) * 256, 15104)
            mask_range = np.zeros(15104, dtype=bool)
            mask_range[start_idx:end_idx] = True
            cluster_scores = {}
            for cid in range(self.n_clusters):
                mask = (self._cluster_labels == cid) & mask_range
                if mask.any():
                    cluster_scores[cid] = float(activations[mask].mean())
            result[label] = sorted(cluster_scores.items(), key=lambda x: -x[1])[:top_k]
        return result

    def load_solver_map(self) -> Dict:
        if self._solver_map is not None:
            return self._solver_map
        if SOLVER_MAP_PATH.exists():
            with open(SOLVER_MAP_PATH) as f:
                self._solver_map = json.load(f)
            return self._solver_map
        return {}

    def recommend_solvers(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """テキスト → 推薦ソルバー [(solver_name, score)]"""
        solver_map = self.load_solver_map()
        if not solver_map:
            return []
        cluster_dist = self.get_cluster_distribution(text, top_k=20)
        solver_scores: Dict[str, float] = {}
        for cid, activation in cluster_dist:
            cid_str = str(cid)
            if cid_str in solver_map:
                for solver_name, solver_stats in solver_map[cid_str].items():
                    success_rate = solver_stats.get("success_rate", 0)
                    count = solver_stats.get("count", 0)
                    count_weight = min(count / 10.0, 1.0)
                    score = activation * success_rate * count_weight
                    solver_scores[solver_name] = solver_scores.get(solver_name, 0) + score
        return sorted(solver_scores.items(), key=lambda x: -x[1])[:top_k]

    def describe_cluster(self, cluster_id: int, top_n: int = 20) -> Dict:
        """クラスタの特徴（上位トークン + layer分布）"""
        self._ensure_clusters()
        self._ensure_loaded()
        center = self._cluster_centers[cluster_id]
        center_norm = center / (np.linalg.norm(center) + 1e-8)

        chunk_size = 10000
        n_tokens = self._embed_tokens.shape[0]
        top_sims = []
        for start in range(0, n_tokens, chunk_size):
            end = min(start + chunk_size, n_tokens)
            chunk = np.array(self._embed_tokens[start:end])
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1
            chunk_normed = chunk / norms
            sims = chunk_normed @ center_norm
            for i, sim in enumerate(sims):
                top_sims.append((start + i, float(sim)))
        top_sims.sort(key=lambda x: -x[1])
        top_sims = top_sims[:top_n]

        vocab = self._tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        tokens = [(id_to_token.get(tid, f"[{tid}]"), sim) for tid, sim in top_sims]

        mask = self._cluster_labels == cluster_id
        expert_indices = np.where(mask)[0]
        layers = [(idx // 256) + 3 for idx in expert_indices]
        layer_counts = {}
        for l in layers:
            layer_counts[l] = layer_counts.get(l, 0) + 1

        return {
            "cluster_id": cluster_id,
            "n_experts": int(mask.sum()),
            "top_tokens": tokens,
            "layer_distribution": dict(sorted(layer_counts.items())),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--describe", type=int, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--describe-all", action="store_true")
    args = parser.parse_args()

    router = ExpertClusterRouter()

    if args.build:
        router._ensure_loaded()
        router._run_clustering()
        print("Done.")

    if args.describe is not None:
        info = router.describe_cluster(args.describe)
        print(f"\n=== Cluster {info['cluster_id']} ({info['n_experts']} experts) ===")
        print("Top tokens:")
        for tok, sim in info['top_tokens']:
            print(f"  {sim:.4f}  {tok}")
        print(f"Layer distribution: {info['layer_distribution']}")

    if args.describe_all:
        router._ensure_clusters()
        for cid in range(router.n_clusters):
            info = router.describe_cluster(cid, top_n=5)
            tokens_str = ", ".join(f"{tok}({sim:.3f})" for tok, sim in info['top_tokens'][:5])
            print(f"C{cid:02d} [{info['n_experts']:3d}] {tokens_str}")

    if args.query:
        print(f"\nQuery: {args.query}")
        dist = router.get_cluster_distribution(args.query)
        print("Cluster distribution:")
        for cid, score in dist:
            print(f"  Cluster {cid:2d}: {score:.4f}")
        layer_dist = router.get_layer_cluster_distribution(args.query)
        for label in ["shallow", "deep"]:
            print(f"\n{label.upper()} layers:")
            for cid, score in layer_dist[label][:5]:
                print(f"  Cluster {cid:2d}: {score:.4f}")


if __name__ == "__main__":
    main()
