"""
expert_loader.py
================
DeepSeek V3-0324 Q8_0 GGUFからExpert重みを読む

設計:
  1. Multi-shardインデックス → tensor_name → (file_path, offset, shape, qtype)
  2. Router重みロード   : blk.L.ffn_gate_inp.weight [7168, 256]  (小さい)
  3. 共有Expert重みロード: blk.L.ffn_{gate,up,down}_shexp.weight (常時発火)
  4. SVD地図 (concept_dirs) との接続
  5. Q8_0 高速デコード  : ループなし vectorized numpy

役割分担:
  concept_dirs.npy  → 案内人 (どのExpertがどの知識を持つか)
  Router重み        → 現場の案内 (このクエリはどのExpertへ)
  共有Expert重み    → 常時変換 (全クエリに適用)
  Packed Expert重み → 深い知識 (expensive, 必要時のみ)

テンソル命名規則 (V3-0324 GGUF):
  dense layers (blk.0-2):
    blk.L.ffn_gate.weight   [7168, 18432]
    blk.L.ffn_up.weight     [7168, 18432]
    blk.L.ffn_down.weight   [18432, 7168]
  MoE layers (blk.3-61):
    blk.L.ffn_gate_inp.weight    [7168, 256]          router
    blk.L.ffn_gate_shexp.weight  [7168, 2048]         shared expert
    blk.L.ffn_up_shexp.weight    [7168, 2048]         shared expert
    blk.L.ffn_down_shexp.weight  [2048, 7168]         shared expert
    blk.L.ffn_gate_exps.weight   [7168, 2048, 256]    256 experts packed
    blk.L.ffn_up_exps.weight     [7168, 2048, 256]    256 experts packed
    blk.L.ffn_down_exps.weight   [2048, 7168, 256]    256 experts packed

Expert flat_id → (layer, expert_id):
  flat_id = (layer - 3) * 256 + expert_id  (layer 3..61 = 59 MoE layers)
  → 59 × 256 = 15104 experts = concept_dirs.shape[0] ✓
"""

from __future__ import annotations

import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================
# Q8_0 vectorized dequantize
# ============================

def dequantize_q8_0_fast(raw: bytes, shape: List[int]) -> np.ndarray:
    """
    Q8_0 デコード (numpy vectorized, ループなし)

    Q8_0 ブロック構造:
      [float16 scale (2 bytes)] + [32 × int8 values (32 bytes)] = 34 bytes/block

    Args:
        raw   : 生バイト列
        shape : テンソル形状 [dim0, dim1, ...]
    Returns:
        float32 numpy array, shape通りに reshape済み
    """
    n_elements = 1
    for d in shape:
        n_elements *= d

    n_blocks = (n_elements + 31) // 32
    expected = n_blocks * 34
    buf = np.frombuffer(raw[:expected], dtype=np.uint8).copy()

    # 各ブロックのオフセット (0, 34, 68, ...)
    block_starts = np.arange(n_blocks, dtype=np.int64) * 34  # (n_blocks,)

    # スケール: bytes 0-1 (IEEE 754 float16, little-endian)
    # low_byte | (high_byte << 8) → uint16 → view as float16
    low  = buf[block_starts].astype(np.uint16)
    high = buf[block_starts + 1].astype(np.uint16)
    scale_u16 = np.empty(n_blocks, dtype=np.uint16)
    scale_u16[:] = low | (high << 8)
    scales = scale_u16.view(np.float16).astype(np.float32)  # (n_blocks,)

    # int8値: bytes 2-33 (signed)
    val_indices = block_starts[:, None] + np.arange(2, 34, dtype=np.int64)[None, :]  # (n_blocks, 32)
    quant_vals = buf[val_indices].view(np.int8).astype(np.float32)  # (n_blocks, 32)

    # NaN/Inf スケールを0に
    scales = np.nan_to_num(scales, nan=0.0, posinf=0.0, neginf=0.0)

    # dequantize: scale × int8
    result = (scales[:, None] * quant_vals).reshape(-1)  # (n_blocks*32,)

    return result[:n_elements].reshape(shape).astype(np.float32)


# ============================
# Built-in GGUF scanner (fallback)
# ============================

def _get_gguf_data_section_start(path: Path) -> int:
    """
    GGUFファイルのデータセクション開始絶対オフセットを返す
    data_offset (ファイル内フィールド) はここからの相対値
    """
    import struct as _s

    def _ru32(f): return _s.unpack("<I", f.read(4))[0]
    def _ru64(f): return _s.unpack("<Q", f.read(8))[0]
    def _rstr(f):
        n = _ru64(f)
        return f.read(n).decode("utf-8", errors="replace")

    ELEM_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    ALIGNMENT = 32

    with open(path, "rb") as f:
        f.read(4)       # magic
        _ru32(f)        # version
        n_tensors = _ru64(f)
        n_kv = _ru64(f)

        for _ in range(n_kv):
            _rstr(f)
            vtype = _ru32(f)
            if vtype == 8:
                _rstr(f)
            elif vtype == 9:
                atype = _ru32(f)
                alen = _ru64(f)
                for _ in range(alen):
                    if atype == 8: _rstr(f)
                    elif atype in ELEM_SIZES: f.read(ELEM_SIZES[atype])
            elif vtype in ELEM_SIZES:
                f.read(ELEM_SIZES[vtype])

        for _ in range(n_tensors):
            _rstr(f)            # name
            n_dims = _ru32(f)
            for _ in range(n_dims): _ru64(f)  # shape
            _ru32(f)            # qtype
            _ru64(f)            # data_offset (skip)

        pos = f.tell()

    return ((pos + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def _scan_gguf_builtin(path: Path):
    """
    engine_coreなしで動くミニGGUFスキャナ
    data_offset を絶対ファイルオフセットに変換して返す
    (GGUF仕様: data_offset はデータセクション先頭からの相対値)
    """
    import struct as _s

    def _ru32(f): return _s.unpack("<I", f.read(4))[0]
    def _ru64(f): return _s.unpack("<Q", f.read(8))[0]
    def _rstr(f):
        n = _ru64(f)
        return f.read(n).decode("utf-8", errors="replace")

    ELEM_SIZES = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
    ALIGNMENT = 32  # GGUF_DEFAULT_ALIGNMENT

    out = []
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        _ru32(f)  # version
        n_tensors = _ru64(f)
        n_kv = _ru64(f)

        for _ in range(n_kv):
            _rstr(f)
            vtype = _ru32(f)
            if vtype == 8:
                _rstr(f)
            elif vtype == 9:
                atype = _ru32(f)
                alen = _ru64(f)
                for _ in range(alen):
                    if atype == 8:
                        _rstr(f)
                    elif atype in ELEM_SIZES:
                        f.read(ELEM_SIZES[atype])
            elif vtype in ELEM_SIZES:
                f.read(ELEM_SIZES[vtype])

        # テンソルメタデータ読み込み (data_offsetはまだ相対値)
        raw_offsets = []
        for _ in range(n_tensors):
            name = _rstr(f)
            n_dims = _ru32(f)
            shape = [_ru64(f) for _ in range(n_dims)]
            qtype = _ru32(f)
            rel_offset = _ru64(f)
            raw_offsets.append({"name": name, "shape": shape, "qtype": qtype,
                                 "rel_offset": rel_offset})

        # データセクション開始位置 (アライメント後の絶対オフセット)
        pos = f.tell()
        data_section_start = ((pos + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT

    # 絶対オフセットに変換
    for r in raw_offsets:
        r["data_offset"] = data_section_start + r["rel_offset"]
    out = raw_offsets

    # data_bytes をオフセット差分から計算
    out_s = sorted(out, key=lambda r: r["data_offset"])
    for i, r in enumerate(out_s):
        nxt = out_s[i+1]["data_offset"] if i+1 < len(out_s) else file_size
        r["data_bytes"] = max(0, nxt - r["data_offset"])

    return out_s


# ============================
# Multi-shard Tensor Index
# ============================

class TensorIndex:
    """
    全シャードのテンソルインデックス

    Usage:
        idx = TensorIndex.build(shard_dir)   # 初回: スキャンして構築
        idx = TensorIndex.load(index_path)   # 2回目以降: キャッシュ読込
    """

    def __init__(self, records: List[Dict]):
        # name → {file, offset, shape, qtype, data_bytes}
        self._map: Dict[str, Dict] = {r["name"]: r for r in records}

    def __contains__(self, name: str) -> bool:
        return name in self._map

    def get(self, name: str) -> Optional[Dict]:
        return self._map.get(name)

    def names_matching(self, pattern: str) -> List[str]:
        return [n for n in self._map if pattern in n]

    @classmethod
    def build(cls, shard_dir: Path, save_path: Optional[Path] = None) -> "TensorIndex":
        """
        GGUFシャードディレクトリを全スキャンしてインデックス構築
        shard 1のみでも動作 (他シャードは後でupdateできる)
        """
        import sys
        avh_root = str(Path.home() / "avh_math")
        if avh_root not in sys.path:
            sys.path.insert(0, avh_root)

        try:
            from engine_core.weight_mining.gguf_scan import scan_gguf
        except ImportError:
            # フォールバック: 内蔵スキャナ
            scan_gguf = _scan_gguf_builtin

        records = []
        files = sorted(p for p in shard_dir.iterdir() if p.suffix == ".gguf")
        print(f"[TensorIndex] Scanning {len(files)} GGUF shards...")

        for shard_file in files:
            print(f"  Scanning {shard_file.name}...", end=" ", flush=True)
            try:
                # データセクション開始位置を計算 (絶対オフセット変換に必要)
                data_start = _get_gguf_data_section_start(shard_file)
                shards = scan_gguf(shard_file)
                for r in shards:
                    # engine_coreのscan_ggufはrelative offsetを返すため絶対値に変換
                    abs_offset = data_start + r["data_offset"]
                    records.append({
                        "name": r["name"],
                        "file": str(shard_file),
                        "offset": abs_offset,
                        "shape": r["shape"],
                        "qtype": r["qtype"],
                        "data_bytes": r.get("data_bytes", 0),
                    })
                print(f"{len(shards)} tensors")
            except Exception as e:
                print(f"ERROR: {e}")

        idx = cls(records)
        print(f"[TensorIndex] Total: {len(idx._map)} tensors")

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            print(f"[TensorIndex] Saved → {save_path}")

        return idx

    @classmethod
    def load(cls, index_path: Path) -> "TensorIndex":
        records = []
        with index_path.open() as f:
            for line in f:
                records.append(json.loads(line))
        print(f"[TensorIndex] Loaded {len(records)} tensors from {index_path}")
        return cls(records)


def _load_tensor_raw(info: Dict) -> Optional[np.ndarray]:
    """
    インデックスエントリからテンソルを読み込んでデコード
    現在はQ8_0 (qtype=8) と F32 (qtype=0) に対応
    """
    file_path = info["file"]
    offset = info["offset"]
    data_bytes = info["data_bytes"]
    shape = info["shape"]
    qtype = info["qtype"]

    with open(file_path, "rb") as f:
        f.seek(offset)
        raw = f.read(data_bytes)

    if qtype == 0:  # F32
        n = 1
        for d in shape:
            n *= d
        return np.frombuffer(raw[:n * 4], dtype=np.float32).reshape(shape).copy()

    elif qtype == 1:  # F16
        n = 1
        for d in shape:
            n *= d
        return np.frombuffer(raw[:n * 2], dtype=np.float16).astype(np.float32).reshape(shape)

    elif qtype == 8:  # Q8_0
        return dequantize_q8_0_fast(raw, shape)

    else:
        print(f"[WARN] Unsupported qtype={qtype}, returning zeros for shape {shape}")
        n = 1
        for d in shape:
            n *= d
        return np.zeros(shape, dtype=np.float32)


# ============================
# ExpertLoader
# ============================

class ExpertLoader:
    """
    SVD地図 (concept_dirs) + GGUF重みを統合するローダー

    主なAPI:
        loader.query_to_experts(query_vec, top_k=5)
            → [(layer, expert_id, score), ...]

        loader.shared_expert_transform(query_vec, layer)
            → transformed_vec (7168,) [常時発火expert通過後]

        loader.router_scores(query_vec, layer)
            → expert_scores (256,) [このlayerでのrouter出力]

    Expert flat_id の定義:
        flat_id = (layer - FIRST_MOE_LAYER) * N_EXPERTS_PER_LAYER + expert_id
        FIRST_MOE_LAYER = 3, N_EXPERTS_PER_LAYER = 256
    """

    FIRST_MOE_LAYER = 3
    N_EXPERTS_PER_LAYER = 256
    HIDDEN_DIM = 7168
    FFN_INTERMEDIATE = 2048  # MoE expert intermediate dim
    FFN_DENSE_INTERMEDIATE = 18432  # dense layer intermediate dim

    def __init__(
        self,
        tensor_index: TensorIndex,
        concept_dirs: np.ndarray,         # (15104, 4, 7168)
        embed_tokens: Optional[np.ndarray] = None,  # (129280, 7168) mmap OK
    ):
        self.idx = tensor_index
        self.concept_dirs = concept_dirs   # (n_experts, 4, hidden)
        self.embed_tokens = embed_tokens

        # ルーター重みのキャッシュ: layer → np.ndarray [7168, 256]
        self._router_cache: Dict[int, np.ndarray] = {}
        # 共有Expertキャッシュ: layer → {gate, up, down}
        self._shexp_cache: Dict[int, Dict[str, np.ndarray]] = {}

    # -----------------------
    # SVD地図を使ったExpert選択
    # -----------------------

    def query_to_experts(
        self,
        query_vec: np.ndarray,  # (7168,)
        top_k: int = 5,
        use_router: bool = False,
    ) -> List[Tuple[int, int, float]]:
        """
        クエリに最も関連するExpertを返す

        Args:
            query_vec  : L2正規化済み 7168次元クエリベクトル
            top_k      : 返すExpert数
            use_router : True→ルーター重みで補正 (layerがロード済みの場合)

        Returns:
            [(layer, expert_id, score), ...] スコア降順
        """
        # concept_dirs (15104, 4, 7168) × query_vec (7168,) → (15104, 4)
        dots = self.concept_dirs @ query_vec          # (15104, 4)
        scores = dots.max(axis=1)                     # (15104,)

        top_flat_ids = np.argsort(scores)[-top_k:][::-1]
        result = []
        for fid in top_flat_ids:
            layer = int(fid // self.N_EXPERTS_PER_LAYER) + self.FIRST_MOE_LAYER
            expert_id = int(fid % self.N_EXPERTS_PER_LAYER)
            result.append((layer, expert_id, float(scores[fid])))

        return result

    # -----------------------
    # Router重みのロード
    # -----------------------

    def load_router(self, layer: int) -> Optional[np.ndarray]:
        """
        blk.L.ffn_gate_inp.weight [7168, 256] をロード
        Router: hidden_vec → expert_scores (256個)
        """
        if layer in self._router_cache:
            return self._router_cache[layer]

        name = f"blk.{layer}.ffn_gate_inp.weight"
        info = self.idx.get(name)
        if info is None:
            return None

        w = _load_tensor_raw(info)
        if w is not None:
            self._router_cache[layer] = w
        return w

    def router_scores(self, query_vec: np.ndarray, layer: int) -> Optional[np.ndarray]:
        """
        Router出力: query_vec (7168,) → expert_scores (256,)
        実際のDeepSeekルーティングを模倣 (発火させずに)
        """
        w = self.load_router(layer)
        if w is None:
            return None
        # w: [7168, 256] → scores = w.T @ query_vec = (256,)
        return (w.T @ query_vec).astype(np.float32)

    # -----------------------
    # 共有Expert (常時発火)
    # -----------------------

    def load_shared_expert(self, layer: int) -> Optional[Dict[str, np.ndarray]]:
        """
        共有Expert重みをロード
        gate: [7168, 2048], up: [7168, 2048], down: [2048, 7168]
        """
        if layer in self._shexp_cache:
            return self._shexp_cache[layer]

        names = {
            "gate": f"blk.{layer}.ffn_gate_shexp.weight",
            "up":   f"blk.{layer}.ffn_up_shexp.weight",
            "down": f"blk.{layer}.ffn_down_shexp.weight",
        }
        weights = {}
        for key, name in names.items():
            info = self.idx.get(name)
            if info is None:
                return None
            w = _load_tensor_raw(info)
            if w is None:
                return None
            weights[key] = w

        self._shexp_cache[layer] = weights
        return weights

    def shared_expert_transform(self, query_vec: np.ndarray, layer: int) -> Optional[np.ndarray]:
        """
        共有Expert変換 (SwiGLU):
            gate_out = W_gate.T @ x      # (2048,)
            up_out   = W_up.T @ x        # (2048,)
            h        = silu(gate_out) * up_out
            out      = W_down.T @ h      # (7168,)

        これは「推論」ではなく「重みを変換関数として適用」
        クエリがこの共有知識空間でどこに写像されるかを表す

        Args:
            query_vec: L2正規化済み (7168,)
            layer    : MoEレイヤー番号 (3以上)

        Returns:
            変換後ベクトル (7168,)、失敗時None
        """
        w = self.load_shared_expert(layer)
        if w is None:
            return None

        # W_gate: [7168, 2048] → W_gate.T: [2048, 7168]
        gate_out = w["gate"].T @ query_vec      # (2048,)
        up_out   = w["up"].T @ query_vec        # (2048,)

        # SiLU activation
        gate_act = gate_out * (1.0 / (1.0 + np.exp(-gate_out)))

        h = gate_act * up_out                   # (2048,)
        out = w["down"].T @ h                   # (7168,)

        # L2正規化
        norm = np.linalg.norm(out)
        if norm > 1e-8:
            out = out / norm

        return out.astype(np.float32)

    # -----------------------
    # MCQ選択肢ランキング
    # -----------------------

    def rank_choices_by_transform(
        self,
        stem_vec: np.ndarray,    # (7168,) L2正規化済み
        choice_vecs: List[np.ndarray],  # [(7168,), ...] 各選択肢
        layers: Optional[List[int]] = None,
    ) -> List[float]:
        """
        共有Expert変換後のベクトルで選択肢をランキング

        アイデア:
          stem_vec → 共有Expert → transformed_stem
          各choice_vec との cosine similarity → 最も高いものが答え

        Args:
            stem_vec    : 問題文のembedding
            choice_vecs : 各選択肢のembedding
            layers      : 使用するlayer番号 (Noneなら利用可能な全層)

        Returns:
            各選択肢のスコアリスト (高いほど有力)
        """
        if layers is None:
            layers = [l for l in range(self.FIRST_MOE_LAYER, 62)
                      if f"blk.{l}.ffn_gate_shexp.weight" in self.idx]

        if not layers:
            # フォールバック: 変換なし、直接cosine
            return [float(stem_vec @ cv) for cv in choice_vecs]

        # 複数層の変換を平均
        transformed_vecs = []
        for layer in layers[:8]:  # 最大8層
            t = self.shared_expert_transform(stem_vec, layer)
            if t is not None:
                transformed_vecs.append(t)

        if not transformed_vecs:
            return [float(stem_vec @ cv) for cv in choice_vecs]

        # 平均変換ベクトル
        mean_t = np.mean(transformed_vecs, axis=0)
        norm = np.linalg.norm(mean_t)
        if norm > 1e-8:
            mean_t /= norm

        # 各選択肢とのcosine similarity
        scores = [float(mean_t @ cv) for cv in choice_vecs]
        return scores

    # -----------------------
    # Cross構造化 (長期用)
    # -----------------------

    def expert_to_cross_coords(self, layer: int, expert_id: int) -> Optional[Dict]:
        """
        Expertの知識方向をCross座標に変換

        concept_dirs[flat_id] (4, 7168) → Cross DB エントリ
        4つのSVD方向 → (抽象度X, 応用度Y, 深さZ) にマッピング
        """
        flat_id = (layer - self.FIRST_MOE_LAYER) * self.N_EXPERTS_PER_LAYER + expert_id
        if flat_id >= len(self.concept_dirs):
            return None

        dirs = self.concept_dirs[flat_id]  # (4, 7168)

        # 各方向の「具体性」: embed_tokensとの類似度の分散
        # 分散が大きい → 具体的な概念を指す (高応用度)
        # 分散が小さい → 抽象的な原理 (高抽象度)

        # 簡易版: SVD特異値の比率から推定
        # dirs[0] = 第1主方向 (最重要), dirs[3] = 第4主方向 (最細部)
        primary_norm = float(np.linalg.norm(dirs[0]))
        secondary_ratio = float(np.linalg.norm(dirs[1])) / (primary_norm + 1e-8)

        return {
            "flat_id": flat_id,
            "layer": layer,
            "expert_id": expert_id,
            "primary_dir": dirs[0].tolist(),
            "cross_x": 1.0 - secondary_ratio,   # 抽象度 (1に近い→抽象的)
            "cross_y": secondary_ratio,           # 応用度 (1に近い→具体的)
            "cross_z": float(primary_norm),       # 深さ (重みの大きさ)
        }


# ============================
# Factory / 便利関数
# ============================

def build_expert_loader(
    shard_dir: str,
    concept_dirs_path: str,
    embed_tokens_path: Optional[str] = None,
    index_cache_path: Optional[str] = None,
) -> ExpertLoader:
    """
    ExpertLoaderをワンライナーで構築

    Args:
        shard_dir         : GGUFシャードが入ったディレクトリ
        concept_dirs_path : concept_dirs.npy のパス
        embed_tokens_path : embed_tokens.npy (任意)
        index_cache_path  : インデックスJSONLのキャッシュパス (任意)

    Returns:
        ExpertLoader インスタンス (ロード済み)
    """
    shard_dir = Path(shard_dir)
    cache_p = Path(index_cache_path) if index_cache_path else None

    # テンソルインデックス
    if cache_p and cache_p.exists():
        idx = TensorIndex.load(cache_p)
    else:
        idx = TensorIndex.build(shard_dir, save_path=cache_p)

    # SVD地図
    print(f"[ExpertLoader] Loading concept_dirs from {concept_dirs_path}...")
    concept_dirs = np.load(concept_dirs_path)
    print(f"  shape: {concept_dirs.shape}")  # (15104, 4, 7168)

    # トークン埋め込み (任意)
    embed_tokens = None
    if embed_tokens_path and Path(embed_tokens_path).exists():
        print(f"[ExpertLoader] Mmap embed_tokens...")
        embed_tokens = np.load(embed_tokens_path, mmap_mode="r")
        print(f"  shape: {embed_tokens.shape}")  # (129280, 7168)

    return ExpertLoader(idx, concept_dirs, embed_tokens)


if __name__ == "__main__":
    """
    Quick test: shard 1のみでもRouter + 共有Expertのテストができる
    """
    import sys

    SHARD_DIR = Path.home() / "avh_math/avh_math/downloads/v3_q8_0/Q8_0"
    CONCEPT_DIRS = Path.home() / "avh_math/avh_math/db/moe_sparse_cross_600b_real/concept_dirs.npy"
    EMBED_TOKENS = Path.home() / "avh_math/avh_math/db/moe_sparse_cross_600b_real/embed_tokens.npy"
    INDEX_CACHE  = Path.home() / "avh_math/avh_math/db/v3_tensor_index.jsonl"

    print("=" * 60)
    print("ExpertLoader Quick Test")
    print("=" * 60)

    loader = build_expert_loader(
        shard_dir=str(SHARD_DIR),
        concept_dirs_path=str(CONCEPT_DIRS),
        embed_tokens_path=str(EMBED_TOKENS),
        index_cache_path=str(INDEX_CACHE),
    )

    # テスト: ダミークエリで動作確認
    query = np.random.randn(7168).astype(np.float32)
    query /= np.linalg.norm(query)

    print("\n[Test 1] query_to_experts:")
    experts = loader.query_to_experts(query, top_k=3)
    for layer, eid, score in experts:
        print(f"  layer={layer:2d} expert={eid:3d} score={score:.4f}")

    print("\n[Test 2] Router scores (layer 3):")
    scores = loader.router_scores(query, layer=3)
    if scores is not None:
        top5 = np.argsort(scores)[-5:][::-1]
        for e in top5:
            print(f"  expert={e:3d} score={scores[e]:.4f}")

    print("\n[Test 3] Shared expert transform (layer 3):")
    t = loader.shared_expert_transform(query, layer=3)
    if t is not None:
        print(f"  output shape: {t.shape}, norm: {np.linalg.norm(t):.4f}")
        print(f"  cosine with input: {float(query @ t):.4f}")
    else:
        print("  Not available (shard not downloaded yet)")

    print("\n✅ ExpertLoader test complete")
