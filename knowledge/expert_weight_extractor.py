"""
Expert Weight Extractor — Range Request方式
DeepSeek V3 Q8_0 GGUFから特定Expertの重みだけを取得

設計：
1. GGUFヘッダーをダウンロードしてテンソルoffset取得
2. expert_vocab_domains.jsonからドメイン別Expertを選択
3. HTTP Range Requestで対象Expertバイトのみ取得
4. Q8_0デクワンタイズ → float32
5. Cross探索（SVD / activation analysis）
6. 知識パターンをpiece_dbに変換
"""

import json
import os
import struct
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# =========================================================
# 定数
# =========================================================
REPO_ID  = "unsloth/DeepSeek-V3-GGUF"
SUBDIR   = "DeepSeek-V3-Q8_0"
FILENAME = "DeepSeek-V3-BF16-256x20B-Q8_0-{shard:05d}-of-00016.gguf"
HF_BASE  = f"https://huggingface.co/{REPO_ID}/resolve/main/{SUBDIR}"

# Q8_0ブロック構造: 32要素 + scale (f16 2byte) = 34 bytes/block
Q8_0_BLOCK_ELEMS = 32
Q8_0_BLOCK_BYTES = 34   # 2(f16 scale) + 32(int8)

N_EXPERTS = 256
EXPERT_ELEMS_DOWN  = 2048 * 7168   # ffn_down: (in=2048, out=7168) per expert
EXPERT_ELEMS_GATE  = 7168 * 2048
EXPERT_ELEMS_UP    = 7168 * 2048
EXPERT_BYTES_DOWN  = EXPERT_ELEMS_DOWN  // Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES  # 15,597,568
EXPERT_BYTES_GATE  = EXPERT_ELEMS_GATE  // Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES
EXPERT_BYTES_UP    = EXPERT_ELEMS_UP    // Q8_0_BLOCK_ELEMS * Q8_0_BLOCK_BYTES

# ドメイン→Verantyx Domain マッピング
DOMAIN_TARGETS = {
    "math":       ["math", "calculus", "algebra", "number_theory", "geometry"],
    "physics":    ["physics", "mechanics", "quantum"],
    "cs":         ["computer_science", "programming", "algorithms"],
    "chemistry":  ["chemistry", "molecular"],
    "logic":      ["logic", "reasoning"],
    "biology":    ["biology", "medicine"],
}

CACHE_DIR = Path(__file__).parent / "expert_weight_cache"
CACHE_DIR.mkdir(exist_ok=True)

# =========================================================
# GGUF ヘッダーパーサー（軽量版）
# =========================================================

def _read_gguf_header(data: bytes) -> Tuple[int, Dict[str, dict]]:
    """
    GGUFヘッダーをパースしてテンソルメタデータを返す
    Returns: (data_section_offset, {tensor_name: {offset, shape, dtype}})
    """
    pos = 0

    # Magic + Version
    magic = data[pos:pos+4]
    pos += 4
    if magic != b'GGUF':
        raise ValueError(f"Not a GGUF file: {magic}")
    version = struct.unpack_from('<I', data, pos)[0]
    pos += 4

    # n_tensors, n_kv
    n_tensors = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    n_kv      = struct.unpack_from('<Q', data, pos)[0]; pos += 8

    # Skip KV pairs
    for _ in range(n_kv):
        # key
        key_len = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        pos += key_len
        # value_type
        vtype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        # Skip value (simplified: handle common types)
        pos = _skip_gguf_value(data, pos, vtype)

    # Tensor infos
    tensors = {}
    for _ in range(n_tensors):
        name_len = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        name = data[pos:pos+name_len].decode('utf-8'); pos += name_len
        n_dims = struct.unpack_from('<I', data, pos)[0]; pos += 4
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack_from('<Q', data, pos)[0]); pos += 8
        dtype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        offset = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        tensors[name] = {'offset': offset, 'shape': shape, 'dtype': dtype}

    # Data section aligned to 32 bytes
    data_offset = (pos + 31) & ~31
    return data_offset, tensors


def _skip_gguf_value(data: bytes, pos: int, vtype: int) -> int:
    """
    GGUF value をスキップ（KVパースに使用）
    GGUF型: 0=uint8 1=int8 2=uint16 3=int16 4=uint32 5=int32
             6=float32 7=bool 8=string 9=array 10=uint64 11=int64 12=float64
    """
    fixed_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                   10: 8, 11: 8, 12: 8}
    if vtype in fixed_sizes:
        return pos + fixed_sizes[vtype]
    elif vtype == 8:  # STRING: uint64 len + data
        slen = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        return pos + slen
    elif vtype == 9:  # ARRAY: uint32 elem_type + uint64 count + count*elem
        atype = struct.unpack_from('<I', data, pos)[0]; pos += 4
        count = struct.unpack_from('<Q', data, pos)[0]; pos += 8
        for _ in range(count):
            pos = _skip_gguf_value(data, pos, atype)
        return pos
    else:
        # Unknown type → skip 4 bytes
        return pos + 4


# =========================================================
# Q8_0 デクワンタイズ
# =========================================================

def dequantize_q8_0(raw: bytes, n_elems: int) -> np.ndarray:
    """
    Q8_0バイト列をfloat32配列に変換
    Format: [f16_scale (2B) | 32×int8 (32B)] repeating
    """
    n_blocks = n_elems // Q8_0_BLOCK_ELEMS
    result = np.empty(n_elems, dtype=np.float32)

    for i in range(n_blocks):
        blk = raw[i * Q8_0_BLOCK_BYTES: (i+1) * Q8_0_BLOCK_BYTES]
        scale = np.frombuffer(blk[:2], dtype=np.float16)[0].astype(np.float32)
        quants = np.frombuffer(blk[2:], dtype=np.int8).astype(np.float32)
        result[i*Q8_0_BLOCK_ELEMS:(i+1)*Q8_0_BLOCK_ELEMS] = quants * scale

    return result


# =========================================================
# HuggingFace Range Request
# =========================================================

def _get_hf_url(shard: int) -> str:
    filename = FILENAME.format(shard=shard)
    return f"{HF_BASE}/{filename}"


def fetch_bytes(url: str, start: int, end: int, token: Optional[str] = None) -> bytes:
    """
    HTTP Range Requestで指定バイト範囲を取得
    """
    headers = {"Range": f"bytes={start}-{end-1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # HF LFS redirect
    resp = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60)
    if resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    return resp.content


# =========================================================
# ヘッダー取得（最初のN MBのみ）
# =========================================================

def get_shard_header(shard: int, token: Optional[str] = None) -> Tuple[int, Dict]:
    """シャードのGGUFヘッダーをRange Requestで取得"""
    cache_path = CACHE_DIR / f"header_shard{shard:05d}.json"
    if cache_path.exists():
        d = json.loads(cache_path.read_text())
        return d['data_offset'], d['tensors']

    url = _get_hf_url(shard)
    print(f"  Fetching header for shard {shard}...")

    # 最初の10MBをダウンロード（ヘッダーは通常1-5MB）
    raw = fetch_bytes(url, 0, 10 * 1024 * 1024, token=token)
    data_offset, tensors = _read_gguf_header(raw)

    cache_path.write_text(json.dumps({'data_offset': data_offset, 'tensors': tensors}, indent=2))
    print(f"    data_offset={data_offset}, tensors={len(tensors)}")
    return data_offset, tensors


# =========================================================
# ドメイン別Expertを選択
# =========================================================

def select_domain_experts(
    expert_vocab_path: str,
    top_k: int = 8
) -> Dict[int, List[str]]:
    """
    expert_vocab_domains.json からドメイン別上位Expertを選択
    Returns: {expert_id: [domain1, domain2, ...]}
    """
    with open(expert_vocab_path) as f:
        vocab_data = json.load(f)

    expert_domains: Dict[int, List[str]] = {}

    # フォーマット確認
    if isinstance(vocab_data, list):
        items = vocab_data
    elif isinstance(vocab_data, dict):
        items = list(vocab_data.items())
    else:
        return {}

    # math/physics/cs domainを持つexpertを収集
    target_keywords = set()
    for kws in DOMAIN_TARGETS.values():
        target_keywords.update(kws)

    for item in items:
        if isinstance(item, dict):
            eid_raw = item.get('expert_id', item.get('id', -1))
            domains = item.get('domains', item.get('top_domains', []))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            eid_raw, domains = item
        else:
            continue

        try:
            eid = int(str(eid_raw).split('_')[-1]) % N_EXPERTS
        except:
            continue

        if isinstance(domains, str):
            domains = [domains]
        if isinstance(domains, dict):
            domains = list(domains.keys())

        matched = [d for d in domains if any(kw in str(d).lower() for kw in target_keywords)]
        if matched:
            expert_domains[eid] = matched

    # top_k 選択
    sorted_experts = sorted(expert_domains.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_experts[:top_k])


# =========================================================
# Expert重みをRange Requestで取得
# =========================================================

def fetch_expert_weights(
    shard: int,
    layer: int,
    expert_id: int,
    data_offset: int,
    tensors: Dict,
    token: Optional[str] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    1つのExpertの down/gate/up 重みを取得してデクワンタイズ
    """
    cache_path = CACHE_DIR / f"expert_s{shard:02d}_l{layer:02d}_e{expert_id:03d}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    url = _get_hf_url(shard)
    result = {}

    for proj_type, elems_per_expert, bytes_per_expert in [
        ('down', EXPERT_ELEMS_DOWN, EXPERT_BYTES_DOWN),
        ('gate', EXPERT_ELEMS_GATE, EXPERT_BYTES_GATE),
        ('up',   EXPERT_ELEMS_UP,   EXPERT_BYTES_UP),
    ]:
        tensor_name = f"blk.{layer}.ffn_{proj_type}_exps.weight"
        if tensor_name not in tensors:
            print(f"    {tensor_name} not in shard {shard}")
            continue

        tensor_info = tensors[tensor_name]
        tensor_data_start = data_offset + tensor_info['offset']

        # Expert kのデータ開始位置 (experts は最後のdimで vary slowest in GGUF)
        expert_byte_start = tensor_data_start + expert_id * bytes_per_expert
        expert_byte_end   = expert_byte_start + bytes_per_expert

        print(f"    Fetching {tensor_name} expert={expert_id}: "
              f"{bytes_per_expert/1e6:.1f}MB @ {expert_byte_start/1e9:.2f}GB")

        raw = fetch_bytes(url, expert_byte_start, expert_byte_end, token=token)
        weights = dequantize_q8_0(raw, elems_per_expert)

        # reshape: down=(2048, 7168), gate/up=(7168, 2048)
        if proj_type == 'down':
            weights = weights.reshape(2048, 7168)
        else:
            weights = weights.reshape(7168, 2048)

        result[proj_type] = weights

    if result:
        np.savez_compressed(cache_path, **result)

    return result if result else None


# =========================================================
# Cross探索：SVD + 知識パターン抽出
# =========================================================

def extract_knowledge_patterns(
    expert_weights: Dict[str, np.ndarray],
    embed_tokens_path: str,
    expert_id: int,
    layer: int,
    domain_labels: List[str],
    top_tokens: int = 20
) -> List[Dict]:
    """
    Expert重みからドメイン特化の推論パターンを抽出

    Strategy:
    1. down projection (7168dim) の行ベクトル = token embedding spaceへの出力
    2. top-activating tokens を embed_tokensとの類似度で発見
    3. SVDで主概念方向を抽出
    4. Cross探索: 入力side (gate/up, 2048dim) と出力side (down, 7168dim) の関係
    """
    patterns = []

    # token埋め込みとの類似度（down projectionの出力空間）
    if embed_tokens_path and os.path.exists(embed_tokens_path):
        embed = np.load(embed_tokens_path, mmap_mode='r')  # (129280, 7168)
        down = expert_weights.get('down')  # (2048, 7168)
        if down is not None:
            # down の行を正規化
            down_norm = down / (np.linalg.norm(down, axis=1, keepdims=True) + 1e-8)
            # embed の列を正規化
            embed_norm = embed / (np.linalg.norm(embed, axis=1, keepdims=True) + 1e-8)

            # 各down行のtop similar tokens
            # メモリ節約のため最初の5行のみ使用
            sims = down_norm[:5] @ embed_norm.T  # (5, 129280)
            top_indices = np.argsort(sims, axis=1)[:, -top_tokens:]  # (5, top_tokens)

            patterns.append({
                'type': 'token_activation',
                'expert_id': expert_id,
                'layer': layer,
                'domains': domain_labels,
                'top_token_indices': top_indices.tolist(),
                'confidence': 0.7
            })

    # SVD分析（gate projection: 7168→2048）
    gate = expert_weights.get('gate')
    if gate is not None:
        # SVD: top-4 概念方向
        U, S, Vt = np.linalg.svd(gate[:, :512], full_matrices=False)  # 部分SVD
        concept_dirs = Vt[:4]  # (4, 512) → top-4 principal directions

        patterns.append({
            'type': 'concept_directions',
            'expert_id': expert_id,
            'layer': layer,
            'domains': domain_labels,
            'singular_values': S[:4].tolist(),
            'concept_dirs_shape': concept_dirs.shape,
            'confidence': 0.8
        })

    # Cross探索: 入力→出力の変換パターン
    up   = expert_weights.get('up')    # (7168, 2048)
    down = expert_weights.get('down')  # (2048, 7168)
    if up is not None and down is not None:
        # SwiGLU: out = down(silu(gate(x)) * up(x))
        # ≈ down @ (up) の有効ランクを確認
        cross = down @ up  # (2048, 2048) approximate transformation
        rank_approx = np.linalg.matrix_rank(cross[:64, :64])

        patterns.append({
            'type': 'cross_transform',
            'expert_id': expert_id,
            'layer': layer,
            'domains': domain_labels,
            'approx_rank': int(rank_approx),
            'cross_norm': float(np.linalg.norm(cross)),
            'confidence': 0.75
        })

    return patterns


# =========================================================
# メインパイプライン
# =========================================================

def run_extraction(
    shard_layers: List[Tuple[int, int]],   # [(shard_id, layer_id), ...]
    expert_vocab_path: str,
    embed_tokens_path: str,
    output_path: str,
    token: Optional[str] = None,
    top_experts: int = 8
):
    """
    メイン実行: 複数シャード/レイヤーのExpert重みを取得して知識パターンを抽出
    """
    print("=" * 60)
    print("Expert Weight Extractor — Range Request Mode")
    print("=" * 60)

    # ドメイン別Expert選択
    print("\n[1/4] Selecting domain experts...")
    domain_experts = select_domain_experts(expert_vocab_path, top_k=top_experts)
    print(f"  Selected {len(domain_experts)} target experts: {list(domain_experts.keys())}")

    all_patterns = []

    for shard_id, layer_id in shard_layers:
        print(f"\n[Shard {shard_id}, Layer {layer_id}]")

        # ヘッダー取得
        try:
            data_offset, tensors = get_shard_header(shard_id, token=token)
        except Exception as e:
            print(f"  Header fetch failed: {e}")
            continue

        # 対象Expertの重みを取得
        for expert_id, domains in list(domain_experts.items())[:3]:  # まず3体
            print(f"  Expert {expert_id} (domains={domains[:2]})...")

            try:
                weights = fetch_expert_weights(
                    shard_id, layer_id, expert_id,
                    data_offset, tensors, token=token
                )
                if weights is None:
                    print(f"    Skipped (tensor not in shard)")
                    continue

                # 知識パターン抽出
                patterns = extract_knowledge_patterns(
                    weights, embed_tokens_path,
                    expert_id, layer_id, domains
                )
                all_patterns.extend(patterns)
                print(f"    Extracted {len(patterns)} patterns")

            except Exception as e:
                print(f"    Error: {e}")
                import traceback; traceback.print_exc()

    # 保存
    print(f"\n[4/4] Saving {len(all_patterns)} patterns → {output_path}")
    with open(output_path, 'w') as f:
        for p in all_patterns:
            f.write(json.dumps(p) + '\n')

    print("Done!")
    return all_patterns


if __name__ == '__main__':
    # テスト実行
    EXPERT_VOCAB = os.path.expanduser(
        "~/avh_math/avh_math/db/moe_sparse_cross_600b_real/expert_vocab_domains.json"
    )
    EMBED_TOKENS = os.path.expanduser(
        "~/avh_math/avh_math/db/moe_sparse_cross_600b_real/embed_tokens.npy"
    )
    OUTPUT = str(CACHE_DIR / "extracted_patterns.jsonl")

    # Shard 1のLayer 3, 4のみテスト（小さめ）
    run_extraction(
        shard_layers=[(1, 3), (1, 4)],
        expert_vocab_path=EXPERT_VOCAB,
        embed_tokens_path=EMBED_TOKENS,
        output_path=OUTPUT,
        token=os.environ.get("HF_TOKEN"),
        top_experts=5
    )
