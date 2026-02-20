"""
cross_db_full.json (8.7GB) から expert features を streaming 抽出して
compact JSON に変換する。

出力: expert_features_compact.json
  { "L3E0": {"layer": 3, "expert_id": 0, "spectral_entropy": 3.45,
              "effective_rank": 31.7, "top_singular_ratio": 1.04,
              "concept_dirs": [[...893dim...], ...]}, ... }

メモリ: ~500MB (全15,104 experts × 4方向 × 893dim × 4bytes)
時間: ~2-5分
"""
import json
import re
import sys
import time

INPUT  = "/Users/motonishikoudai/Downloads/cross_db_full.json"
OUTPUT = "/Users/motonishikoudai/avh_math/avh_math/db/moe_sparse_cross_600b_real/expert_features_compact.json"

def stream_experts(path):
    """
    ファイルを256KBずつ読んで expert エントリを逐次 yield する。
    フォーマット: "L{layer}E{expert}": { ... }
    """
    pattern = re.compile(
        r'"(L\d+E\d+)":\s*(\{[^}]+\})',
        re.DOTALL
    )
    buf = ""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(262144)  # 256KB
            if not chunk:
                break
            buf += chunk
            # バッファ内のマッチを全部処理
            for m in pattern.finditer(buf):
                key = m.group(1)
                # features だけ抽出（concept_directions は別途）
                yield key, m.start(), m.end()
            # 最後の不完全マッチを保持
            last = buf.rfind('"L')
            if last > 0:
                buf = buf[last:]
            count += 1


def main():
    print("=== Expert Features Extractor ===")
    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT}")
    print()

    # ijson があれば使う、なければ独自 streaming
    try:
        import ijson
        use_ijson = True
        print("Using ijson for streaming parse")
    except ImportError:
        use_ijson = False
        print("ijson not found, using regex streaming")

    experts_out = {}
    t0 = time.time()

    if use_ijson:
        with open(INPUT, "rb") as f:
            # experts オブジェクト配下を iter
            parser = ijson.kvitems(f, "experts")
            for key, val in parser:
                layer = val.get("layer", 0)
                eid   = val.get("expert_id", 0)
                feat  = val.get("features", {})
                dirs  = val.get("concept_directions", [])
                experts_out[key] = {
                    "layer": layer,
                    "expert_id": eid,
                    "spectral_entropy":  feat.get("spectral_entropy", 0),
                    "effective_rank":    feat.get("effective_rank", 0),
                    "top_singular_ratio": feat.get("top_singular_ratio", 0),
                    "frobenius_norm":    feat.get("frobenius_norm", 0),
                    "concept_dirs": dirs,  # 4方向 × 893次元
                }
                if len(experts_out) % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"  {len(experts_out):5d} experts ({elapsed:.1f}s)")
                    sys.stdout.flush()
    else:
        # Fallback: バッファ読み + regex（重い）
        print("WARNING: ijson not available, this will be slow")
        print("Install with: pip3 install ijson --break-system-packages")
        return

    elapsed = time.time() - t0
    print(f"\nTotal: {len(experts_out)} experts in {elapsed:.1f}s")

    # routing_patterns.json も更新
    routing = {
        "total_experts": len(experts_out),
        "layer_count": 61,
        "model": "deepseek-v3-671b",
        "svd_rank": 32,
        "concept_dirs_per_expert": 4,
        "expert_dim": 893,
        "note": "Extracted from H100 SVD analysis (2026-02-17)",
        "experts": {},
    }
    for key, v in experts_out.items():
        routing["experts"][key] = {
            "layer": v["layer"],
            "expert_id": v["expert_id"],
            "spectral_entropy": v["spectral_entropy"],
            "effective_rank": v["effective_rank"],
            "top_singular_ratio": v["top_singular_ratio"],
        }

    routing_path = "/Users/motonishikoudai/avh_math/avh_math/db/moe_sparse_cross_600b_real/routing_patterns.json"
    with open(routing_path, "w") as f:
        json.dump(routing, f, indent=2)
    print(f"routing_patterns.json updated: {len(routing['experts'])} experts")

    # full compact (concept_dirs 含む) は別ファイルへ
    with open(OUTPUT, "w") as f:
        json.dump(experts_out, f)
    print(f"expert_features_compact.json saved")


if __name__ == "__main__":
    main()
