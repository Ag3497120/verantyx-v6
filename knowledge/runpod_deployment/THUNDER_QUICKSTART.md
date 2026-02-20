# Thunder Compute クイックスタート
## 8× A100 80GB / FP8 / DeepSeek V3 知識抽出

---

## 1. Thunder Compute インスタンス設定

| 項目 | 値 |
|------|---|
| GPU | A100 80GB × **8台** |
| ストレージ | **800GB** |
| 時間 | 6時間（目安） |
| 費用 | ~$37（$0.78 × 8 × 6h） |

---

## 2. インスタンス起動後 — 最初に実行するコマンド

```bash
# ① プロジェクトコードを転送（ローカルMacから）
rsync -avz --exclude='__pycache__' --exclude='*.pyc' \
  /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/ \
  <thunder_ip>:/workspace/verantyx_v6/

# ② Thunder インスタンス内で実行
cd /workspace/verantyx_v6/knowledge/runpod_deployment
MODEL_VARIANT=fp8 bash setup.sh
```

setup.sh が行うこと:
- pip install (torch, vllm, transformers, safetensors...)
- DeepSeek V3 FP8 (~670GB) を HuggingFace からダウンロード
- GPU確認

---

## 3. ダウンロード完了後 — 知識抽出を実行

```bash
cd /workspace/verantyx_v6

# FP8 / 8GPU で本番実行
python knowledge/runpod_deployment/run_extraction.py \
  --model fp8 \
  --model-path /workspace/models/deepseek-v3-fp8/ \
  --probes 15 \
  --output pieces/pieces_600b_extracted.jsonl
```

実行時間の目安:
- モデルロード: ~15分
- Expert routing分析: ~2時間
- セマンティック知識抽出: ~2時間
- **合計: ~4-5時間**

---

## 4. 結果をローカルに持ち帰る

```bash
# Thunder → Mac
scp <thunder_ip>:/workspace/verantyx_v6/pieces/pieces_600b_extracted.jsonl \
  /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/pieces/

# または routing_cache.json も保存（再実行コスト削減）
scp <thunder_ip>:/workspace/verantyx_v6/knowledge/runpod_deployment/routing_cache.json \
  /Users/motonishikoudai/.openclaw/workspace/verantyx_v6/knowledge/runpod_deployment/
```

---

## 5. Verantyx に統合してHLE評価

```bash
# ローカルMacで
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6
python quick_eval_hle.py
```

---

## トラブルシューティング

### FP8がVRAMに収まらない場合
```bash
# cpu_offload_gbを増やす（run_extraction.py内の設定）
# または max_model_len を 1024 に下げる
```

### HuggingFaceダウンロードが遅い/失敗する場合
```bash
# HF_TOKEN を設定（アカウントが必要）
export HF_TOKEN=<your_token>
huggingface-cli download deepseek-ai/DeepSeek-V3 \
  --local-dir /workspace/models/deepseek-v3-fp8 \
  --resume-download
```

### 途中で止まった場合（routing_cache.jsonが存在すれば再開可能）
```bash
python knowledge/runpod_deployment/run_extraction.py \
  --model fp8 \
  --model-path /workspace/models/deepseek-v3-fp8/ \
  --skip-routing  # ルーティング分析をスキップ（キャッシュ使用）
```

---

*最終更新: 2026-02-17*
