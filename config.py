"""
config.py
Verantyx Knowledge Pipeline — LLM エンドポイント設定

【Mac ローカル】Ollama + Qwen2.5-7B-Instruct（知識補充に最適）
【Thunder Compute】vLLM + Nemotron-70B（将来用）

注意: Nemotron-Orchestrator-8B はツール指揮専用モデル。
      知識抽出には不向き（!!!を返す）。知識補充には Qwen2.5 を使う。
"""

# ── Mac ローカル (Ollama) ──────────────────────────────
VLLM_BASE_URL = "http://localhost:11434/v1"
VLLM_MODEL = "qwen2.5:7b-instruct"    # 知識補充に最適（スキーマ通りのJSON出力確認済み）
VLLM_API_KEY = "ollama"
VLLM_MAX_TOKENS = 2048
VLLM_TEMPERATURE = 0.0
VLLM_TIMEOUT_SEC = 60

# ── Nemotron-Orchestrator-8B（別用途）──────────────────
# ツール指揮・ルーティング用。知識抽出には使わない。
# VLLM_MODEL = "nemotron-orchestrator"

# ── Thunder Compute H100 (vLLM) — 将来用 ──────────────
# VLLM_BASE_URL = "http://localhost:8000/v1"
# VLLM_MODEL = "nvidia/Llama-3.1-Nemotron-70B-Instruct"
# VLLM_API_KEY = "none"
# VLLM_TIMEOUT_SEC = 30
