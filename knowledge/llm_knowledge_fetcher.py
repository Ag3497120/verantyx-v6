"""
llm_knowledge_fetcher.py
Phase 2: Knowledge Query IR → LLM 呼び出し → raw JSON response

問題文は渡さない。Knowledge Query IR だけを渡す。
LLM の差し替えは config.py の URL/MODEL だけ変えれば OK。

対応エンドポイント:
  - Ollama (Mac local):       http://localhost:11434/v1  ← 現在の設定
  - vLLM  (Thunder Compute):  http://localhost:8000/v1   ← config.py でコメント切替
"""

import json
import re
import time
from dataclasses import dataclass, field

from config import (
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_API_KEY,
    VLLM_MAX_TOKENS,
    VLLM_TEMPERATURE,
    VLLM_TIMEOUT_SEC,
)
from knowledge.knowledge_query_ir import (
    KnowledgeQueryIR,
    KNOWLEDGE_SYSTEM_PROMPT,
    build_knowledge_user_prompt,
)

# シングルトンクライアント（セッション内で再利用、初回ロード高速化）
_client = None


# ---------------------------------------------------------------------------
# 1. LLMレスポンスの型
# ---------------------------------------------------------------------------

@dataclass
class LLMKnowledgeResponse:
    query_id: str
    ok: bool
    raw_output: str = ""
    parsed: dict = field(default_factory=dict)
    error: str = ""
    problem_text_shared: bool = False  # 常に False（設計保証 + 監査用）


# ---------------------------------------------------------------------------
# 2. LLM 呼び出し（Ollama / vLLM 対応）
# ---------------------------------------------------------------------------

def _get_client():
    """OpenAI 互換クライアントのシングルトン。"""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=VLLM_TIMEOUT_SEC,
        )
    return _client


def call_knowledge_llm(system: str, user: str, max_tokens: int = VLLM_MAX_TOKENS) -> str:
    """
    Ollama / vLLM の OpenAI 互換エンドポイントに Knowledge Query を投げる。

    Ollama は json_object モードに非対応のモデルがあるため:
      1. json_object モードで試みる
      2. "response_format 未サポート" 系エラーなら text モードで再試行
      3. 3-retry + 指数バックオフ (1s → 2s → 4s)

    Returns:
        LLM のレスポンス文字列（JSON が期待されるが、parse はここではしない）

    Raises:
        Exception: 3 回のリトライ全て失敗した場合
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    retry_delays = [1, 2, 4]
    last_exception: Exception | None = None

    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            # --- まず json_object モードで試みる（vLLM / Ollama 新版）---
            try:
                resp = client.chat.completions.create(
                    model=VLLM_MODEL,
                    messages=messages,
                    temperature=VLLM_TEMPERATURE,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content

            except Exception as json_err:
                # Ollama が response_format を未サポートな場合のみフォールバック
                err_lower = str(json_err).lower()
                if any(kw in err_lower for kw in
                       ("response_format", "unsupported", "unknown field", "json_object")):
                    # text モードで再試行（system prompt が JSON を強制している）
                    resp = client.chat.completions.create(
                        model=VLLM_MODEL,
                        messages=messages,
                        temperature=VLLM_TEMPERATURE,
                        max_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                else:
                    raise  # 別のエラーはそのまま伝搬

        except Exception as e:
            last_exception = e
            if attempt < len(retry_delays):
                time.sleep(delay)
            # 最後のリトライは外側で raise

    # 全リトライ失敗
    raise last_exception or RuntimeError("call_knowledge_llm: unknown error")


# ---------------------------------------------------------------------------
# 3. JSON パーサ（markdown fence / コードブロック除去）
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> tuple[dict | None, str | None]:
    """raw 文字列を JSON に parse する。markdown fence があれば除去。"""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        data = json.loads(cleaned)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"json_parse_error:{e}"


# ---------------------------------------------------------------------------
# 4. Knowledge Fetcher（メインクラス）
# ---------------------------------------------------------------------------

class LLMKnowledgeFetcher:
    """
    KnowledgeQueryIR を受け取り、LLM に knowledge-only モードで問い合わせる。
    問題文は絶対に渡さない（problem_text_shared: False を型・監査の両方で保証）。
    """

    def __init__(self, max_tokens: int = VLLM_MAX_TOKENS):
        self.max_tokens = max_tokens

    def fetch(self, query_ir: KnowledgeQueryIR) -> LLMKnowledgeResponse:
        # 問題文非共有を型レベルで確認（設計保証）
        assert query_ir.problem_text_shared is False, \
            "BUG: problem_text_shared must always be False. Never pass problem_text to LLM."

        user_prompt = build_knowledge_user_prompt(query_ir)

        try:
            raw_output = call_knowledge_llm(
                system=KNOWLEDGE_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            return LLMKnowledgeResponse(
                query_id=query_ir.query_id,
                ok=False,
                raw_output=str(e),
                error=f"llm_call_failed:{e}",
                problem_text_shared=False,
            )

        parsed, parse_error = _parse_json(raw_output)
        return LLMKnowledgeResponse(
            query_id=query_ir.query_id,
            ok=(parsed is not None),
            raw_output=raw_output,
            parsed=parsed or {},
            error=parse_error or "",
            problem_text_shared=False,  # 常に False
        )

    def fetch_batch(self, queries: list[KnowledgeQueryIR]) -> list[LLMKnowledgeResponse]:
        """複数クエリを順次実行（並列化したい場合は asyncio に差し替え可）。"""
        return [self.fetch(q) for q in queries]
