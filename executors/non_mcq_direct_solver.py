"""
non_mcq_direct_solver.py
non-MCQ直接回答器 — Qwen 7B で鉄の壁レベル2準拠

鉄の壁レベル2: IR構造 + retrievedファクト → Qwen → 短答
問題文は渡さない。IR フィールド（domain, entities, constraints, query, missing）+ facts のみ。

対象:
  - 数値 (integers, floats, fractions)
  - Yes/No/True/False
  - 短い固有名詞 (人名, 地名, etc.)

設計原則:
  - 問題文は LLM に渡さない（鉄の壁）
  - IR の構造化フィールド + Wikipedia facts のみ使用
  - 回答が短く構造化されている場合のみ採用
  - INCONCLUSIVE > wrong answer (HLE no-penalty)
"""

from __future__ import annotations
import json
import re
import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# 最大facts使用数
MAX_FACTS_TO_USE = 8


def solve_non_mcq_directly(
    ir_dict: dict,
    facts: Optional[List[dict]] = None,
    answer_schema: str = "free_form",
) -> Optional[Tuple[str, float, str]]:
    """
    Qwen 7B に直接 non-MCQ 問題を解かせる（鉄の壁レベル2）。

    Args:
        ir_dict: IR の to_dict() 出力
        facts: Wikipedia 等から取得したファクト
        answer_schema: 期待される回答形式

    Returns:
        (answer, confidence, method) or None
    """
    facts = facts or []
    if not facts:
        # factsなしでは回答不可（Wikipedia知識が必要）
        return None

    try:
        answer = _ask_qwen_non_mcq(ir_dict, facts, answer_schema)
        if answer:
            # Validate answer format
            cleaned = answer.strip()

            # Block garbage: too long, or definition-like
            if len(cleaned) > 50:
                return None
            if any(cleaned.lower().startswith(p) for p in (
                'the ', 'a ', 'an ', 'this ', 'that ', 'which ', 'where ',
                'it is', 'it was', 'they ', 'there ',
            )):
                return None
            # Block if answer has too many words (likely a sentence, not a short answer)
            if len(cleaned.split()) > 6:
                return None

            # Confidence based on facts count and answer format
            conf = 0.30
            if len(facts) >= 3:
                conf += 0.10
            if len(facts) >= 5:
                conf += 0.05
            # Boost for well-structured answers
            if re.match(r'^-?\d+(\.\d+)?$', cleaned):
                conf += 0.10  # numeric answer
            elif re.match(r'^-?\d+/\d+$', cleaned):
                conf += 0.10  # fraction
            elif cleaned.lower() in ('yes', 'no', 'true', 'false'):
                conf += 0.10  # boolean

            method = f"non_mcq_direct:qwen7b(facts={len(facts)},schema={answer_schema})"
            log.info(f"non_mcq_direct: answer={cleaned} conf={conf:.2f}")
            return cleaned, conf, method
    except Exception as e:
        log.debug(f"non_mcq_direct error: {e}")

    return None


def _build_context(ir_dict: dict) -> str:
    """IR dict からコンテキスト文字列を構築（問題文は含まない）"""
    parts = []

    domain = ir_dict.get("domain", "")
    if domain and domain not in ("unknown", "Domain.MULTIPLE_CHOICE", "multiple_choice"):
        parts.append(f"Domain: {domain}")

    entities = ir_dict.get("entities", [])
    if entities:
        ent_names = []
        for e in entities[:8]:
            name = e.get("name", "") if isinstance(e, dict) else str(e)
            if name:
                ent_names.append(name)
        if ent_names:
            parts.append(f"Key concepts: {', '.join(ent_names)}")

    constraints = ir_dict.get("constraints", [])
    if constraints:
        con_strs = []
        for c in constraints[:4]:
            if isinstance(c, dict):
                expr = c.get("expression", "") or c.get("lhs", "")
                if expr:
                    con_strs.append(str(expr))
        if con_strs:
            parts.append(f"Constraints: {'; '.join(con_strs)}")

    # Task/query
    task = ir_dict.get("task", "")
    if task:
        parts.append(f"Task: {task}")

    missing = ir_dict.get("missing", [])
    if missing:
        miss_parts = []
        for m in missing[:6]:
            if isinstance(m, dict):
                concept = m.get("concept", "")
                kind = m.get("kind", "")
                if concept:
                    miss_parts.append(f"{concept} ({kind})" if kind else concept)
            elif isinstance(m, str):
                miss_parts.append(m)
        if miss_parts:
            parts.append(f"Topic area: {', '.join(miss_parts)}")

    # metadata keywords
    metadata = ir_dict.get("metadata", {})
    keywords = metadata.get("keywords", [])
    meaningful_kws = [k for k in keywords if k not in ("multiple", "choice", "question", "answer")]
    if meaningful_kws:
        parts.append(f"Keywords: {', '.join(meaningful_kws[:8])}")

    return "\n".join(parts) if parts else "Domain: unknown"


def _build_facts_str(facts: List[dict]) -> str:
    """facts からコンパクトな文字列を生成"""
    lines = []
    for idx, f in enumerate(facts[:MAX_FACTS_TO_USE]):
        if isinstance(f, dict):
            s = (f.get("summary", "") or f.get("plain", ""))[:250].strip()
            props = f.get("properties", [])
            if props:
                s += " | " + "; ".join(str(p) for p in props[:2])
            formulas = f.get("formulas", [])
            if formulas:
                s += " | " + "; ".join(str(fl) for fl in formulas[:1])
            numeric = f.get("numeric", {})
            if numeric:
                s += " | " + "; ".join(f"{k}={v}" for k, v in list(numeric.items())[:3])
        elif hasattr(f, "summary"):
            s = (f.summary or "")[:250].strip()
        else:
            continue
        if s:
            lines.append(f"[F{idx}] {s}")
    return "\n".join(lines)


def _ask_qwen_non_mcq(
    ir_dict: dict,
    facts: List[dict],
    answer_schema: str,
) -> Optional[str]:
    """Qwen 7B に直接 non-MCQ を解かせる"""
    import urllib.request
    from config import VLLM_BASE_URL, VLLM_MODEL

    context_str = _build_context(ir_dict)
    facts_str = _build_facts_str(facts)

    # Answer format hints based on schema
    schema_hint = ""
    if answer_schema in ("integer", "numeric"):
        schema_hint = "The answer should be a number (integer)."
    elif answer_schema == "decimal":
        schema_hint = "The answer should be a decimal number."
    elif answer_schema == "boolean":
        schema_hint = "The answer should be Yes or No."
    elif answer_schema == "fraction":
        schema_hint = "The answer should be a fraction (e.g., 3/7)."
    else:
        schema_hint = "Give a short, precise answer (a number, name, or brief phrase)."

    prompt = f"""You are a PhD-level expert. Based on the context and facts below, answer the question.

Context:
{context_str}

Retrieved facts:
{facts_str}

{schema_hint}

Instructions:
- Use ONLY the facts provided to derive your answer.
- Give ONLY the final answer, nothing else.
- If the answer is a number, give just the number.
- If the answer is a name, give just the name.
- If unsure, respond with "UNSURE".

Answer:"""

    api_url = VLLM_BASE_URL + "/chat/completions"
    payload = json.dumps({
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50,
    }).encode()

    req = urllib.request.Request(
        api_url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"].strip()

    # Filter out "UNSURE" or empty
    if not content or content.upper() in ("UNSURE", "I DON'T KNOW", "UNKNOWN", "N/A"):
        return None

    # Clean up common prefixes
    content = re.sub(r'^(?:The answer is|Answer:|The answer:)\s*', '', content, flags=re.I)
    content = content.strip().rstrip('.')

    return content if content else None
