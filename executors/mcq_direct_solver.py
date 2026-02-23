"""
mcq_direct_solver.py
直接MCQ回答器 — Qwen 7B で鉄の壁レベル2準拠

鉄の壁レベル2: IR構造 + 選択肢 + retrievedファクト → Qwen → 直接回答
問題文は渡さない。IR フィールド（domain, entities, constraints, query, missing）+ 選択肢 + facts のみ。

設計:
  - km_v2, elimination, SymPy が全て INCONCLUSIVE の後の最終フォールバック
  - Qwen 7B の事前知識 + retrieved facts を組み合わせて答える
  - HLEのMCQ 40% (≈1000問) に適用 → 25%正解 = +250問 = +10%スコア改善期待
  - no-penalty: 誤答でもペナルティなし → 積極的回答は損しない

鉄の壁違反の防止:
  - `problem_text` は絶対に Qwen に渡さない
  - IR の metadata.normalized_text も渡さない
  - IR の structured fields (entities, constraints, query, missing) のみ使用
"""

from __future__ import annotations
import json
import re
import logging
import random
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# 最小facts数（0でも発火する — Qwen の事前知識で答える）
MIN_FACTS_TO_FIRE = 0
# facts がある場合の最大使用数
MAX_FACTS_TO_USE = 10


def solve_mcq_directly(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: Optional[List[dict]] = None,
) -> Optional[Tuple[str, float, str]]:
    """
    Qwen 7B に直接 MCQ を解かせる（鉄の壁レベル2）。

    Args:
        ir_dict: IR の to_dict() 出力
        choices: {"A": "...", "B": "...", ...}
        facts: Wikipedia 等から取得したファクト（省略可）

    Returns:
        (answer_label, confidence, method) or None
    """
    if not choices or len(choices) < 2:
        return None

    facts = facts or []

    try:
        answer = _ask_qwen_direct(ir_dict, choices, facts)
        if answer and answer.upper() in choices:
            label = answer.upper()
            # confidence は facts の豊富さに応じて設定
            # Raised: mcq_direct with facts had 4/6 correct in 50q test (66.7%)
            # Needs to be competitive with cross_decompose to win when both fire
            if not facts:
                conf = 0.30
            elif len(facts) >= 3:
                conf = min(0.55, 0.35 + 0.03 * len(facts))
            else:
                conf = min(0.45, 0.30 + 0.03 * len(facts))
            method = f"mcq_direct:qwen7b(facts={len(facts)},choices={len(choices)})"
            log.info(f"mcq_direct: answer={label} conf={conf:.2f} facts={len(facts)}")
            return label, conf, method
    except Exception as e:
        log.debug(f"mcq_direct error: {e}")

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

    # metadata keywords (but NOT normalized_text = problem text)
    metadata = ir_dict.get("metadata", {})
    keywords = metadata.get("keywords", [])
    meaningful_kws = [k for k in keywords if k not in ("multiple", "choice", "question", "answer")]
    if meaningful_kws:
        parts.append(f"Keywords: {', '.join(meaningful_kws[:8])}")

    # LLM IR fields (may be set by LLM decomposer)
    llm_missing = ir_dict.get("llm_missing", [])
    if llm_missing:
        lm_strs = [str(m) for m in llm_missing[:5] if m]
        if lm_strs:
            parts.append(f"LLM topics: {', '.join(lm_strs)}")

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
        elif hasattr(f, "summary"):
            s = (f.summary or "")[:250].strip()
        else:
            continue
        if s:
            lines.append(f"[F{idx}] {s}")
    return "\n".join(lines)


def _ask_qwen_direct(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: List[dict],
) -> Optional[str]:
    """Qwen 7B に直接 MCQ を解かせる"""
    import urllib.request
    from config import VLLM_BASE_URL, VLLM_MODEL

    context_str = _build_context(ir_dict)
    facts_str = _build_facts_str(facts)

    # 選択肢をシャッフルして position bias を排除
    choice_items = list(choices.items())
    random.shuffle(choice_items)
    choices_str = "\n".join(f"  {k}: {v}" for k, v in choice_items)

    # ファクトがある場合とない場合でプロンプトを変える
    if facts_str:
        prompt = f"""You are a PhD-level expert answering a multiple-choice question. Use the provided context and retrieved facts to select the best answer.

Context (structured, not problem text):
{context_str}

Retrieved facts:
{facts_str}

Answer choices:
{choices_str}

Instructions:
- Carefully analyze each choice against the facts and context.
- Eliminate choices that contradict the facts.
- Select the choice that is most consistent with the evidence.
- Think step by step, then give your final answer.
- Your response MUST end with "Answer: X" where X is a single letter.

Reasoning:"""
    else:
        prompt = f"""You are a PhD-level expert answering a multiple-choice question. Use the provided context to select the best answer.

Context (structured, not problem text):
{context_str}

Answer choices:
{choices_str}

Instructions:
- Use your knowledge of the topic area to analyze each choice.
- Think step by step, then give your final answer.
- Your response MUST end with "Answer: X" where X is a single letter.

Reasoning:"""

    api_url = VLLM_BASE_URL + "/chat/completions"
    payload = json.dumps({
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 200,  # increased from 10 to allow chain-of-thought reasoning
    }).encode()

    req = urllib.request.Request(
        api_url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"].strip()

    # Extract "Answer: X" pattern from chain-of-thought response
    m = re.search(r'[Aa]nswer[:\s]+([A-Za-z])\b', content)
    if m:
        letter = m.group(1).upper()
        if letter in choices:
            return letter

    # Fallback: extract letter from the very end of content
    m = re.search(r'\b([A-Z])\s*$', content)
    if m and m.group(1) in choices:
        return m.group(1)

    # Fallback: first letter at the start
    m = re.match(r'^([A-Z])', content.upper())
    if m and m.group(1) in choices:
        return m.group(1)

    # Last resort: any standalone letter that matches a choice
    for m in re.finditer(r'\b([A-Z])\b', content.upper()):
        if m.group(1) in choices:
            return m.group(1)

    return None
