"""
mcq_knowledge_matcher.py
MCQ知識マッチングエンジン（レベル2鉄の壁: IR + 選択肢をLLMに渡す）

設計:
  1. Wikipedia/LLM から取得した facts を受け取る
  2. 各選択肢と facts のキーワード重複・意味的整合性をスコアリング
  3. LLMにはIR（構造化表現）+ 選択肢テキストのみ渡す（問題文本体は渡さない）
  4. 最高スコアの選択肢を返す

鉄の壁レベル2:
  - 問題文 → Decomposer → IR（構造化）→ ここで使う
  - 選択肢テキストはそのまま渡す（選択肢は問題文ではない）
  - Wikipedia facts はパイプライン経由で取得済み
"""

from __future__ import annotations
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MatchScore:
    label: str           # "A", "B", etc.
    text: str            # 選択肢テキスト
    keyword_score: float # キーワード重複スコア
    fact_score: float    # facts との整合性スコア
    llm_score: float     # LLM 判定スコア（レベル2）
    total: float = 0.0
    reasoning: str = ""


def score_choices_against_facts(
    ir_dict: dict,
    choices: Dict[str, str],
    facts: List[dict],
    use_llm: bool = True,
) -> Optional[Tuple[str, float, str]]:
    """
    IRと取得済みfactsを使って各MCQ選択肢をスコアリング。

    Args:
        ir_dict: Decomposer が作った構造化IR
        choices: {"A": "text_a", "B": "text_b", ...}
        facts: [{summary: str, properties: [...], formulas: [...], ...}]
        use_llm: LLMにIR+選択肢を渡すか（レベル2）

    Returns:
        (label, confidence, method) or None
    """
    if not choices or not facts:
        return None

    # ─── Phase 1: キーワードマッチング ───
    # facts からキーワードセットを構築
    fact_keywords = _extract_fact_keywords(facts)

    scores: List[MatchScore] = []
    for label, text in choices.items():
        choice_kws = _extract_keywords(text)

        # キーワード重複
        overlap = fact_keywords & choice_kws
        kw_score = len(overlap) / max(len(choice_kws), 1)

        # facts 文との部分一致スコア
        fact_score = _compute_fact_alignment(text, facts)

        scores.append(MatchScore(
            label=label,
            text=text,
            keyword_score=kw_score,
            fact_score=fact_score,
            llm_score=0.0,
        ))

    # ─── Phase 2: LLMスコアリング（レベル2鉄の壁） ───
    if use_llm and len(facts) > 0:
        llm_scores = _llm_score_choices(ir_dict, choices, facts)
        if llm_scores:
            for ms in scores:
                ms.llm_score = llm_scores.get(ms.label, 0.0)

    # ─── Phase 3: 統合スコア ───
    for ms in scores:
        # 重み: LLM > fact_alignment > keyword
        ms.total = ms.llm_score * 0.6 + ms.fact_score * 0.3 + ms.keyword_score * 0.1

    scores.sort(key=lambda s: s.total, reverse=True)

    if not scores:
        return None

    best = scores[0]
    second = scores[1] if len(scores) > 1 else MatchScore(label="", text="", keyword_score=0, fact_score=0, llm_score=0)

    # 信頼度: トップとセカンドの差
    gap = best.total - second.total
    confidence = min(gap * 2, 1.0)  # gap 0.5 → conf 1.0

    # 閾値: 差が小さすぎたらINCONCLUSIVE
    if gap < 0.05:
        return None

    method = f"mcq_knowledge_match(kw={best.keyword_score:.2f},fact={best.fact_score:.2f},llm={best.llm_score:.2f})"
    return best.label, confidence, method


def _extract_keywords(text: str) -> set:
    """テキストからキーワードを抽出"""
    # 小文字化、記号除去
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    # ストップワード除去
    stops = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'can', 'may', 'might', 'must',
        'and', 'or', 'but', 'not', 'no', 'if', 'then',
        'for', 'with', 'from', 'to', 'in', 'on', 'at', 'by', 'of',
        'it', 'its', 'this', 'that', 'which', 'what', 'who',
        'all', 'each', 'every', 'some', 'any', 'more', 'than',
        'also', 'only', 'very', 'much', 'just',
    }
    return {w for w in words if w not in stops and len(w) > 2}


def _extract_fact_keywords(facts: List[dict]) -> set:
    """facts リストからキーワードセットを構築"""
    all_text = ""
    for f in facts:
        if isinstance(f, dict):
            all_text += " " + f.get("summary", "") + " " + f.get("plain", "")
            for p in f.get("properties", []):
                all_text += " " + str(p)
        elif hasattr(f, 'summary'):
            all_text += " " + (f.summary or "")
    return _extract_keywords(all_text)


def _compute_fact_alignment(choice_text: str, facts: List[dict]) -> float:
    """選択肢テキストと facts の文レベルアライメントスコア"""
    choice_lower = choice_text.lower()
    choice_words = set(choice_lower.split())

    best_score = 0.0
    for f in facts:
        if isinstance(f, dict):
            fact_text = f.get("summary", "") + " " + f.get("plain", "")
        elif hasattr(f, 'summary'):
            fact_text = f.summary or ""
        else:
            continue

        fact_lower = fact_text.lower()
        fact_words = set(fact_lower.split())

        # Jaccard similarity
        intersection = choice_words & fact_words
        union = choice_words | fact_words
        if union:
            jaccard = len(intersection) / len(union)
            best_score = max(best_score, jaccard)

        # サブストリングマッチ（選択肢の主要フレーズが facts に含まれるか）
        # 3-gram マッチング
        choice_ngrams = _ngrams(choice_lower, 3)
        fact_ngrams = _ngrams(fact_lower, 3)
        if choice_ngrams:
            ngram_overlap = len(choice_ngrams & fact_ngrams) / len(choice_ngrams)
            best_score = max(best_score, ngram_overlap)

    return best_score


def _ngrams(text: str, n: int) -> set:
    """テキストからn-gramセットを生成"""
    words = text.split()
    if len(words) < n:
        return set()
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _llm_score_choices(
    ir_dict: dict, choices: Dict[str, str], facts: List[dict]
) -> Optional[Dict[str, float]]:
    """
    LLM にIR + 選択肢 + facts を渡してスコアリング。

    鉄の壁レベル2: 問題文本体は渡さない。IRの構造情報 + 選択肢 + 取得済み知識のみ。
    """
    try:
        import urllib.request

        # IR から構造化情報を抽出（問題文ではない）
        domain = ir_dict.get("domain", ir_dict.get("domain_hint", ["unknown"]))
        if isinstance(domain, list):
            domain = domain[0] if domain else "unknown"
        task = ir_dict.get("task", "unknown")
        entities = ir_dict.get("entities", [])
        entity_str = ", ".join(
            f"{e.get('name', '')}: {e.get('value', '')}"
            for e in entities[:5] if e.get('name')
        )

        # facts を要約
        facts_str = "\n".join(
            f"- {(f.get('summary', '') if isinstance(f, dict) else getattr(f, 'summary', ''))[:200]}"
            for f in facts[:5]
        )

        # 選択肢
        choices_str = "\n".join(f"{k}: {v}" for k, v in sorted(choices.items()))

        # レベル2プロンプト: IR + 選択肢 + facts（問題文は含まない）
        prompt = f"""Given the following structured information, determine which choice best matches the known facts.

Domain: {domain}
Task: {task}
Key entities: {entity_str}

Known facts (from knowledge base):
{facts_str}

Choices:
{choices_str}

Based ONLY on the known facts above, which choice is most consistent?
Reply with ONLY a JSON object: {{"scores": {{"A": 0.0, "B": 0.0, ...}}, "best": "X", "reason": "brief"}}
Where scores are 0.0-1.0 confidence for each choice."""

        # Qwen2.5-7B via Ollama
        from config import LLM_CONFIG
        api_url = LLM_CONFIG.get("base_url", "http://localhost:11434/v1") + "/chat/completions"
        model = LLM_CONFIG.get("model", "qwen2.5:7b-instruct")

        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
        }).encode()

        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]

            # JSON パース
            json_match = re.search(r'\{[^}]*"scores"[^}]*\{[^}]*\}[^}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {k: float(v) for k, v in parsed.get("scores", {}).items()}

            # フォールバック: "best": "X" を探す
            best_match = re.search(r'"best"\s*:\s*"([A-Z])"', content)
            if best_match:
                best_label = best_match.group(1)
                return {label: (1.0 if label == best_label else 0.0) for label in choices}

    except Exception:
        pass

    return None
