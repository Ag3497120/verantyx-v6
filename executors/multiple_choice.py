"""
Multiple Choice Executor (v2)

Solves A/B/C/D/E format questions.

Key fix (v2): Question stem and choices are separated before scoring.
The old version matched choice words against the full question (which includes the
choices themselves), causing circular scoring bias.
"""
import re
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def split_stem_choices(question: str) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Split a question into (stem, choices_dict).
    Stem = everything before the choice block.
    """
    # Try "Answer Choices:" or "Choices:" header
    header_pat = re.split(
        r'\n\s*(?:Answer\s+Choices?|Options?)\s*:?\s*\n',
        question,
        maxsplit=1,
        flags=re.IGNORECASE
    )
    if len(header_pat) == 2:
        stem = header_pat[0].strip()
        choice_block = header_pat[1]
    else:
        # No header — find the first occurrence of "A." or "A)"
        m = re.search(r'\n\s*[A-Z][.\)]\s+\w', question)
        if m:
            stem = question[:m.start()].strip()
            choice_block = question[m.start():]
        else:
            return question, None

    # Parse individual choices (A-Z support for extended MCQ)
    choices: Dict[str, str] = {}
    lines = choice_block.splitlines()
    current_letter = None
    current_text = []
    for line in lines:
        m = re.match(r'^\s*([A-Z])[.\)]\s+(.*)', line)
        if m:
            if current_letter:
                choices[current_letter] = ' '.join(current_text).strip()
            current_letter = m.group(1).upper()
            current_text = [m.group(2).strip()]
        elif current_letter and line.strip():
            current_text.append(line.strip())
    if current_letter:
        choices[current_letter] = ' '.join(current_text).strip()

    return stem, (choices if len(choices) >= 2 else None)


def parse_choices(question: str) -> Optional[Dict[str, str]]:
    """Extract choices dict from question text (backward-compatible)."""
    _, choices = split_stem_choices(question)
    return choices


# --------------------------------------------------------------------------- #
# Scoring strategies
# --------------------------------------------------------------------------- #

def _score_entity_match(stem: str, choices: Dict[str, str]) -> Dict[str, float]:
    """
    Score choices by matching key entities from the STEM only.
    Stops circular scoring caused by entities appearing in both stem and choices.
    """
    scores = {k: 0.0 for k in choices}

    # Important nouns from stem only (capitalized, 3+ chars)
    stem_entities = set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', stem))
    # Remove common false positives
    STOP_CAPS = {'The', 'This', 'That', 'What', 'Which', 'How', 'Why',
                 'When', 'Where', 'Who', 'Find', 'Give', 'Consider'}
    stem_entities -= STOP_CAPS

    # Content words from stem (4+ chars)
    stem_words = set(re.findall(r'\b[a-z]{4,}\b', stem.lower()))
    STOP_WORDS = {'that', 'this', 'with', 'from', 'have', 'will', 'what',
                  'which', 'does', 'each', 'when', 'then', 'than', 'also',
                  'into', 'been', 'were', 'they', 'their', 'there', 'more',
                  'some', 'such', 'only', 'most', 'very', 'both', 'these'}
    stem_words -= STOP_WORDS

    for letter, text in choices.items():
        text_lower = text.lower()
        # Entity hits (high value)
        for ent in stem_entities:
            if ent.lower() in text_lower:
                scores[letter] += 1.0
        # Content word hits (low value)
        choice_words = set(re.findall(r'\b[a-z]{4,}\b', text_lower))
        shared = stem_words & choice_words
        scores[letter] += len(shared) * 0.2

    return scores


def _score_negation(stem: str, choices: Dict[str, str]) -> Dict[str, float]:
    """
    Detect negation/exception questions and boost less-common-looking choices.
    e.g., "Which is NOT...", "Which does NOT...", "except"
    """
    scores = {k: 0.0 for k in choices}
    neg_pat = re.search(
        r'\b(not|never|except|neither|cannot|false|incorrect|wrong|least)\b',
        stem.lower()
    )
    if not neg_pat:
        return scores
    # In negation questions, the "odd one out" is often shorter or distinct
    # Give a small bonus to the option that shares the fewest words with others
    all_words = [set(re.findall(r'\b\w{3,}\b', t.lower())) for t in choices.values()]
    letters = list(choices.keys())
    for i, letter in enumerate(letters):
        # Score = how many words this option has that others don't share
        unique = all_words[i].copy()
        for j, other in enumerate(all_words):
            if j != i:
                unique -= other
        scores[letter] += len(unique) * 0.3
    return scores


def _score_specificity(choices: Dict[str, str]) -> Dict[str, float]:
    """
    Prefer specific/informative answers over vague ones.
    Longer, more technical answers tend to be correct.
    """
    scores = {k: 0.0 for k in choices}
    lengths = {k: len(v.split()) for k, v in choices.items()}
    max_len = max(lengths.values()) if lengths else 1
    for letter, wcount in lengths.items():
        # Very gentle length preference — reduced to avoid E-bias
        norm = wcount / max_len
        scores[letter] += norm * 0.05
    return scores


def elimination_logic(question: str, choices: Dict[str, str]) -> Optional[str]:
    """
    Eliminate obviously wrong answers.
    Returns the answer if only one remains after elimination.
    """
    stem, _ = split_stem_choices(question)
    remaining = set(choices.keys())
    stem_lower = stem.lower()

    for letter, choice in choices.items():
        choice_lower = choice.lower()
        if 'always' in stem_lower and 'never' in choice_lower:
            remaining.discard(letter)
        if 'none' in stem_lower and 'all' in choice_lower:
            remaining.discard(letter)
        if 'not' in stem_lower and 'is not' in choice_lower:
            remaining.discard(letter)

    if len(remaining) == 1:
        return list(remaining)[0]
    return None


def verify_mcq_by_computation(
    question: str,
    choices: Dict[str, str],
) -> Optional[Tuple[str, float, str]]:
    """
    Verify MCQ options by attempting to compute/verify each choice.

    Strategy:
    1. For each option, try to extract a computational claim
    2. Use executors to verify each claim
    3. Return the option that verifies successfully

    Returns:
        (answer_label, confidence, method) or None
    """
    import re

    stem, _ = split_stem_choices(question)

    # Try to identify if this is a computational MCQ
    # Look for: "what is", "calculate", "find", "equals", numbers in question
    computational_keywords = [
        r'\bwhat\s+is\b', r'\bcalculate\b', r'\bfind\b', r'\bequals?\b',
        r'\bvalue\b', r'\bresult\b', r'\bsum\b', r'\bproduct\b',
        r'\d+', r'\+', r'\*', r'/', r'\^'
    ]

    is_computational = any(re.search(pat, stem.lower()) for pat in computational_keywords)
    if not is_computational:
        return None

    # Try each choice with arithmetic/algebra executors
    verified_options = []

    for label, text in choices.items():
        # Extract numeric value or expression from choice
        # Pattern 1: "42" or "3.14"
        num_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text)
        if num_match:
            value = num_match.group(1)

            # Try to verify this value using computational methods
            # For now, we'll use a simple heuristic: if the stem contains
            # the same operations and this value satisfies them, accept it

            # Check if value appears to be correct based on stem
            # This is a placeholder for more sophisticated verification
            try:
                # Try basic arithmetic evaluation
                if any(op in stem for op in ['+', '-', '*', '/', '^', '=']):
                    # Extract expression from stem
                    expr_match = re.search(r'([0-9+\-*/^()= ]+)', stem)
                    if expr_match:
                        expr = expr_match.group(1).strip()
                        # Simple evaluation (placeholder)
                        # In production, this should use proper executors
                        verified_options.append((label, 0.70, "computational_match"))
            except Exception:
                pass

    # For now, return None to avoid false positives
    # Full implementation would use arithmetic/algebra executors
    return None


# --------------------------------------------------------------------------- #
# Main solver
# --------------------------------------------------------------------------- #

def solve_multiple_choice(
    question: str,
    choices: Optional[Dict[str, str]] = None,
    positive_keywords: Optional[List[str]] = None,
    negative_keywords: Optional[List[str]] = None
) -> Optional[str]:
    """
    Solve multiple choice question (v2 — stem-separated scoring).

    Args:
        question: Full question text (with embedded choices)
        choices: Optional pre-parsed choices dict
        positive_keywords: Optional hint keywords for correct answer
        negative_keywords: Optional hint keywords for wrong answer

    Returns:
        Letter of best answer (A/B/C/D/E)
    """
    # Split stem from choices
    stem, parsed_choices = split_stem_choices(question)
    if choices is None:
        choices = parsed_choices
    if not choices:
        return None  # 選択肢なし → 推論不能

    # Try elimination first
    eliminated = elimination_logic(question, choices)
    if eliminated:
        return eliminated

    # Auto-derive keywords from knowledge_lookup (CS/physics/chemistry/math hints)
    # Only for high-confidence, specific domain knowledge (not general questions)
    if not positive_keywords:
        try:
            from executors.knowledge_lookup import lookup
            kl_result = lookup(question=stem)
            if kl_result and isinstance(kl_result, dict) and kl_result.get('confidence', 0) >= 0.80:
                val = str(kl_result.get('value', ''))
                # Only use for short, specific values (O(log n), constants, etc.)
                if val and len(val) < 20 and kl_result.get('type') in ('constant', 'formula', None):
                    positive_keywords = [val]
        except Exception:
            pass

    # Aggregate scores from multiple strategies
    scores = {k: 0.0 for k in choices}

    # ⛔ POLICY_GATE:STATISTICAL_BIAS
    # position_prior (HLE answer distribution B=22.5%, D=22.3%...) は禁止。
    # 統計的バイアスなし。純粋な推論・知識マッチのみ。

    # 1. Entity/keyword match
    entity_scores = _score_entity_match(stem, choices)
    for k in scores:
        scores[k] += entity_scores.get(k, 0.0) * 1.0  # Optimal: 1.0 (equiv to 1.5, cleaner)

    # 2. Negation pattern
    neg_scores = _score_negation(stem, choices)
    for k in scores:
        scores[k] += neg_scores.get(k, 0.0)

    # 3. Specificity (mild preference for detailed answers)
    spec_scores = _score_specificity(choices)
    for k in scores:
        scores[k] += spec_scores.get(k, 0.0)

    # 4. Explicit keyword hints if provided
    if positive_keywords:
        for letter, text in choices.items():
            text_lower = text.lower()
            for kw in positive_keywords:
                if kw.lower() in text_lower:
                    scores[letter] += 1.5
    if negative_keywords:
        for letter, text in choices.items():
            text_lower = text.lower()
            for kw in negative_keywords:
                if kw.lower() in text_lower:
                    scores[letter] -= 1.5

    # 意味のある信号があるか確認（閾値以下ならスキップ）
    # 純粋な推論信号なしに guess することは禁止
    _MEANINGFUL_SIGNAL_THRESHOLD = 0.15  # specificity(0.05) + αを超える有意な信号
    max_score = max(scores.values()) if scores else 0.0
    if max_score < _MEANINGFUL_SIGNAL_THRESHOLD:
        # 推論できない → None を返してパイプラインがスキップ
        return None

    # Return best (using sorted keys for stable deterministic behavior)
    if scores:
        return max(sorted(scores.keys()), key=lambda k: scores[k])
    return None


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def detect_choice_pattern(question: str) -> bool:
    """Detect if question is multiple choice format."""
    indicators = [
        r'Answer Choices?:?',
        r'\b[A-E][.:\)]\s+\w',
        r'Choose.*[A-E]',
        r'Select.*[A-E]'
    ]
    for pattern in indicators:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    return False


# --------------------------------------------------------------------------- #
# Function registry
# --------------------------------------------------------------------------- #

FUNCTIONS = {
    'solve_multiple_choice': solve_multiple_choice,
    'detect_choice_pattern': detect_choice_pattern,
    'parse_choices': parse_choices
}


def execute(function_name: str, params: Dict[str, Any]) -> Any:
    """Execute multiple choice function"""
    if function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}")
    return FUNCTIONS[function_name](**params)
