"""
LaTeX 正規化モジュール
HLE Math 問題の LaTeX 記号を Verantyx が扱える形に変換
"""
import re


def normalize_latex(text: str) -> str:
    """LaTeX テキストを正規化（壊さない）"""
    # $...$ を除去（数式マーカー）
    text = re.sub(r'\$+', '', text)

    # \mathbb{Z} → Z, \mathbb{R} → R など
    text = re.sub(r'\\mathbb\{([A-Za-z])\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)

    # \frac{a}{b} → (a)/(b)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)

    # \sqrt{x} → sqrt(x)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'(\2)^(1/\1)', text)

    # x^{n} → x^(n)
    text = re.sub(r'\^\{([^}]+)\}', r'^(\1)', text)

    # _{n} → _(n) (subscripts)
    text = re.sub(r'_\{([^}]+)\}', r'_(\1)', text)

    # \cdot → *
    text = text.replace('\\cdot', '*')
    text = text.replace('\\times', '*')
    text = text.replace('\\div', '/')

    # \infty → inf
    text = text.replace('\\infty', 'inf')

    # \leq \geq \neq
    text = text.replace('\\leq', '<=').replace('\\geq', '>=').replace('\\neq', '!=')

    # \forall \exists
    text = text.replace('\\forall', 'forall').replace('\\exists', 'exists')

    # \in \notin
    text = text.replace('\\in', 'in').replace('\\notin', 'not in')

    # 残留 LaTeX コマンドをスペースに
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)

    # 孤立括弧・連続スペースを整理
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def detect_answer_schema(question: str, answer_type: str = '') -> str:
    """
    質問文と answer_type から答えの型を判定
    Returns: 'mcq' | 'integer' | 'float' | 'latex_expr' | 'yesno' | 'construction' | 'text'
    """
    q_lower = question.lower()

    # MCQ判定: (A) (B) などの選択肢が質問中にある
    if re.search(r'[\(\[]\s*[A-E]\s*[\)\]]', question):
        return 'mcq'
    if answer_type == 'multiple_choice':
        return 'mcq'

    # YesNo判定
    yesno_keywords = ['is it true', 'does there exist', 'is there', 'can we',
                       'prove or disprove', 'true or false', 'yes or no']
    if any(kw in q_lower for kw in yesno_keywords):
        return 'yesno'

    # 整数判定
    integer_keywords = ['how many', 'count the', 'find the number', 'what is the number',
                        'number of', 'in how many', 'ways to', 'find all']
    if any(kw in q_lower for kw in integer_keywords):
        return 'integer'

    # 構成判定
    construct_keywords = ['find all', 'construct', 'give an example', 'what are all']
    if any(kw in q_lower for kw in construct_keywords):
        return 'construction'

    # デフォルト
    return 'text'


def extract_numerical_entities(text: str) -> list:
    """LaTeX正規化後のテキストから数値エンティティを抽出"""
    normalized = normalize_latex(text)
    entities = []

    # 整数
    for m in re.finditer(r'\b(-?\d+)\b', normalized):
        entities.append({'type': 'integer', 'value': int(m.group(1)), 'raw': m.group(0)})

    # 小数
    for m in re.finditer(r'\b(-?\d+\.\d+)\b', normalized):
        entities.append({'type': 'float', 'value': float(m.group(1)), 'raw': m.group(0)})

    # 分数 (a)/(b)
    for m in re.finditer(r'\((-?\d+)\)/\((-?\d+)\)', normalized):
        a, b = int(m.group(1)), int(m.group(2))
        if b != 0:
            entities.append({'type': 'fraction', 'value': a/b, 'numerator': a, 'denominator': b})

    return entities
