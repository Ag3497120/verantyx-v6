# Agent D: LaTeX正規化 + Answer Schema強制

## 目標
Math問題で `None` が返る最大原因: LaTeX記号の抽出失敗 + 答えの型不明。これを修正してMath正解数を増やす。

## ワークスペース
`/Users/motonishikoudai/.openclaw/workspace/verantyx_v6/`

## 現状の問題

### 問題1: LaTeX記号が壊れる
`decomposer/decomposer.py` の entity 抽出で：
- `\mathbb{Z}` → ゴミ
- `x^{2}` → `x` と `2` が分離
- `\frac{1}{2}` → `1` か `2` だけ抽出
- `\sqrt{3}` → 無視
- `$10^{980}$` → `False` (CEGIS が誤答)

### 問題2: 答えの型が不明
HLE Math の答えは：
- `整数` (例: 29010)
- `MCQ` (例: A, B, C, D, E)
- `LaTeX式` (例: $14+2\sqrt{13}$)
- `YesNo` (例: Yes/No/True/False)
- `構成` (例: [cylinder r=6, h=21.5])

型を最初に判定せずに候補生成するから、候補が答えの形式に合わない。

## 実装タスク

### Step 1: LaTeX正規化モジュール作成
`decomposer/latex_normalizer.py` を新規作成:

```python
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
    
    # \frac{a}{b} → a/b
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    
    # \sqrt{x} → sqrt(x)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\sqrt\[([^\]]+)\]\{([^}]+)\}', r'(\2)^(1/\1)', text)
    
    # x^{n} → x^n
    text = re.sub(r'\^\{([^}]+)\}', r'^(\1)', text)
    
    # _{n} → _n (subscripts)
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
```

### Step 2: decomposer.py に LaTeX正規化を統合
`decomposer/decomposer.py` の `decompose()` メソッドを修正:

```python
# decompose() の最初で LaTeX 正規化
from decomposer.latex_normalizer import normalize_latex, detect_answer_schema

def decompose(self, question: str, answer_type: str = '') -> IR:
    # LaTeX 正規化（最初に必ず実行）
    normalized_question = normalize_latex(question)
    
    # answer schema を判定
    schema = detect_answer_schema(question, answer_type)  # 元のテキストで判定
    
    # 以降は normalized_question を使う
    ...
```

### Step 3: pipeline_enhanced.py に answer schema を活用
`pipeline_enhanced.py` の `solve()` で:
- schema='mcq' → MCQ verifier を優先
- schema='integer' → 整数 executorを優先、float は reject
- schema='yesno' → True/False/Yes/No のみ返す
- schema='latex_expr' → answer=None のまま（解けない → 時間無駄にしない）

### Step 4: SyntaxWarning 修正
`quick_eval_hle.py` の先頭にある `SyntaxWarning: invalid decimal literal` を見つけて修正。

```bash
python3 -W error::SyntaxWarning quick_eval_hle.py 2>&1 | head -20
```

### Step 5: テストと評価

```bash
cd /Users/motonishikoudai/.openclaw/workspace/verantyx_v6

# LaTeX正規化テスト
python3 -c "
from decomposer.latex_normalizer import normalize_latex, detect_answer_schema
tests = [
    '\$10^{980}\$',
    '\$14+2\\\\sqrt{13}\$',
    '\\\\mathbb{Z}/n\\\\mathbb{Z}',
    '\\\\frac{1}{2}',
]
for t in tests:
    print(t, '->', normalize_latex(t))
"

# 評価実行
python3 quick_eval_hle.py 2>&1 | tail -20
```

## 完了条件
- `latex_normalizer.py` 作成済み
- `decomposer.py` に統合済み  
- Math スコアが改善している（または None が減っている）
- SyntaxWarning が解消されている

## 結果報告
`AGENT_D_RESULTS.md` に書いて：
```bash
openclaw system event --text "Agent D完了: LaTeX正規化+Schema統合。結果: AGENT_D_RESULTS.md" --mode now
```
