"""
Answer Matcher - 柔軟な正解判定

HLE問題の多様な解答形式に対応
"""
import re
from typing import Any, Union


def normalize_string(s: str) -> str:
    """
    文字列の正規化
    
    - 空白除去
    - 小文字化
    - LaTeX記法の正規化
    """
    if not isinstance(s, str):
        return str(s)
    
    s = s.strip().lower()
    
    # LaTeX記法の正規化（拡張版）
    latex_replacements = {
        # 集合記号
        r'\\mathbb{z}': 'z',
        r'\\mathbb{r}': 'r',
        r'\\mathbb{q}': 'q',
        r'\\mathbb{n}': 'n',
        r'\\mathbb{c}': 'c',
        r'\\mathbb{p}': 'p',
        # ギリシャ文字
        r'\\pi': 'pi',
        r'\\theta': 'theta',
        r'\\phi': 'phi',
        r'\\alpha': 'alpha',
        r'\\beta': 'beta',
        r'\\gamma': 'gamma',
        r'\\delta': 'delta',
        # 演算子
        r'\\times': '*',
        r'\\cdot': '*',
        r'\\div': '/',
        r'\\frac': '',
        # 括弧・その他
        r'\\left': '',
        r'\\right': '',
        r'\\text{': '',
        r'\\mathrm{': '',
        r'}': '',
        r'\\': '',
        r'\$': '',
    }
    
    for pattern, replacement in latex_replacements.items():
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)
    
    # 余分な空白を除去
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s


def normalize_number(value: Any) -> Union[float, complex, None]:
    """
    数値の正規化（拡張版）
    
    対応形式:
    - 整数・小数: 42, 3.14
    - 分数: 1/2, 3/4
    - パーセント: 50%, 12.5%
    - 科学的記法: 1e-6, 2.5E3
    - 複素数: 3+4i, 3+4j
    
    Returns:
        float, complex, または None
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, complex):
        return value
    
    if isinstance(value, str):
        # 文字列から数値を抽出
        value = value.strip().replace(',', '')  # カンマ除去（例: 1,000 -> 1000）
        
        # パーセント（例: "50%"）
        if '%' in value:
            try:
                num = float(value.replace('%', '').strip())
                return num / 100.0
            except ValueError:
                pass
        
        # 分数（例: "1/2", "3/4"）
        fraction_match = re.match(r'^([+-]?\d+\.?\d*)\s*/\s*([+-]?\d+\.?\d*)$', value)
        if fraction_match:
            try:
                numerator = float(fraction_match.group(1))
                denominator = float(fraction_match.group(2))
                if denominator != 0:
                    return numerator / denominator
            except (ValueError, ZeroDivisionError):
                pass
        
        # × 10^ 記法 (例: "1.23 × 10^-6", "2.5 * 10^3", "1.23 * 10^{-6}")
        sci_match = re.match(r'^([+-]?\d+\.?\d*)\s*[×\*x]\s*10\^[{]?([+-]?\d+)[}]?$', value.strip())
        if sci_match:
            try:
                mantissa = float(sci_match.group(1))
                exp = int(sci_match.group(2))
                return mantissa * (10 ** exp)
            except (ValueError, OverflowError):
                pass

        # infinity
        if value.strip().lower() in ('inf', 'infinity', '∞', 'infty'):
            return float('inf')
        if value.strip().lower() in ('-inf', '-infinity', '-∞'):
            return float('-inf')

        # 複素数（例: "3+4i", "3+4j"）
        complex_match = re.match(r'([+-]?\d+\.?\d*)\s*([+-])\s*(\d+\.?\d*)[ij]', value)
        if complex_match:
            try:
                real = float(complex_match.group(1))
                sign = 1 if complex_match.group(2) == '+' else -1
                imag = sign * float(complex_match.group(3))
                return complex(real, imag)
            except ValueError:
                pass
        
        # 科学的記法 + 通常の数値（例: "1e-6", "2.5E3", "42"）
        try:
            return float(value)
        except ValueError:
            pass
    
    return None


def numbers_match(predicted: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """
    数値の比較（許容誤差あり）
    
    Args:
        predicted: 予測値
        expected: 期待値
        tolerance: 許容誤差
    
    Returns:
        一致するかどうか
    """
    pred_num = normalize_number(predicted)
    exp_num = normalize_number(expected)
    
    if pred_num is None or exp_num is None:
        return False
    
    if isinstance(pred_num, complex) or isinstance(exp_num, complex):
        # 複素数の比較
        return abs(pred_num - exp_num) < tolerance
    
    # 実数の比較 (inf同士は等しい)
    import math
    if math.isinf(pred_num) and math.isinf(exp_num):
        return pred_num == exp_num  # +inf == +inf, -inf == -inf
    return abs(pred_num - exp_num) < tolerance


def flexible_match(predicted: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    """
    柔軟な正解判定
    
    - 数値比較（許容誤差）
    - 文字列比較（正規化）
    - リスト比較（順序無視）
    - 集合比較
    
    Args:
        predicted: 予測値
        expected: 期待値
        tolerance: 数値の許容誤差
    
    Returns:
        一致するかどうか
    """
    # Empty predicted answer never matches non-empty expected
    if isinstance(predicted, str) and predicted.strip() == '':
        if not (isinstance(expected, str) and expected.strip() == ''):
            return False

    # 完全一致
    if predicted == expected:
        return True
    
    # 数値比較
    if numbers_match(predicted, expected, tolerance):
        return True
    
    # 文字列比較（正規化後）
    if isinstance(predicted, str) and isinstance(expected, str):
        pred_norm = normalize_string(predicted)
        exp_norm = normalize_string(expected)
        
        if pred_norm == exp_norm:
            return True
        
        # 文字列内の数値を比較
        pred_num = normalize_number(pred_norm)
        exp_num = normalize_number(exp_norm)
        if pred_num is not None and exp_num is not None:
            return numbers_match(pred_num, exp_num, tolerance)
    
    # 文字列 vs 数値
    if isinstance(predicted, str):
        pred_num = normalize_number(predicted)
        if pred_num is not None and numbers_match(pred_num, expected, tolerance):
            return True
    
    if isinstance(expected, str):
        exp_num = normalize_number(expected)
        if exp_num is not None and numbers_match(predicted, exp_num, tolerance):
            return True
    
    # リスト比較（順序無視）
    if isinstance(predicted, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(predicted) != len(expected):
            return False
        
        # ソートして比較
        try:
            return sorted(str(p) for p in predicted) == sorted(str(e) for e in expected)
        except:
            pass
    
    # 集合比較
    if isinstance(predicted, set) and isinstance(expected, set):
        return predicted == expected
    
    # 多肢選択問題の文字（A, B, C, D, E）
    if isinstance(predicted, str) and isinstance(expected, str):
        pred_clean = predicted.strip().upper()
        exp_clean = expected.strip().upper()
        
        # 単一文字の場合
        if len(pred_clean) == 1 and len(exp_clean) == 1:
            return pred_clean == exp_clean
        
        # "Option A" vs "A" などの場合
        if pred_clean.endswith(exp_clean) or exp_clean.endswith(pred_clean):
            return True
    
    # Yes/No, True/Falseなどのboolean値
    if isinstance(predicted, str) and isinstance(expected, str):
        boolean_map = {
            'yes': True, 'no': False,
            'true': True, 'false': False,
            'correct': True, 'incorrect': False,
            '1': True, '0': False
        }
        pred_bool = boolean_map.get(predicted.strip().lower())
        exp_bool = boolean_map.get(expected.strip().lower())
        if pred_bool is not None and exp_bool is not None:
            return pred_bool == exp_bool
    
    # 最後の手段: 文字列化して比較
    return str(predicted).strip().lower() == str(expected).strip().lower()
