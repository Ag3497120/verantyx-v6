"""
MCQ Option Verifier - Computational verification of MCQ options

Strategy:
1. For computational MCQs, evaluate each option numerically
2. For logical MCQs, use elimination and consistency checking
3. NO statistical bias (no position prior)
"""

import re
from typing import Any, Dict, Optional, Tuple, List


def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from option text"""
    # Pattern 1: Simple number "42" or "3.14"
    match = re.search(r'^\s*(-?\d+(?:\.\d+)?)\s*$', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Pattern 2: Number with units "42 meters"
    match = re.search(r'^\s*(-?\d+(?:\.\d+)?)\s+\w+', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Pattern 3: Fraction notation
    match = re.search(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', text)
    if match:
        try:
            num, den = int(match.group(1)), int(match.group(2))
            if den != 0:
                return num / den
        except (ValueError, ZeroDivisionError):
            pass

    return None


def verify_option_by_computation(
    stem: str,
    option_label: str,
    option_text: str,
) -> Tuple[bool, float, str]:
    """
    Verify if an MCQ option is correct using computation.

    Returns:
        (is_verified, confidence, reason)
    """
    # Strategy 1: Extract computational expression from stem
    # Look for "what is X" or "calculate Y" patterns

    # Check for direct computation
    computation_patterns = [
        r'what\s+is\s+([\d\s\+\-\*/\(\)]+)',
        r'calculate\s+([\d\s\+\-\*/\(\)]+)',
        r'evaluate\s+([\d\s\+\-\*/\(\)]+)',
        r'=\s*\?\s*$',  # Ends with = ?
        r'find\s+([\d\s\+\-\*/\(\)]+)',
    ]

    for pattern in computation_patterns:
        match = re.search(pattern, stem.lower())
        if match:
            # Try to extract value from option
            option_value = extract_numeric_value(option_text)
            if option_value is not None:
                # Try to evaluate expression (simple cases only)
                try:
                    if match.lastindex and match.lastindex >= 1:
                        expr = match.group(1).strip()
                        # Safety: only allow basic arithmetic
                        if all(c in '0123456789+-*/().' or c.isspace() for c in expr):
                            computed = eval(expr)  # noqa: S307
                            if abs(computed - option_value) < 1e-9:
                                return True, 0.85, f"computation_match:{expr}={computed}"
                except Exception:
                    pass

    # Strategy 2: Boolean/True-False questions
    if any(kw in stem.lower() for kw in ['true or false', 'is it true', 'correct statement']):
        option_lower = option_text.lower().strip()
        if option_lower in ['true', 'false', 'yes', 'no']:
            # Cannot auto-verify boolean without domain knowledge
            return False, 0.0, "boolean_requires_knowledge"

    # Strategy 3: Use domain-specific executors
    # For arithmetic questions, try to use arithmetic executor
    if any(kw in stem.lower() for kw in ['sum', 'product', 'difference', 'quotient', 'remainder', 'modulo']):
        option_value = extract_numeric_value(option_text)
        if option_value is not None:
            # Try to extract numbers from stem and verify operation
            numbers = [float(n) for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', stem)]
            if len(numbers) >= 2:
                # Check common operations
                if 'sum' in stem.lower() and abs(sum(numbers) - option_value) < 1e-9:
                    return True, 0.80, f"sum_verified:{sum(numbers)}"
                if 'product' in stem.lower():
                    prod = 1
                    for n in numbers:
                        prod *= n
                    if abs(prod - option_value) < 1e-9:
                        return True, 0.80, f"product_verified:{prod}"

    return False, 0.0, "no_verification"


def verify_mcq_by_executor(
    stem: str,
    choices: Dict[str, str],
) -> Optional[Tuple[str, float, str]]:
    """
    Verify MCQ options using executor-based verification.

    This is a bias-free approach that only returns an answer if
    we can computationally verify one option.

    Returns:
        (answer_label, confidence, method) or None
    """
    verified = []
    disproved = []

    for label, text in sorted(choices.items()):
        is_verified, conf, reason = verify_option_by_computation(stem, label, text)

        if is_verified:
            verified.append((label, conf, reason))
        elif conf < 0:
            disproved.append(label)

    # Return if exactly one option verified
    if len(verified) == 1:
        label, conf, reason = verified[0]
        return label, conf, f"mcq_executor_verified:{reason}"

    # Return if all but one disproved
    if len(disproved) == len(choices) - 1:
        remaining = [l for l in choices if l not in disproved][0]
        return remaining, 0.75, "mcq_elimination"

    # Strategy 2: Try using specialized executors for complex questions
    result = try_executor_based_verification(stem, choices)
    if result:
        return result

    return None


def try_executor_based_verification(
    stem: str,
    choices: Dict[str, str],
) -> Optional[Tuple[str, float, str]]:
    """
    Try to verify MCQ using existing executors (arithmetic, algebra, etc.)

    Returns:
        (answer_label, confidence, method) or None
    """
    # Try arithmetic executor
    try:
        from executors.arithmetic import execute as arithmetic_execute

        # Look for arithmetic keywords
        if any(kw in stem.lower() for kw in ['sum', 'product', 'divide', 'multiply', 'add', 'subtract']):
            # Extract operation and numbers from stem
            numbers = [float(n) for n in re.findall(r'\b(\d+(?:\.\d+)?)\b', stem)]
            if len(numbers) >= 2:
                # Try each operation
                operations = {
                    'sum': sum(numbers),
                    'product': lambda nums: eval('*'.join(str(n) for n in nums)),  # noqa: S307
                }

                for op_name, op_func in operations.items():
                    if op_name in stem.lower():
                        try:
                            expected = op_func(numbers) if callable(op_func) else op_func

                            # Find matching option
                            for label, text in choices.items():
                                opt_val = extract_numeric_value(text)
                                if opt_val is not None and abs(opt_val - expected) < 1e-9:
                                    return label, 0.80, f"arithmetic_executor:{op_name}"
                        except Exception:
                            pass
    except ImportError:
        pass

    # Try algebra executor for equation solving
    try:
        from executors.algebra import execute as algebra_execute

        # Look for equation keywords
        if any(kw in stem.lower() for kw in ['solve', 'equation', 'find x', 'value of']):
            # Extract simple equations like "2x + 3 = 7"
            eq_match = re.search(r'([0-9x\+\-\*/\^= ]+)', stem)
            if eq_match:
                eq = eq_match.group(1).strip()
                if '=' in eq and 'x' in eq.lower():
                    # Try to solve using algebra executor
                    try:
                        result = algebra_execute('solve_linear_equation', {'equation': eq})
                        if result and 'value' in result:
                            solution = result['value']

                            # Find matching option
                            for label, text in choices.items():
                                opt_val = extract_numeric_value(text)
                                if opt_val is not None and abs(opt_val - solution) < 1e-9:
                                    return label, 0.80, f"algebra_executor:solve"
                    except Exception:
                        pass
    except ImportError:
        pass

    return None


# Function registry
FUNCTIONS = {
    'verify_mcq_by_executor': verify_mcq_by_executor,
    'extract_numeric_value': extract_numeric_value,
}


def execute(function_name: str, params: Dict[str, Any]) -> Any:
    """Execute MCQ verifier function"""
    if function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}")
    return FUNCTIONS[function_name](**params)
