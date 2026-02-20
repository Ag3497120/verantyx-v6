"""
String Operations Executor

String manipulation, ciphers, pattern matching
"""
import re
from typing import Any, Dict, List, Optional


def string_length(text: str) -> int:
    """
    Calculate string length
    
    Args:
        text: Input string
    
    Returns:
        Length of string
    """
    return len(text)


def caesar_cipher(text: str, shift: int, decode: bool = False) -> str:
    """
    Apply Caesar cipher (shift cipher)
    
    Args:
        text: Input text
        shift: Shift amount
        decode: If True, decode (shift backwards)
    
    Returns:
        Encoded/decoded text
    """
    if decode:
        shift = -shift
    
    result = []
    for char in text:
        if char.isalpha():
            # Determine if uppercase or lowercase
            base = ord('A') if char.isupper() else ord('a')
            # Shift and wrap around
            shifted = (ord(char) - base + shift) % 26
            result.append(chr(base + shifted))
        else:
            # Keep non-alphabetic characters as-is
            result.append(char)
    
    return ''.join(result)


def substitution_cipher_solve(
    ciphertext: str,
    known_mappings: Optional[Dict[str, str]] = None,
    frequency_analysis: bool = True
) -> str:
    """
    Attempt to solve substitution cipher
    
    Args:
        ciphertext: Encrypted text
        known_mappings: Optional known character mappings
        frequency_analysis: Use frequency analysis if True
    
    Returns:
        Best guess at plaintext
    """
    # English letter frequency (most common first)
    eng_freq = 'ETAOINSHRDLCUMWFGYPBVKJXQZ'
    
    if known_mappings is None:
        known_mappings = {}
    
    # Character frequency in ciphertext
    char_freq = {}
    for char in ciphertext.upper():
        if char.isalpha():
            char_freq[char] = char_freq.get(char, 0) + 1
    
    # Sort by frequency
    sorted_cipher = sorted(char_freq.keys(), key=lambda x: -char_freq[x])
    
    # Build mapping based on frequency
    mapping = known_mappings.copy()
    for i, cipher_char in enumerate(sorted_cipher):
        if cipher_char not in mapping and i < len(eng_freq):
            mapping[cipher_char] = eng_freq[i]
    
    # Apply mapping
    result = []
    for char in ciphertext:
        if char.upper() in mapping:
            decrypted = mapping[char.upper()]
            result.append(decrypted if char.isupper() else decrypted.lower())
        else:
            result.append(char)
    
    return ''.join(result)


def pattern_match(text: str, pattern: str) -> bool:
    """
    Check if text matches regex pattern
    
    Args:
        text: Input text
        pattern: Regex pattern
    
    Returns:
        True if matches, False otherwise
    """
    try:
        return bool(re.search(pattern, text))
    except re.error:
        return False


def extract_pattern(text: str, pattern: str) -> List[str]:
    """
    Extract all matches of regex pattern
    
    Args:
        text: Input text
        pattern: Regex pattern
    
    Returns:
        List of matched strings
    """
    try:
        return re.findall(pattern, text)
    except re.error:
        return []


def word_count(text: str) -> int:
    """
    Count words in text
    
    Args:
        text: Input text
    
    Returns:
        Number of words
    """
    # Split on whitespace and count non-empty
    return len([w for w in text.split() if w])


def character_count(text: str, char: str) -> int:
    """
    Count occurrences of character in text
    
    Args:
        text: Input text
        char: Character to count
    
    Returns:
        Number of occurrences
    """
    return text.count(char)


def reverse_string(text: str) -> str:
    """
    Reverse a string
    
    Args:
        text: Input string
    
    Returns:
        Reversed string
    """
    return text[::-1]


def is_palindrome(text: str, ignore_spaces: bool = True, 
                   ignore_case: bool = True) -> bool:
    """
    Check if string is a palindrome
    
    Args:
        text: Input string
        ignore_spaces: Ignore whitespace if True
        ignore_case: Ignore case if True
    
    Returns:
        True if palindrome, False otherwise
    """
    processed = text
    
    if ignore_spaces:
        processed = ''.join(processed.split())
    
    if ignore_case:
        processed = processed.lower()
    
    return processed == processed[::-1]


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance (edit distance) between two strings
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Edit distance
    """
    m, n = len(s1), len(s2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Delete
                    dp[i][j-1],      # Insert
                    dp[i-1][j-1]     # Replace
                )
    
    return dp[m][n]


def longest_common_substring(s1: str, s2: str) -> str:
    """
    Find longest common substring
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Longest common substring
    """
    m, n = len(s1), len(s2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    end_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_pos = i
    
    return s1[end_pos - max_len:end_pos]


# Function registry
FUNCTIONS = {
    'string_length': string_length,
    'caesar_cipher': caesar_cipher,
    'substitution_cipher_solve': substitution_cipher_solve,
    'pattern_match': pattern_match,
    'extract_pattern': extract_pattern,
    'word_count': word_count,
    'character_count': character_count,
    'reverse_string': reverse_string,
    'is_palindrome': is_palindrome,
    'levenshtein_distance': levenshtein_distance,
    'longest_common_substring': longest_common_substring
}


def execute(function_name: str, params: Dict[str, Any]) -> Any:
    """Execute string operation function"""
    if function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}")
    
    func = FUNCTIONS[function_name]
    return func(**params)
