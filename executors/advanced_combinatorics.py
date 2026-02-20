"""
Advanced Combinatorics Executor

Stirling numbers, partitions, and advanced counting
"""
import math
from typing import Any, Dict, Optional


def stirling_first(n: int = None, k: int = None) -> int:
    """
    Stirling number of the first kind: s(n,k)
    Number of permutations of n elements with k cycles

    Args:
        n: Number of elements
        k: Number of cycles

    Returns:
        Stirling number s(n,k)
    """
    if n is None or k is None:
        return None
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    
    # Use recurrence: s(n,k) = s(n-1,k-1) - (n-1)*s(n-1,k)
    # Build DP table
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, min(i + 1, k + 1)):
            dp[i][j] = dp[i-1][j-1] - (i-1) * dp[i-1][j]
    
    return abs(dp[n][k])  # Return unsigned version


def stirling_second(n: int, k: int) -> int:
    """
    Stirling number of the second kind: S(n,k)
    Number of ways to partition n elements into k non-empty subsets
    
    Args:
        n: Number of elements
        k: Number of subsets
    
    Returns:
        Stirling number S(n,k)
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0
    
    # Use recurrence: S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    
    for i in range(1, n + 1):
        for j in range(1, min(i + 1, k + 1)):
            dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
    
    return dp[n][k]


def bell_number(n: int) -> int:
    """
    Bell number: B(n)
    Number of ways to partition n elements (into any number of subsets)
    
    Args:
        n: Number of elements
    
    Returns:
        Bell number B(n)
    """
    if n == 0:
        return 1
    
    # B(n) = sum_{k=0}^{n} S(n,k)
    # Use Bell triangle for efficiency
    bell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    bell[0][0] = 1
    
    for i in range(1, n + 1):
        # First element is same as last element of previous row
        bell[i][0] = bell[i-1][i-1]
        
        for j in range(1, i + 1):
            bell[i][j] = bell[i-1][j-1] + bell[i][j-1]
    
    return bell[n][0]


def catalan_number(n: int) -> int:
    """
    Catalan number: C(n)
    Counts many combinatorial structures (binary trees, parenthesizations, etc.)
    
    Args:
        n: Index
    
    Returns:
        Catalan number C(n)
    """
    if n <= 1:
        return 1
    
    # C(n) = (2n)! / ((n+1)! * n!)
    # Equivalent: C(n) = C(2n, n) / (n+1)
    return math.comb(2 * n, n) // (n + 1)


def partition_count(n: int, k: Optional[int] = None) -> int:
    """
    Integer partition count
    
    Args:
        n: Number to partition
        k: Maximum part size (None for unrestricted)
    
    Returns:
        Number of partitions
    """
    if n == 0:
        return 1
    if n < 0:
        return 0
    
    if k is None:
        k = n
    
    # DP: p[i][j] = partitions of i with max part j
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    
    # Base case: empty partition
    for j in range(k + 1):
        dp[0][j] = 1
    
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            # Either don't use j, or use at least one j
            dp[i][j] = dp[i][j-1]
            if i >= j:
                dp[i][j] += dp[i-j][j]
    
    return dp[n][k]


def derangement(n: int) -> int:
    """
    Derangement: D(n)
    Number of permutations with no fixed points
    
    Args:
        n: Number of elements
    
    Returns:
        Number of derangements
    """
    if n == 0:
        return 1
    if n == 1:
        return 0
    
    # D(n) = (n-1) * (D(n-1) + D(n-2))
    # Or: D(n) = n! * sum_{i=0}^{n} (-1)^i / i!
    
    d = [0] * (n + 1)
    d[0] = 1
    d[1] = 0
    
    for i in range(2, n + 1):
        d[i] = (i - 1) * (d[i-1] + d[i-2])
    
    return d[n]


# Function registry for executor
FUNCTIONS = {
    'stirling_first': stirling_first,
    'stirling_second': stirling_second,
    'bell_number': bell_number,
    'catalan_number': catalan_number,
    'partition_count': partition_count,
    'derangement': derangement
}


def execute(function_name: str, params: Dict[str, Any]) -> Any:
    """Execute advanced combinatorics function"""
    if function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}")
    
    func = FUNCTIONS[function_name]
    return func(**params)
