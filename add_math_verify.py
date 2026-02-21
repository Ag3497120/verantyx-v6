"""
Add verify/worldgen specs to Math pieces
Agent F Task - Part 1
"""
import json
import shutil
from datetime import datetime

# Backup first
backup_path = 'pieces/piece_db_pre_agent_f.jsonl.bak'
shutil.copy('pieces/piece_db.jsonl', backup_path)
print(f'Backup created: {backup_path}')

# Load pieces
with open('pieces/piece_db.jsonl') as f:
    pieces = [json.loads(line) for line in f]

# Define verify/worldgen templates for missing pieces
TEMPLATES = {
    'arithmetic_equality': {
        'verify': {
            'kind': 'cross_check',
            'method': 'double_eval',
            'params': {
                'description': 'LHS と RHS を評価して boolean 一致確認',
                'type_check': 'boolean'
            }
        },
        'worldgen': {
            'domain': 'number',
            'params': {
                'lo': -20,
                'hi': 20
            },
            'constraints': ['integer']
        }
    },
    'combinatorics_combination': {
        'verify': {
            'kind': 'cross_check',
            'method': 'double_eval',
            'params': {
                'type_check': 'integer',
                'range': {'lo': 1, 'hi': 1e15},
                'constraints': ['0 <= r <= n']
            }
        },
        'worldgen': {
            'domain': 'number',
            'params': {
                'n': {'type': 'int', 'min': 1, 'max': 20},
                'r': {'type': 'int', 'min': 0, 'max': 'n'},
                'cross_check_fn': 'binomial(n, r)'
            }
        }
    },
    'algebra_solve_equation': {
        'verify': {
            'kind': 'substitution',
            'method': 'sympy_solve',
            'params': {
                'description': 'SymPy で方程式を解いて解を代入して確認',
                'library': 'sympy'
            }
        },
        'worldgen': {
            'domain': 'equation',
            'params': {
                'equation_type': 'linear',
                'max_terms': 3,
                'coeff_range': [-10, 10]
            }
        }
    },
    'nt_prime_compute': {
        'verify': {
            'kind': 'small_world',
            'method': 'miller_rabin',
            'params': {
                'description': '決定論的 Miller–Rabin (64bit 対応 witness セット)',
                'witnesses': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37],
                'schema_mapping': {'True': 'Yes', 'False': 'No'}
            }
        },
        'worldgen': {
            'domain': 'number',
            'params': {
                'lo': 2,
                'hi': 200,
                'description': '素数表から素数 / p*q 形式の合成数を生成'
            },
            'constraints': ['positive', 'integer']
        }
    },
    'algebra_solve_linear': {
        'verify': {
            'kind': 'substitution',
            'method': 'sympy_solve',
            'params': {
                'description': 'SymPy で一次方程式を解いて解を代入確認',
                'library': 'sympy'
            }
        },
        'worldgen': {
            'domain': 'equation',
            'params': {
                'equation_type': 'linear',
                'max_terms': 2,
                'coeff_range': [-10, 10]
            }
        }
    },
    'algebra_simplify': {
        'verify': {
            'kind': 'cross_check',
            'method': 'sympy_simplify',
            'params': {
                'description': 'SymPy で式を簡略化して等価性チェック',
                'library': 'sympy'
            }
        },
        'worldgen': {
            'domain': 'expression',
            'params': {
                'max_terms': 5,
                'coeff_range': [-5, 5]
            }
        }
    },
    'algebra_factor': {
        'verify': {
            'kind': 'cross_check',
            'method': 'sympy_factor',
            'params': {
                'description': 'SymPy で因数分解して展開して元の式と一致確認',
                'library': 'sympy'
            }
        },
        'worldgen': {
            'domain': 'polynomial',
            'params': {
                'max_deg': 3,
                'coeff_range': [-5, 5]
            }
        }
    },
    'linear_algebra_determinant': {
        'verify': {
            'kind': 'cross_check',
            'method': 'numpy_det',
            'params': {
                'description': 'NumPy で行列式を計算して一致確認',
                'library': 'numpy',
                'tolerance': 1e-9
            }
        },
        'worldgen': {
            'domain': 'matrix',
            'params': {
                'size': [2, 3],
                'entry_range': [-10, 10]
            }
        }
    },
    'linear_algebra_dot_product': {
        'verify': {
            'kind': 'cross_check',
            'method': 'numpy_dot',
            'params': {
                'description': 'NumPy で内積を計算して一致確認',
                'library': 'numpy',
                'tolerance': 1e-9
            }
        },
        'worldgen': {
            'domain': 'vector',
            'params': {
                'dim': [2, 3, 4],
                'entry_range': [-10, 10]
            }
        }
    },
    'linear_algebra_inverse': {
        'verify': {
            'kind': 'cross_check',
            'method': 'numpy_inv',
            'params': {
                'description': 'NumPy で逆行列を計算して A * A^-1 = I 確認',
                'library': 'numpy',
                'tolerance': 1e-9
            }
        },
        'worldgen': {
            'domain': 'matrix',
            'params': {
                'size': [2, 3],
                'entry_range': [-10, 10],
                'invertible': True
            }
        }
    },
    'nt_lcm_compute': {
        'verify': {
            'kind': 'cross_check',
            'method': 'gcd_lcm_relation',
            'params': {
                'description': 'LCM(a,b) * GCD(a,b) = a * b 関係を確認',
                'checks': ['product_relation', 'divisibility']
            }
        },
        'worldgen': {
            'domain': 'number',
            'params': {
                'lo': 1,
                'hi': 100
            },
            'constraints': ['positive', 'integer']
        }
    },
    'nt_divisor_count_compute': {
        'verify': {
            'kind': 'computation_log',
            'method': 'sympy_divisor_count',
            'params': {
                'description': 'SymPy の divisor_count で検証',
                'library': 'sympy'
            }
        },
        'worldgen': {
            'domain': 'number',
            'params': {
                'lo': 1,
                'hi': 100
            },
            'constraints': ['positive', 'integer']
        }
    },
    'nt_divisor_count_find': {
        'verify': {
            'kind': 'computation_log',
            'method': 'sympy_divisor_count',
            'params': {
                'description': 'SymPy の divisor_count で検証',
                'library': 'sympy'
            }
        },
        'worldgen': {
            'domain': 'number',
            'params': {
                'lo': 1,
                'hi': 100
            },
            'constraints': ['positive', 'integer']
        }
    },
    'comb_binomial': {
        'verify': {
            'kind': 'cross_check',
            'method': 'double_eval',
            'params': {
                'type_check': 'integer',
                'range': {'lo': 1, 'hi': 1e15},
                'constraints': ['0 <= k <= n']
            }
        },
        'worldgen': {
            'domain': 'number',
            'params': {
                'n': {'type': 'int', 'min': 1, 'max': 20},
                'k': {'type': 'int', 'min': 0, 'max': 'n'},
                'cross_check_fn': 'binomial(n, k)'
            }
        }
    },
}

# Update pieces
updated_count = 0
for piece in pieces:
    piece_id = piece.get('piece_id', '')

    if piece_id in TEMPLATES:
        if not piece.get('verify'):
            piece['verify'] = TEMPLATES[piece_id]['verify']
            updated_count += 1
            print(f'Added verify to: {piece_id}')

        if not piece.get('worldgen'):
            piece['worldgen'] = TEMPLATES[piece_id]['worldgen']
            print(f'Added worldgen to: {piece_id}')

print(f'\n✓ Updated {updated_count} pieces with verify/worldgen specs')

# Write back
with open('pieces/piece_db.jsonl', 'w') as f:
    for piece in pieces:
        f.write(json.dumps(piece, ensure_ascii=False) + '\n')

print(f'✓ Written to pieces/piece_db.jsonl')

# Summary
print('\n=== Summary ===')
math_patterns = ['factorial', 'gcd', 'prime', 'power', 'combination', 'permutation',
                 'algebra', 'arithmetic', 'calculus', 'linear_algebra']
math_pieces = [p for p in pieces if any(pat in p.get('piece_id', '').lower() for pat in math_patterns)]
with_verify = len([p for p in math_pieces if p.get('verify')])
with_worldgen = len([p for p in math_pieces if p.get('worldgen')])

print(f'Total Math pieces: {len(math_pieces)}')
print(f'Math pieces with verify: {with_verify}')
print(f'Math pieces with worldgen: {with_worldgen}')
