"""
Linear Algebra Executor - 線形代数計算
"""
import re
from typing import Any, Dict, List, Optional
import numpy as np


# =======================================
# Helper functions
# =======================================

def _parse_matrix(text: str) -> Optional[List[List[float]]]:
    """
    テキストから行列を抽出
    
    Patterns:
        - [[1, 2], [3, 4]]
        - [[1,2],[3,4]]
        - [1 2; 3 4]
        - identity matrix (3x3) → [[1,0,0],[0,1,0],[0,0,1]]
    """
    text_lower = text.lower()
    
    # Identity matrix
    if "identity" in text_lower:
        # "3x3 identity" または "identity matrix" を検出
        match = re.search(r'(\d+)\s*x\s*\d+', text_lower)
        if match:
            n = int(match.group(1))
            return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        # デフォルト 2x2
        return [[1, 0], [0, 1]]
    
    # [[a, b], [c, d]] 形式
    if "[[" in text:
        try:
            # Python list literalとして評価
            matrix_str = text[text.index("[["):text.rindex("]]")+2]
            matrix = eval(matrix_str)
            return matrix
        except:
            pass
    
    # [a b; c d] 形式（MATLAB風）
    if ";" in text:
        try:
            inside = text[text.index("["):text.rindex("]")+1]
            rows = inside.strip("[]").split(";")
            matrix = []
            for row in rows:
                nums = [float(x.strip()) for x in row.split() if x.strip()]
                matrix.append(nums)
            return matrix
        except:
            pass
    
    return None


def _parse_vector(text: str) -> Optional[List[float]]:
    """
    テキストからベクトルを抽出
    
    Patterns:
        - [1, 2, 3]
        - [1,2,3]
        - (1, 2, 3)
    """
    # [a, b, c] 形式
    if "[" in text:
        try:
            vector_str = text[text.index("["):text.rindex("]")+1]
            vector = eval(vector_str)
            if isinstance(vector, list):
                return [float(x) for x in vector]
        except:
            pass
    
    # (a, b, c) 形式
    if "(" in text:
        try:
            vector_str = text[text.index("("):text.rindex(")")+1]
            vector = eval(vector_str)
            if isinstance(vector, tuple):
                return [float(x) for x in vector]
        except:
            pass
    
    return None


# =======================================
# Executor functions (for piece calls)
# =======================================

def matrix_determinant(**kwargs) -> Dict[str, Any]:
    """
    行列式を計算
    
    Params:
        ir: Dict (IRオブジェクト、metadata.source_textを含む)
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    matrix = kwargs.get("matrix")
    
    if matrix is None and source_text:
        matrix = _parse_matrix(source_text)
    
    if matrix is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        mat = np.array(matrix, dtype=float)
        det = np.linalg.det(mat)
        
        # 整数に近い場合は整数化
        if abs(det - round(det)) < 1e-9:
            det = int(round(det))
        
        return {
            "success": True,
            "value": det,
            "confidence": 1.0,
            "schema": "decimal"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }


def dot_product(**kwargs) -> Dict[str, Any]:
    """
    ベクトルの内積を計算
    
    Params:
        ir: Dict (IRオブジェクト、source_textを含む)
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    v1 = kwargs.get("vector1")
    v2 = kwargs.get("vector2")
    
    if v1 is None or v2 is None:
        # source_textから抽出
        if source_text:
            # 複数のベクトルを検出
            vectors = []
            for match in re.finditer(r'\[([^\]]+)\]', source_text):
                try:
                    vec_str = "[" + match.group(1) + "]"
                    vec = eval(vec_str)
                    if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                        vectors.append([float(x) for x in vec])
                except:
                    pass
            
            if len(vectors) >= 2:
                v1, v2 = vectors[0], vectors[1]
    
    if v1 is None or v2 is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    if len(v1) != len(v2):
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }
    
    try:
        dot = sum(a * b for a, b in zip(v1, v2))
        
        # 整数に近い場合は整数化
        if abs(dot - round(dot)) < 1e-9:
            dot = int(round(dot))
        
        return {
            "success": True,
            "value": dot,
            "confidence": 1.0,
            "schema": "decimal"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "decimal"
        }


def matrix_inverse(**kwargs) -> Dict[str, Any]:
    """
    逆行列を計算
    
    Params:
        ir: Dict (IRオブジェクト、source_textを含む)
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    matrix = kwargs.get("matrix")
    
    if matrix is None and source_text:
        matrix = _parse_matrix(source_text)
    
    if matrix is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "matrix"
        }
    
    try:
        mat = np.array(matrix, dtype=float)
        inv = np.linalg.inv(mat)
        
        # リストに変換
        inv_list = inv.tolist()
        
        return {
            "success": True,
            "value": inv_list,
            "confidence": 1.0,
            "schema": "matrix"
        }
    except np.linalg.LinAlgError:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "matrix"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "matrix"
        }


def eigenvalues(**kwargs) -> Dict[str, Any]:
    """
    固有値を計算
    
    Params:
        ir: Dict (IRオブジェクト、source_textを含む)
    """
    ir = kwargs.get("ir", {})
    source_text = ir.get("metadata", {}).get("source_text", "")
    matrix = kwargs.get("matrix")
    
    if matrix is None and source_text:
        matrix = _parse_matrix(source_text)
    
    if matrix is None:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "list"
        }
    
    try:
        mat = np.array(matrix, dtype=float)
        eigenvals = np.linalg.eigvals(mat)
        
        # 実数部分のみ（虚数部分が小さい場合）
        result = []
        for ev in eigenvals:
            if np.iscomplex(ev):
                if abs(ev.imag) < 1e-9:
                    result.append(float(ev.real))
                else:
                    result.append(complex(ev))
            else:
                result.append(float(ev))
        
        return {
            "success": True,
            "value": result,
            "confidence": 1.0,
            "schema": "list"
        }
    except Exception as e:
        return {
            "success": False,
            "value": None,
            "confidence": 0.0,
            "schema": "list"
        }


# =======================================
# Legacy class-based implementation (unused)
# =======================================

class LinearAlgebraExecutor:
    """線形代数計算の実行"""
    
    def execute(self, func_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        線形代数関数を実行
        
        Args:
            func_name: 関数名 (matrix_determinant, dot_product, matrix_inverse, eigenvalues)
            params: パラメータ辞書
            
        Returns:
            {
                "success": bool,
                "result": Any,
                "confidence": float,
                "method": str
            }
        """
        try:
            if func_name == "matrix_determinant":
                return self._matrix_determinant(params)
            elif func_name == "dot_product":
                return self._dot_product(params)
            elif func_name == "matrix_inverse":
                return self._matrix_inverse(params)
            elif func_name == "eigenvalues":
                return self._eigenvalues(params)
            else:
                return {
                    "success": False,
                    "result": None,
                    "confidence": 0.0,
                    "method": f"unknown_function: {func_name}"
                }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"error: {str(e)}"
            }
    
    def _parse_matrix(self, text: str) -> Optional[List[List[float]]]:
        """
        テキストから行列を抽出
        
        Patterns:
            - [[1, 2], [3, 4]]
            - [[1,2],[3,4]]
            - [1 2; 3 4]
            - identity matrix (3x3) → [[1,0,0],[0,1,0],[0,0,1]]
        """
        text_lower = text.lower()
        
        # Identity matrix
        if "identity" in text_lower:
            # "3x3 identity" または "identity matrix" を検出
            match = re.search(r'(\d+)\s*x\s*\d+', text_lower)
            if match:
                n = int(match.group(1))
                return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            # デフォルト 2x2
            return [[1, 0], [0, 1]]
        
        # [[a, b], [c, d]] 形式
        if "[[" in text:
            try:
                # Python list literalとして評価
                matrix_str = text[text.index("[["):text.rindex("]]")+2]
                matrix = eval(matrix_str)
                return matrix
            except:
                pass
        
        # [a b; c d] 形式（MATLAB風）
        if ";" in text:
            try:
                inside = text[text.index("["):text.rindex("]")+1]
                rows = inside.strip("[]").split(";")
                matrix = []
                for row in rows:
                    nums = [float(x.strip()) for x in row.split() if x.strip()]
                    matrix.append(nums)
                return matrix
            except:
                pass
        
        return None
    
    def _parse_vector(self, text: str) -> Optional[List[float]]:
        """
        テキストからベクトルを抽出
        
        Patterns:
            - [1, 2, 3]
            - [1,2,3]
            - (1, 2, 3)
        """
        # [a, b, c] 形式
        if "[" in text:
            try:
                vector_str = text[text.index("["):text.rindex("]")+1]
                vector = eval(vector_str)
                if isinstance(vector, list):
                    return [float(x) for x in vector]
            except:
                pass
        
        # (a, b, c) 形式
        if "(" in text:
            try:
                vector_str = text[text.index("("):text.rindex(")")+1]
                vector = eval(vector_str)
                if isinstance(vector, tuple):
                    return [float(x) for x in vector]
            except:
                pass
        
        return None
    
    def _matrix_determinant(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        行列式を計算
        
        Params:
            matrix: List[List[float]] または
            source_text: str (行列を含むテキスト)
        """
        matrix = None
        
        if "matrix" in params:
            matrix = params["matrix"]
        elif "source_text" in params:
            matrix = self._parse_matrix(params["source_text"])
        
        if matrix is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_matrix"
            }
        
        try:
            mat = np.array(matrix, dtype=float)
            det = np.linalg.det(mat)
            
            # 整数に近い場合は整数化
            if abs(det - round(det)) < 1e-9:
                det = int(round(det))
            
            return {
                "success": True,
                "result": det,
                "confidence": 1.0,
                "method": "numpy_det"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"computation_error: {str(e)}"
            }
    
    def _dot_product(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ベクトルの内積を計算
        
        Params:
            vector1, vector2: List[float] または
            source_text: str (2つのベクトルを含むテキスト)
        """
        v1 = params.get("vector1")
        v2 = params.get("vector2")
        
        if v1 is None or v2 is None:
            # source_textから抽出
            if "source_text" in params:
                text = params["source_text"]
                # 複数のベクトルを検出
                vectors = []
                for match in re.finditer(r'\[([^\]]+)\]', text):
                    try:
                        vec_str = "[" + match.group(1) + "]"
                        vec = eval(vec_str)
                        if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                            vectors.append([float(x) for x in vec])
                    except:
                        pass
                
                if len(vectors) >= 2:
                    v1, v2 = vectors[0], vectors[1]
        
        if v1 is None or v2 is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_vectors"
            }
        
        if len(v1) != len(v2):
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "dimension_mismatch"
            }
        
        try:
            dot = sum(a * b for a, b in zip(v1, v2))
            
            # 整数に近い場合は整数化
            if abs(dot - round(dot)) < 1e-9:
                dot = int(round(dot))
            
            return {
                "success": True,
                "result": dot,
                "confidence": 1.0,
                "method": "dot_product"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"computation_error: {str(e)}"
            }
    
    def _matrix_inverse(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        逆行列を計算
        
        Params:
            matrix: List[List[float]] または
            source_text: str
        """
        matrix = None
        
        if "matrix" in params:
            matrix = params["matrix"]
        elif "source_text" in params:
            matrix = self._parse_matrix(params["source_text"])
        
        if matrix is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_matrix"
            }
        
        try:
            mat = np.array(matrix, dtype=float)
            inv = np.linalg.inv(mat)
            
            # リストに変換
            inv_list = inv.tolist()
            
            return {
                "success": True,
                "result": inv_list,
                "confidence": 1.0,
                "method": "numpy_inv"
            }
        except np.linalg.LinAlgError:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "singular_matrix"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"computation_error: {str(e)}"
            }
    
    def _eigenvalues(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        固有値を計算
        
        Params:
            matrix: List[List[float]] または
            source_text: str
        """
        matrix = None
        
        if "matrix" in params:
            matrix = params["matrix"]
        elif "source_text" in params:
            matrix = self._parse_matrix(params["source_text"])
        
        if matrix is None:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": "no_matrix"
            }
        
        try:
            mat = np.array(matrix, dtype=float)
            eigenvals = np.linalg.eigvals(mat)
            
            # 実数部分のみ（虚数部分が小さい場合）
            result = []
            for ev in eigenvals:
                if np.iscomplex(ev):
                    if abs(ev.imag) < 1e-9:
                        result.append(float(ev.real))
                    else:
                        result.append(complex(ev))
                else:
                    result.append(float(ev))
            
            return {
                "success": True,
                "result": result,
                "confidence": 1.0,
                "method": "numpy_eig"
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "confidence": 0.0,
                "method": f"computation_error: {str(e)}"
            }
