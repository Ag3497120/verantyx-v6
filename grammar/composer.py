"""
Composer - 構造化候補を最終答えに変換

Grammar Glueを使って構造体を文字列化
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import re


@dataclass
class GrammarPiece:
    """文法ピース"""
    grammar_id: str
    name: Optional[str]
    description: Optional[str]
    in_schema: str  # "integer", "move_sequence", etc.
    in_fields: List[str]
    optional_fields: List[str]
    template: str
    constraints: List[str]
    examples: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrammarPiece':
        """辞書から復元"""
        return cls(
            grammar_id=data["grammar_id"],
            name=data.get("name"),
            description=data.get("description"),
            in_schema=data["in"]["answer_schema"],
            in_fields=data["in"]["fields"],
            optional_fields=data["in"].get("optional_fields", []),
            template=data["template"],
            constraints=data.get("constraints", []),
            examples=data.get("examples", [])
        )
    
    def matches(self, schema: str, fields: Dict[str, Any]) -> bool:
        """スキーマとフィールドがマッチするかチェック"""
        if self.in_schema != schema:
            return False
        
        # 必須フィールドが全て存在するか
        for field in self.in_fields:
            if field not in fields:
                return False
        
        return True
    
    def apply(self, fields: Dict[str, Any]) -> str:
        """テンプレートにフィールドを適用"""
        output = self.template
        
        # プレースホルダーを置換
        for field, value in fields.items():
            placeholder = f"{{{field}}}"
            
            # 値を文字列化
            if isinstance(value, list):
                # リストの場合
                if field == "moves":
                    # チェスの手順
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = ", ".join(str(v) for v in value)
            elif isinstance(value, bool):
                value_str = "True" if value else "False"
            else:
                value_str = str(value)
            
            output = output.replace(placeholder, value_str)
        
        # 制約適用
        output = self._apply_constraints(output)
        
        return output
    
    def _apply_constraints(self, text: str) -> str:
        """制約を適用"""
        for constraint in self.constraints:
            if constraint == "uppercase":
                text = text.upper()
            elif constraint == "lowercase":
                text = text.lower()
            elif constraint == "single_letter":
                # 1文字のみ抽出
                match = re.search(r'[A-Za-z]', text)
                if match:
                    text = match.group(0).upper()
            elif constraint == "integer_format":
                # 整数フォーマット
                try:
                    text = str(int(float(text)))
                except:
                    pass
            elif constraint == "decimal_format":
                # 小数フォーマット
                try:
                    text = str(float(text))
                except:
                    pass
        
        return text


class GrammarDB:
    """文法ピースデータベース"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.pieces: List[GrammarPiece] = []
        self.load()
    
    def load(self):
        """JSONLからロード"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.pieces.append(GrammarPiece.from_dict(data))
        except FileNotFoundError:
            pass
    
    def find(self, schema: str, fields: Dict[str, Any]) -> Optional[GrammarPiece]:
        """スキーマとフィールドに合う文法ピースを検索"""
        for piece in self.pieces:
            if piece.matches(schema, fields):
                return piece
        return None


class AnswerComposer:
    """
    Answer Composer
    
    構造化候補を最終答え文字列に変換
    """
    
    def __init__(self, grammar_db: GrammarDB):
        self.grammar_db = grammar_db
    
    def compose(
        self,
        candidate: 'StructuredCandidate'
    ) -> Optional[str]:
        """
        構造化候補を最終答えに変換
        
        Args:
            candidate: 構造化候補
        
        Returns:
            最終答え文字列、変換失敗時None
        """
        # スキーマに合う文法ピースを検索
        grammar = self.grammar_db.find(candidate.schema, candidate.fields)
        
        if grammar is None:
            # 文法ピースが見つからない場合はフォールバック
            return self._fallback_compose(candidate)
        
        # テンプレート適用
        try:
            answer = grammar.apply(candidate.fields)
            return answer
        except Exception as e:
            print(f"[COMPOSER] Error applying grammar {grammar.grammar_id}: {e}")
            return self._fallback_compose(candidate)
    
    def _fallback_compose(self, candidate: 'StructuredCandidate') -> str:
        """
        フォールバック変換
        
        文法ピースが見つからない場合の簡易変換
        """
        # "value"フィールドがあればそれを返す
        if "value" in candidate.fields:
            value = candidate.fields["value"]
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return str(value)
        
        # その他のフィールドを連結
        parts = []
        for key, value in candidate.fields.items():
            if isinstance(value, list):
                parts.append(", ".join(str(v) for v in value))
            else:
                parts.append(str(value))
        
        return " ".join(parts) if parts else ""
    
    def validate(self, answer: str, schema: str) -> bool:
        """
        答えがスキーマを満たすか検証
        
        Args:
            answer: 答え文字列
            schema: 期待されるスキーマ
        
        Returns:
            検証結果
        """
        if not answer:
            return False
        
        # スキーマ別検証
        if schema == "integer":
            try:
                int(answer)
                return True
            except:
                return False
        
        elif schema == "decimal":
            try:
                float(answer)
                return True
            except:
                return False
        
        elif schema == "boolean":
            return answer.lower() in ["true", "false", "yes", "no"]
        
        elif schema == "option_label":
            return len(answer) == 1 and answer.upper() in "ABCDEFGHIJ"
        
        # その他のスキーマは文字列があればOK
        return len(answer) > 0
