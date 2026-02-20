"""
Chess executor wrapper - bridges piece_db executor reference to chess_stockfish module.
"""
from executors.chess_stockfish import chess_stockfish


def stockfish_analyze(question: str = '', fen: str = '', depth: int = 20, **kwargs) -> dict:
    """
    Stockfish chess analyzer.
    
    Args:
        question: Full question text (used to extract FEN/PGN if fen not given)
        fen: Optional FEN position string
        depth: Stockfish search depth
    
    Returns:
        dict with 'value', 'confidence', 'schema'
    """
    # If question text is available, use it for context
    text = question or fen or kwargs.get('source_text', '')
    
    answer = chess_stockfish(text)
    
    if answer is not None:
        return {
            'value': answer,
            'confidence': 0.65,
            'schema': 'move_sequence',
        }
    return {
        'value': None,
        'confidence': 0.0,
        'schema': 'move_sequence',
    }
