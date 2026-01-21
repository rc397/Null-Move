from __future__ import annotations

import chess

# Action encoding:
# action = promo_idx * 4096 + from_square * 64 + to_square
# promo_idx: 0 = no promotion, 1 = knight, 2 = bishop, 3 = rook, 4 = queen
ACTION_SIZE = 5 * 64 * 64  # 20480

_PROMO_TO_IDX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
_IDX_TO_PROMO = {
    0: None,
    1: chess.KNIGHT,
    2: chess.BISHOP,
    3: chess.ROOK,
    4: chess.QUEEN,
}


def move_to_action(move: chess.Move) -> int:
    promo_idx = _PROMO_TO_IDX.get(move.promotion, 0)
    return promo_idx * 4096 + (move.from_square << 6) + move.to_square


def action_to_move(action: int) -> chess.Move:
    promo_idx, rem = divmod(int(action), 4096)
    from_sq, to_sq = divmod(rem, 64)
    promo = _IDX_TO_PROMO.get(promo_idx, None)
    return chess.Move(from_sq, to_sq, promotion=promo)


def board_to_planes(board: chess.Board) -> list[float]:
    """Simple AlphaZero-ish input.

    12 piece planes (absolute), + 1 side-to-move plane.
    - Planes are flattened to a single list.
    - The side-to-move plane is all 1.0 if white to move else 0.0.

    This is intentionally minimal and 'original'. You can extend later with:
    castling rights, repetition count, move count, etc.
    """

    planes = [0.0] * (13 * 64)

    def set_plane(plane: int, piece_type: chess.PieceType, color: chess.Color) -> None:
        for sq in board.pieces(piece_type, color):
            planes[plane * 64 + sq] = 1.0

    # White
    set_plane(0, chess.PAWN, chess.WHITE)
    set_plane(1, chess.KNIGHT, chess.WHITE)
    set_plane(2, chess.BISHOP, chess.WHITE)
    set_plane(3, chess.ROOK, chess.WHITE)
    set_plane(4, chess.QUEEN, chess.WHITE)
    set_plane(5, chess.KING, chess.WHITE)

    # Black
    set_plane(6, chess.PAWN, chess.BLACK)
    set_plane(7, chess.KNIGHT, chess.BLACK)
    set_plane(8, chess.BISHOP, chess.BLACK)
    set_plane(9, chess.ROOK, chess.BLACK)
    set_plane(10, chess.QUEEN, chess.BLACK)
    set_plane(11, chess.KING, chess.BLACK)

    # Side to move plane
    stm_val = 1.0 if board.turn == chess.WHITE else 0.0
    base = 12 * 64
    for i in range(64):
        planes[base + i] = stm_val

    return planes
