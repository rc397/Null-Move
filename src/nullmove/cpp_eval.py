from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

import chess


class _EvalDLL:
    def __init__(self, dll: ctypes.CDLL):
        self.dll = dll
        self.fn = dll.nm_eval
        self.fn.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        self.fn.restype = ctypes.c_int

    def evaluate(self, board: chess.Board) -> int:
        pm = board.pieces_mask
        wp = int(pm(chess.PAWN, chess.WHITE))
        wn = int(pm(chess.KNIGHT, chess.WHITE))
        wb = int(pm(chess.BISHOP, chess.WHITE))
        wr = int(pm(chess.ROOK, chess.WHITE))
        wq = int(pm(chess.QUEEN, chess.WHITE))
        wk = int(pm(chess.KING, chess.WHITE))

        bp = int(pm(chess.PAWN, chess.BLACK))
        bn = int(pm(chess.KNIGHT, chess.BLACK))
        bb = int(pm(chess.BISHOP, chess.BLACK))
        br = int(pm(chess.ROOK, chess.BLACK))
        bq = int(pm(chess.QUEEN, chess.BLACK))
        bk = int(pm(chess.KING, chess.BLACK))

        turn_white = 1 if board.turn == chess.WHITE else 0

        return int(
            self.fn(wp, wn, wb, wr, wq, wk, bp, bn, bb, br, bq, bk, turn_white)
        )


_DLL: Optional[_EvalDLL] = None


def try_load_default() -> Optional[_EvalDLL]:
    global _DLL
    if _DLL is not None:
        return _DLL

    env = os.environ.get("NULLMOVE_EVAL_DLL")
    if env:
        dll_path = Path(env)
    else:
        # repo-root/bin/nullmove_eval.dll
        dll_path = Path(__file__).resolve().parents[2] / "bin" / "nullmove_eval.dll"

    if not dll_path.exists():
        return None

    try:
        _DLL = _EvalDLL(ctypes.CDLL(str(dll_path)))
        return _DLL
    except OSError:
        return None
