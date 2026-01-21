from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import chess

from .eval import evaluate

MATE_SCORE = 100_000


@dataclass
class SearchResult:
    best_move: chess.Move | None
    score_cp: int
    depth: int
    nodes: int
    time_ms: int
    pv: list[chess.Move]


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: int  # 0 exact, 1 lowerbound, 2 upperbound
    best: chess.Move | None


EXACT, LOWER, UPPER = 0, 1, 2


class Searcher:
    def __init__(self) -> None:
        self.tt: dict[int, TTEntry] = {}
        self.nodes = 0
        self.stop = False
        self.start_time = 0.0
        self.deadline = 0.0

        self.evaluator: Callable[[chess.Board], int] = evaluate

        # Move ordering heuristics
        self._history: list[int] = [0] * (64 * 64)  # from*64 + to
        self._killer1: list[chess.Move | None] = [None] * 128
        self._killer2: list[chess.Move | None] = [None] * 128

        # Pruning/tuning
        self.enable_null_move_pruning = True
        self.null_move_reduction = 2
        self.enable_lmr = True
        self.lmr_full_depth_moves = 4
        self.lmr_reduction = 1

    def reset(self) -> None:
        self.tt.clear()
        self._history = [0] * (64 * 64)
        self._killer1 = [None] * 128
        self._killer2 = [None] * 128

    def set_evaluator(self, evaluator: Callable[[chess.Board], int]) -> None:
        self.evaluator = evaluator

    def _time_up(self) -> bool:
        if self.deadline <= 0:
            return False
        return time.perf_counter() >= self.deadline

    def _move_index(self, m: chess.Move) -> int:
        return (m.from_square << 6) | m.to_square

    def _order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_best: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        if not moves:
            return moves

        def score(m: chess.Move) -> int:
            s = 0
            if tt_best is not None and m == tt_best:
                s += 10_000_000

            # Captures/promotions first
            if board.is_capture(m):
                # MVV-LVA-ish using piece values
                victim = board.piece_type_at(m.to_square)
                attacker = board.piece_type_at(m.from_square)
                if victim is not None:
                    s += 1000 * victim
                if attacker is not None:
                    s -= attacker
                s += 1_000_000
            if m.promotion is not None:
                s += 900_000

            # Killer moves (quiet moves that caused beta cutoffs)
            k1 = self._killer1[ply] if ply < len(self._killer1) else None
            k2 = self._killer2[ply] if ply < len(self._killer2) else None
            if k1 is not None and m == k1:
                s += 800_000
            elif k2 is not None and m == k2:
                s += 700_000

            # History heuristic (quiet move success over time)
            if not board.is_capture(m) and m.promotion is None:
                s += self._history[self._move_index(m)]
            return s

        return sorted(moves, key=score, reverse=True)

    def search(
        self,
        board: chess.Board,
        max_depth: int,
        time_limit_ms: int | None,
        info_cb: Callable[[int, int, int, int, list[chess.Move]], None] | None = None,
    ) -> SearchResult:
        self.nodes = 0
        self.stop = False
        self.start_time = time.perf_counter()
        self.deadline = 0.0 if time_limit_ms is None else (self.start_time + time_limit_ms / 1000.0)

        best_move: chess.Move | None = None
        best_score = -MATE_SCORE
        best_pv: list[chess.Move] = []

        for depth in range(1, max_depth + 1):
            # Aspiration window around previous score
            if depth == 1:
                alpha, beta = -MATE_SCORE, MATE_SCORE
            else:
                window = 50
                alpha = max(-MATE_SCORE, best_score - window)
                beta = min(MATE_SCORE, best_score + window)

            while True:
                score, pv = self._negamax(board, depth, alpha, beta, ply=0)
                if self.stop:
                    break

                # fail-low/high => widen and retry
                if score <= alpha:
                    alpha = max(-MATE_SCORE, alpha - 200)
                    continue
                if score >= beta:
                    beta = min(MATE_SCORE, beta + 200)
                    continue
                break

            if self.stop:
                break
            best_score = score
            best_pv = pv
            best_move = pv[0] if pv else None

            if info_cb is not None:
                now_ms = int((time.perf_counter() - self.start_time) * 1000)
                info_cb(depth, best_score, self.nodes, now_ms, best_pv)

            if abs(best_score) >= MATE_SCORE - 1000:
                break

        time_ms = int((time.perf_counter() - self.start_time) * 1000)
        return SearchResult(best_move=best_move, score_cp=best_score, depth=depth if best_move else 0, nodes=self.nodes, time_ms=time_ms, pv=best_pv)

    def _is_null_move_safe(self, board: chess.Board) -> bool:
        if board.is_check():
            return False
        # Avoid likely zugzwang endgames: allow null move only if some non-pawn material exists.
        pieces = board.piece_map().values()
        return any(p.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN) for p in pieces)

    def _record_cutoff(self, move: chess.Move, ply: int, depth: int, board: chess.Board) -> None:
        if board.is_capture(move) or move.promotion is not None:
            return

        if ply < len(self._killer1):
            k1 = self._killer1[ply]
            if k1 is None or move != k1:
                self._killer2[ply] = k1
                self._killer1[ply] = move

        self._history[self._move_index(move)] += depth * depth

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> tuple[int, list[chess.Move]]:
        if self.stop or self._time_up():
            self.stop = True
            return 0, []

        self.nodes += 1

        if depth <= 0:
            return self._quiescence(board, alpha, beta)

        if board.is_game_over(claim_draw=True):
            return self.evaluator(board), []

        # python-chess 1.11.x exposes this as a private API.
        # We hash it to keep the TT key compact.
        key = hash(board._transposition_key())
        entry = self.tt.get(key)
        tt_best = entry.best if entry else None
        if entry and entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.score, [entry.best] if entry.best else []
            if entry.flag == LOWER:
                alpha = max(alpha, entry.score)
            elif entry.flag == UPPER:
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score, [entry.best] if entry.best else []

        # Null-move pruning (very effective in the middlegame)
        if (
            self.enable_null_move_pruning
            and depth >= 3
            and ply > 0
            and alpha < beta - 1
            and self._is_null_move_safe(board)
        ):
            board.push(chess.Move.null())
            null_depth = depth - 1 - self.null_move_reduction
            score, _ = self._negamax(board, null_depth, -beta, -beta + 1, ply + 1)
            board.pop()
            if not self.stop:
                score = -score
                if score >= beta:
                    return score, []

        best_move: chess.Move | None = None
        best_line: list[chess.Move] = []
        original_alpha = alpha

        moves = list(board.legal_moves)
        moves = self._order_moves(board, moves, tt_best, ply)

        if not moves:
            return self.evaluator(board), []

        value = -MATE_SCORE
        for move_index, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_promo = move.promotion is not None

            board.push(move)

            # Principal Variation Search (PVS) + Late Move Reductions (LMR)
            reduced = False
            next_depth = depth - 1
            if (
                self.enable_lmr
                and depth >= 3
                and move_index >= self.lmr_full_depth_moves
                and not is_capture
                and not is_promo
                and not board.is_check()
            ):
                next_depth = max(0, next_depth - self.lmr_reduction)
                reduced = True

            if move_index == 0:
                child_score, child_pv = self._negamax(board, next_depth, -beta, -alpha, ply + 1)
            else:
                # null-window search first
                child_score, child_pv = self._negamax(board, next_depth, -alpha - 1, -alpha, ply + 1)
                if (not self.stop) and (-child_score > alpha) and (-child_score < beta):
                    # re-search with full window
                    child_score, child_pv = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                elif reduced and (not self.stop) and (-child_score > alpha):
                    # if reduced line improved alpha, confirm at full depth
                    child_score, child_pv = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)

            board.pop()
            if self.stop:
                break

            score = -child_score
            if score > value:
                value = score
                best_move = move
                best_line = [move] + child_pv

            alpha = max(alpha, value)
            if alpha >= beta:
                self._record_cutoff(move, ply, depth, board)
                break

        flag = EXACT
        if value <= original_alpha:
            flag = UPPER
        elif value >= beta:
            flag = LOWER

        self.tt[key] = TTEntry(depth=depth, score=value, flag=flag, best=best_move)
        return value, best_line

    def _quiescence(self, board: chess.Board, alpha: int, beta: int) -> tuple[int, list[chess.Move]]:
        if self.stop or self._time_up():
            self.stop = True
            return 0, []

        self.nodes += 1

        stand_pat = self.evaluator(board)
        if stand_pat >= beta:
            return beta, []
        if stand_pat > alpha:
            alpha = stand_pat

        # Only consider captures (and promotions)
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion is not None]
        moves = self._order_moves(board, moves, None, ply=0)

        best_line: list[chess.Move] = []
        for move in moves:
            board.push(move)
            score, pv = self._quiescence(board, -beta, -alpha)
            board.pop()
            if self.stop:
                break

            score = -score
            if score >= beta:
                return beta, [move] + pv
            if score > alpha:
                alpha = score
                best_line = [move] + pv

        return alpha, best_line
