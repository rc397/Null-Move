from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import chess

from .az_features import move_to_action
from .az_net import AZNet


@dataclass
class EdgeStats:
    prior: float
    n: int = 0
    w: float = 0.0

    @property
    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n


class Node:
    def __init__(self) -> None:
        self.expanded = False
        self.edges: dict[chess.Move, EdgeStats] = {}
        # Candidate moves (sorted by prior desc) used for top-k expansion / progressive widening.
        self._candidates: list[tuple[chess.Move, float]] = []
        self._cand_next: int = 0
        self.n = 0


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


@dataclass
class _Leaf:
    path: list[tuple[Node, chess.Move]]
    terminal: bool
    v: float
    fen: str | None = None
    node: Node | None = None


class MCTS:
    def __init__(self, net: AZNet) -> None:
        self.net = net
        # Use the full python-chess transposition tuple as the dict key (avoids hash collisions).
        self._tree: dict[object, Node] = {}
        # Per-run expansion controls (set by run())
        self._expand_topk: int = 0
        self._pw_alpha: float = 0.0
        self._pw_beta: float = 0.5

    def clear(self) -> None:
        self._tree.clear()

    def run(
        self,
        board: chess.Board,
        simulations: int,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        add_root_noise: bool = True,
        batch_size: int = 16,
        time_limit_ms: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        early_stop_min_sims: int = 0,
        early_stop_frac: float = 0.0,
        early_stop_lead: int = 0,
        expand_topk: int = 0,
        pw_alpha: float = 0.0,
        pw_beta: float = 0.5,
    ) -> dict[chess.Move, int]:
        """Run MCTS from this position and return visit counts per legal move."""

        # Configure expansion controls for this run.
        self._expand_topk = max(0, int(expand_topk))
        self._pw_alpha = max(0.0, float(pw_alpha))
        self._pw_beta = max(0.0, float(pw_beta))

        root_key = board._transposition_key()
        root = self._tree.get(root_key)
        if root is None:
            root = Node()
            self._tree[root_key] = root

        if not root.expanded:
            self._expand(board, root)

        # Ensure root has at least one selectable move (progressive widening may delay expansion).
        self._maybe_widen(root)

        if add_root_noise and root.edges:
            self._apply_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

        deadline = 0.0
        if time_limit_ms is not None:
            deadline = time.perf_counter() + max(0, time_limit_ms) / 1000.0

        sims_done = 0
        bs = max(1, int(batch_size))

        def _early_stop_ready() -> bool:
            if early_stop_min_sims <= 0 or early_stop_frac <= 0.0:
                return False
            if root is None or (not root.edges):
                return False
            total = sum(st.n for st in root.edges.values())
            if total < int(early_stop_min_sims):
                return False
            if len(root.edges) < 2:
                return False
            counts = sorted((st.n for st in root.edges.values()), reverse=True)
            best = int(counts[0])
            second = int(counts[1])
            if total <= 0:
                return False
            frac = float(best) / float(total)
            if frac < float(early_stop_frac):
                return False
            if int(early_stop_lead) > 0 and (best - second) < int(early_stop_lead):
                return False
            return True

        while sims_done < simulations:
            if should_stop is not None and should_stop():
                break
            if deadline and time.perf_counter() >= deadline:
                break
            # Avoid expensive claim-draw logic (e.g., threefold repetition checks) in the inner loop.
            if board.is_game_over():
                break

            leaves: list[_Leaf] = []
            want = min(bs, simulations - sims_done)
            for _ in range(want):
                if should_stop is not None and should_stop():
                    break
                if deadline and time.perf_counter() >= deadline:
                    break
                leaf = self._collect_leaf(board, c_puct=c_puct)
                sims_done += 1
                if leaf.terminal:
                    self._backprop(leaf.path, leaf.v)
                else:
                    leaves.append(leaf)

            if not leaves:
                continue

            valid: list[tuple[_Leaf, chess.Board]] = []
            for leaf in leaves:
                if leaf.node is None or leaf.fen is None:
                    continue
                fen = leaf.fen
                # Defensive sanity checks: a valid full FEN is short and has 6 fields (5 spaces).
                # If something upstream corrupts the string, parsing can get very slow.
                if (not isinstance(fen, str)) or (len(fen) > 200) or (fen.count(" ") < 5):
                    self._backprop(leaf.path, 0.0)
                    continue
                try:
                    b = chess.Board(fen)
                except Exception:
                    # Defensive: don't crash the whole run on a malformed FEN.
                    self._backprop(leaf.path, 0.0)
                    continue
                valid.append((leaf, b))

            if not valid:
                continue

            logits_b, v_b = self.net.infer_many([b for _, b in valid])
            for (leaf, b), logits_row, v_row in zip(valid, logits_b, v_b):
                v = float(v_row.item())
                self._expand_with_logits(b, leaf.node, logits_row, v)
                self._backprop(leaf.path, v)

            if _early_stop_ready():
                break

        return {m: st.n for m, st in root.edges.items()}

    def _collect_leaf(self, board: chess.Board, c_puct: float) -> _Leaf:
        path: list[tuple[Node, chess.Move]] = []
        pushed: list[chess.Move] = []

        while True:
            key = board._transposition_key()
            node = self._tree.get(key)
            if node is None:
                node = Node()
                self._tree[key] = node

            if board.is_game_over():
                v = self._terminal_value(board)
                for _ in range(len(pushed)):
                    board.pop()
                return _Leaf(path=path, terminal=True, v=v)

            if not node.expanded:
                fen = board.fen()
                for _ in range(len(pushed)):
                    board.pop()
                return _Leaf(path=path, terminal=False, v=0.0, fen=fen, node=node)

            # Progressive widening: add more moves as visits grow.
            self._maybe_widen(node)

            # Safety: if we still don't have any edges, force-add one candidate.
            if not node.edges:
                self._maybe_widen(node, force_one=True)
                if not node.edges:
                    # No legal moves (should have been terminal), treat as leaf.
                    v = self._terminal_value(board)
                    for _ in range(len(pushed)):
                        board.pop()
                    return _Leaf(path=path, terminal=True, v=v)

            move = self._select(node, c_puct)
            path.append((node, move))
            board.push(move)
            pushed.append(move)

    def _terminal_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        return 0.0

    def _expand(self, board: chess.Board, node: Node) -> float:
        logits, v = self.net.infer(board)
        return self._expand_with_logits(board, node, logits, v)

    def _expand_with_logits(self, board: chess.Board, node: Node, logits, v: float) -> float:
        legal = list(board.legal_moves)
        if not legal:
            node.expanded = True
            node.edges = {}
            node._candidates = []
            node._cand_next = 0
            return v

        scores: list[float] = []
        for m in legal:
            a = move_to_action(m)
            scores.append(float(logits[a].item()))
        priors = _softmax(scores)

        # Sort moves by prior desc.
        cand = list(zip(legal, priors))
        cand.sort(key=lambda t: t[1], reverse=True)

        # Hard cap (top-k) if requested.
        topk = int(self._expand_topk)
        if topk > 0 and len(cand) > topk:
            cand = cand[:topk]

        # Renormalize priors on kept candidates.
        s = float(sum(p for _m, p in cand))
        if s > 0.0:
            cand = [(m, float(p) / s) for m, p in cand]

        node._candidates = cand
        node._cand_next = 0
        node.edges = {}
        node.expanded = True

        # Add initial moves (at least one).
        self._maybe_widen(node, force_one=True)
        return v

    def _select(self, node: Node, c_puct: float) -> chess.Move:
        # Expand more moves if warranted by progressive widening.
        self._maybe_widen(node)
        best_m = None
        best_s = -1e9
        sqrt_n = math.sqrt(max(1, node.n))
        for m, st in node.edges.items():
            u = c_puct * st.prior * (sqrt_n / (1 + st.n))
            s = st.q + u
            if s > best_s:
                best_s = s
                best_m = m
        assert best_m is not None
        return best_m

    def _maybe_widen(self, node: Node, force_one: bool = False) -> None:
        """Progressively add candidate moves into node.edges.

        If pw_alpha<=0, this behaves like 'expand everything we kept' (top-k or full).
        """

        if not node.expanded:
            return

        total_cand = len(node._candidates)
        if total_cand <= 0:
            return

        # Determine how many moves should be active.
        if force_one:
            desired = max(1, len(node.edges))
        else:
            if self._pw_alpha > 0.0:
                desired = int(self._pw_alpha * ((float(node.n) + 1.0) ** float(self._pw_beta)))
                desired = max(1, desired)
            else:
                desired = total_cand

        if desired > total_cand:
            desired = total_cand

        # Add moves until we reach desired.
        while len(node.edges) < desired and node._cand_next < total_cand:
            m, p = node._candidates[node._cand_next]
            node._cand_next += 1
            # Avoid duplicates (shouldn't happen, but defensive).
            if m in node.edges:
                continue
            node.edges[m] = EdgeStats(prior=float(p))

    def _apply_dirichlet_noise(self, root: Node, alpha: float, eps: float) -> None:
        moves = list(root.edges.keys())
        if not moves:
            return
        gammas = [random.gammavariate(alpha, 1.0) for _ in moves]
        s = sum(gammas)
        noise = [g / s for g in gammas]
        for m, n in zip(moves, noise):
            st = root.edges[m]
            st.prior = (1.0 - eps) * st.prior + eps * n

    def _backprop(self, path: list[tuple[Node, chess.Move]], leaf_v: float) -> None:
        v = leaf_v
        for node, move in reversed(path):
            node.n += 1
            st = node.edges[move]
            st.n += 1
            st.w += v
            v = -v
