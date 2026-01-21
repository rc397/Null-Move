from __future__ import annotations

import time
import chess

from .search import Searcher, SearchResult
from .types import SearchLimits
from .nn_eval import NNConfig, maybe_load_nn


class Engine:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.searcher = Searcher()

        self._stop = False

        self.nn = None
        self.nn_config = NNConfig(enabled=False, weights_path="nn.pt")

        # AlphaZero-style (MCTS + policy/value net)
        self.az_enabled = False
        self.az_weights_path = "az.pt"
        self.az_sims = 200
        self.az_batch_size = 16
        self.az_c_puct = 1.5
        self._az_net = None
        self._az_net_loaded_from: str | None = None

    def new_game(self) -> None:
        self.board.reset()
        self.searcher.reset()
        self._az_net = None
        self._az_net_loaded_from = None

    def set_option(self, name: str, value: str) -> None:
        n = name.strip().lower()
        v = value.strip()

        if n == "usenn":
            self.nn_config = NNConfig(enabled=(v.lower() in ("true", "1", "yes", "on")), weights_path=self.nn_config.weights_path)
            self.nn = None
        elif n == "nnweights":
            self.nn_config = NNConfig(enabled=self.nn_config.enabled, weights_path=v)
            self.nn = None
        elif n == "useaz":
            self.az_enabled = v.lower() in ("true", "1", "yes", "on")
        elif n == "azweights":
            self.az_weights_path = v
            self._az_net = None
            self._az_net_loaded_from = None
        elif n == "azsims":
            try:
                self.az_sims = max(1, int(v))
            except ValueError:
                pass
        elif n == "azbatchsize":
            try:
                self.az_batch_size = max(1, int(v))
            except ValueError:
                pass
        elif n == "azcpuct":
            try:
                self.az_c_puct = float(v)
            except ValueError:
                pass

    def set_position(self, fen: str | None, moves_uci: list[str]) -> None:
        if fen is None:
            self.board.reset()
        else:
            self.board.set_fen(fen)

        for m in moves_uci:
            self.board.push_uci(m)

    def go(self, limits: SearchLimits, info_cb=None) -> SearchResult:
        self._stop = False
        self.searcher.stop = False

        # Lazy-load NN on demand (optional)
        if self.nn is None and self.nn_config.enabled:
            self.nn = maybe_load_nn(self.nn_config)
            if self.nn is not None:
                self.searcher.set_evaluator(self.nn.evaluate)
            else:
                from .eval import evaluate as default_eval

                self.searcher.set_evaluator(default_eval)
        elif not self.nn_config.enabled:
            # reset to default evaluator
            from .eval import evaluate as default_eval

            self.searcher.set_evaluator(default_eval)

        time_limit_ms: int | None = None
        if limits.infinite:
            time_limit_ms = None
        elif limits.movetime_ms is not None:
            time_limit_ms = max(1, limits.movetime_ms)
        else:
            # Simple time management: spend ~1/30 of remaining time + increment.
            if self.board.turn == chess.WHITE:
                t = limits.wtime_ms
                inc = limits.winc_ms
            else:
                t = limits.btime_ms
                inc = limits.binc_ms
            if t is not None:
                budget = int(t / 30) + int(0.8 * inc)
                time_limit_ms = max(5, min(budget, t - 5))

        if self.az_enabled:
            return self._go_az(time_limit_ms=time_limit_ms, info_cb=info_cb)

        max_depth = limits.depth or 6
        return self.searcher.search(self.board, max_depth=max_depth, time_limit_ms=time_limit_ms, info_cb=info_cb)

    def _go_az(self, time_limit_ms: int | None, info_cb=None) -> SearchResult:
        from .az_mcts import MCTS
        from .az_net import AZNet

        if self._az_net is None or self._az_net_loaded_from != self.az_weights_path:
            self._az_net = AZNet(weights_path=self.az_weights_path)
            self._az_net_loaded_from = self.az_weights_path

        mcts = MCTS(self._az_net)

        start = time.perf_counter()
        visits = mcts.run(
            self.board,
            simulations=self.az_sims,
            c_puct=self.az_c_puct,
            add_root_noise=False,
            batch_size=self.az_batch_size,
            time_limit_ms=time_limit_ms,
            should_stop=lambda: self._stop,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if not visits:
            return SearchResult(best_move=None, score_cp=0, depth=1, nodes=0, time_ms=elapsed_ms, pv=[])

        best_move = max(visits.items(), key=lambda kv: kv[1])[0]
        nodes = sum(visits.values())

        # Rough score from NN value after playing best move.
        self.board.push(best_move)
        _logits, v = self._az_net.infer(self.board)
        self.board.pop()
        score_cp = int(1000.0 * v)

        pv = [best_move]
        if info_cb is not None:
            info_cb(1, score_cp, nodes, elapsed_ms, pv)

        return SearchResult(best_move=best_move, score_cp=score_cp, depth=1, nodes=nodes, time_ms=elapsed_ms, pv=pv)

    def stop(self) -> None:
        self._stop = True
        self.searcher.stop = True
