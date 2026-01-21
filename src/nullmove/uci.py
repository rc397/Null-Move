from __future__ import annotations

import sys
from dataclasses import dataclass

from .engine import Engine
from .types import SearchLimits


@dataclass
class UCIState:
    engine: Engine


def _parse_go(tokens: list[str]) -> SearchLimits:
    limits = SearchLimits()

    i = 0
    while i < len(tokens):
        t = tokens[i]

        def next_int() -> int:
            nonlocal i
            i += 1
            return int(tokens[i])

        if t == "depth":
            limits = SearchLimits(**{**limits.__dict__, "depth": next_int()})
        elif t == "movetime":
            limits = SearchLimits(**{**limits.__dict__, "movetime_ms": next_int()})
        elif t == "wtime":
            limits = SearchLimits(**{**limits.__dict__, "wtime_ms": next_int()})
        elif t == "btime":
            limits = SearchLimits(**{**limits.__dict__, "btime_ms": next_int()})
        elif t == "winc":
            limits = SearchLimits(**{**limits.__dict__, "winc_ms": next_int()})
        elif t == "binc":
            limits = SearchLimits(**{**limits.__dict__, "binc_ms": next_int()})
        elif t == "infinite":
            limits = SearchLimits(**{**limits.__dict__, "infinite": True})
        # ignore: nodes, mate, movestogo, ponder, binc/winc already handled
        i += 1

    return limits


def _handle_position(state: UCIState, args: list[str]) -> None:
    # position startpos moves e2e4 e7e5
    # position fen <fen...> moves ...
    if not args:
        return

    fen: str | None = None
    moves: list[str] = []

    if args[0] == "startpos":
        fen = None
        rest = args[1:]
    elif args[0] == "fen":
        # FEN is 6 space-separated fields
        fen = " ".join(args[1:7])
        rest = args[7:]
    else:
        return

    if rest and rest[0] == "moves":
        moves = rest[1:]

    state.engine.set_position(fen=fen, moves_uci=moves)


def main() -> None:
    engine = Engine()
    state = UCIState(engine=engine)

    # Basic UCI handshake
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            print("id name NullMove")
            print("id author You")
            print("option name UseNN type check default false")
            print("option name NNWeights type string default nn.pt")
            print("option name UseAZ type check default false")
            print("option name AZWeights type string default az.pt")
            print("option name AZSims type spin default 200 min 1 max 100000")
            print("option name AZBatchSize type spin default 16 min 1 max 1024")
            print("option name AZCPuct type string default 1.5")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd == "ucinewgame":
            state.engine.new_game()
            # UCI spec doesn't require any output here; GUIs often send isready separately.

        elif cmd == "position":
            _handle_position(state, args)

        elif cmd == "go":
            limits = _parse_go(args)
            def info_cb(depth: int, score_cp: int, nodes: int, time_ms: int, pv: list) -> None:
                pv_uci = " ".join(m.uci() for m in pv)
                print(f"info depth {depth} score cp {score_cp} nodes {nodes} time {time_ms} pv {pv_uci}".strip())
                sys.stdout.flush()

            result = state.engine.go(limits, info_cb=info_cb)
            best = result.best_move.uci() if result.best_move else "0000"
            print(f"bestmove {best}")
            sys.stdout.flush()

        elif cmd == "stop":
            state.engine.stop()

        elif cmd == "setoption":
            # setoption name <id> [value <x>]
            if "name" not in args:
                continue
            name_i = args.index("name")
            try:
                value_i = args.index("value")
            except ValueError:
                value_i = -1

            if value_i == -1:
                name = " ".join(args[name_i + 1 :])
                value = "true"
            else:
                name = " ".join(args[name_i + 1 : value_i])
                value = " ".join(args[value_i + 1 :])
            state.engine.set_option(name, value)

        elif cmd == "quit":
            return

        # ignore: setoption, debug, ponderhit, etc.


if __name__ == "__main__":
    main()
