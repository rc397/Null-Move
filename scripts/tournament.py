from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass

import chess
import chess.engine


logging.getLogger("chess.engine").setLevel(logging.ERROR)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    cmd: list[str]


def play_one(white: chess.engine.SimpleEngine, black: chess.engine.SimpleEngine, limit: chess.engine.Limit, max_plies: int = 300) -> str:
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        engine = white if board.turn == chess.WHITE else black
        result = engine.play(board, limit)
        board.push(result.move)
    return board.result(claim_draw=True)


def main() -> None:
    # Manual parsing so engine commands can include flags like `-m`.
    # Usage:
    #   tournament.py --engine Name python.exe -m nullmove.uci --engine SF stockfish --games 4 --depth 6
    #   tournament.py --engine Name=D:/path/to/python.exe -m nullmove.uci --engine SF=stockfish --games 4 --movetime 200
    argv = sys.argv[1:]
    engine_specs: list[EngineSpec] = []
    rest: list[str] = []

    stop_tokens = {"--engine", "--games", "--depth", "--movetime", "-h", "--help"}
    i = 0
    while i < len(argv):
        if argv[i] == "--engine":
            if i + 2 >= len(argv):
                raise SystemExit("--engine needs: Name cmd...")
            name_token = argv[i + 1]
            name = name_token
            cmd: list[str] = []
            i += 2

            # Support a common convenience form: --engine Name=cmd ...
            if "=" in name_token:
                name, first = name_token.split("=", 1)
                name = name.strip()
                first = first.strip()
                if first:
                    cmd.append(first)
            while i < len(argv) and argv[i] not in stop_tokens:
                cmd.append(argv[i])
                i += 1
            if not cmd:
                raise SystemExit(f"--engine {name} missing command")
            engine_specs.append(EngineSpec(name=name.strip(), cmd=cmd))
        else:
            rest.append(argv[i])
            i += 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2, help="Games per pairing (includes both colors)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--movetime", type=int, default=0)
    args = ap.parse_args(rest)

    engines = engine_specs
    if len(engines) < 2:
        raise SystemExit("Need at least two --engine entries")

    if args.movetime > 0:
        limit = chess.engine.Limit(time=args.movetime / 1000.0)
    else:
        limit = chess.engine.Limit(depth=args.depth)

    scores = {e.name: 0.0 for e in engines}

    # Round-robin
    for a, b in itertools.combinations(engines, 2):
        with chess.engine.SimpleEngine.popen_uci(a.cmd) as ea, chess.engine.SimpleEngine.popen_uci(b.cmd) as eb:
            for g in range(args.games):
                if g % 2 == 0:
                    res = play_one(ea, eb, limit)
                    white_name, black_name = a.name, b.name
                else:
                    res = play_one(eb, ea, limit)
                    white_name, black_name = b.name, a.name

                if res == "1-0":
                    scores[white_name] += 1.0
                elif res == "0-1":
                    scores[black_name] += 1.0
                else:
                    scores[white_name] += 0.5
                    scores[black_name] += 0.5

                print(f"{white_name} vs {black_name}: {res}")

    print("\nFinal scores")
    for name, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {sc}")


if __name__ == "__main__":
    main()
