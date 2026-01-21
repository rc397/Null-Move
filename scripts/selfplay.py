from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn


logging.getLogger("chess.engine").setLevel(logging.ERROR)


def play_game(
    engine_white: chess.engine.SimpleEngine,
    engine_black: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    max_plies: int = 300,
) -> chess.pgn.Game:
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        engine = engine_white if board.turn == chess.WHITE else engine_black
        result = engine.play(board, limit)
        board.push(result.move)
        node = node.add_variation(result.move)

    game.headers["Event"] = "NullMove Self-Play"
    game.headers["Date"] = dt.date.today().isoformat()
    game.headers["Result"] = board.result(claim_draw=True)
    game.headers["PlyCount"] = str(board.ply())
    return game


def main() -> None:
    # Manual parsing so engine commands can include flags like `-m`.
    argv = sys.argv[1:]
    cmd = ["D:/NullMove/.venv/Scripts/python.exe", "-m", "nullmove.uci"]
    rest: list[str] = []

    stop_tokens = {"--games", "--depth", "--movetime", "--out", "-h", "--help"}
    i = 0
    while i < len(argv):
        if argv[i] == "--engine":
            i += 1
            if i >= len(argv):
                raise SystemExit("--engine missing command")
            cmd = []
            while i < len(argv) and argv[i] not in stop_tokens:
                cmd.append(argv[i])
                i += 1
            if not cmd:
                raise SystemExit("--engine missing command")

            # Allow --engine Name=cmd ... as a convenience; ignore name.
            if cmd and "=" in cmd[0]:
                _, first = cmd[0].split("=", 1)
                cmd[0] = first
        else:
            rest.append(argv[i])
            i += 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--movetime", type=int, default=0, help="ms per move (0 = use depth)")
    ap.add_argument("--out", default=str(Path("D:/NullMove/Self_Play_Games/selfplay.pgn")))
    args = ap.parse_args(rest)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.movetime > 0:
        limit = chess.engine.Limit(time=args.movetime / 1000.0)
    else:
        limit = chess.engine.Limit(depth=args.depth)

    with chess.engine.SimpleEngine.popen_uci(cmd) as e1, chess.engine.SimpleEngine.popen_uci(cmd) as e2:
        with out_path.open("w", encoding="utf-8") as f:
            for i in range(args.games):
                game = play_game(e1, e2, limit)
                f.write(str(game))
                f.write("\n\n")
                print(f"game {i+1}/{args.games}: {game.headers['Result']}")


if __name__ == "__main__":
    main()
