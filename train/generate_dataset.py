from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import chess
import chess.engine


def random_play_positions(games: int, plies: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    fens: list[str] = []
    for _ in range(games):
        board = chess.Board()
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
            fens.append(board.fen())
    return fens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish", required=True, help="Path/command to a UCI eval engine (e.g. stockfish)")
    ap.add_argument("--out", default="dataset.jsonl")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--plies", type=int, default=40)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cmd = args.stockfish.split()
    out = Path(args.out)

    fens = random_play_positions(args.games, args.plies, args.seed)

    with chess.engine.SimpleEngine.popen_uci(cmd) as eng, out.open("w", encoding="utf-8") as f:
        for i, fen in enumerate(fens, 1):
            board = chess.Board(fen)
            info = eng.analyse(board, chess.engine.Limit(depth=args.depth))
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=100000)
            if cp is None:
                continue
            f.write(json.dumps({"fen": fen, "cp": int(cp)}) + "\n")
            if i % 500 == 0:
                print(f"{i}/{len(fens)}")


if __name__ == "__main__":
    main()
