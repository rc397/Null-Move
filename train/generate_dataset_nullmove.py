import argparse
import json
import random
import sys
from pathlib import Path

import chess

# Allow running as a script without requiring `pip install -e .`
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nullmove.search import Searcher  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset.jsonl")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--play-depth", type=int, default=2, help="Depth used to pick self-play moves")
    ap.add_argument("--label-depth", type=int, default=6, help="Depth used to label positions (cp)")
    ap.add_argument("--stride", type=int, default=2, help="Record every N plies")
    ap.add_argument("--epsilon", type=float, default=0.05, help="Random move probability for diversity")
    ap.add_argument("--max-positions", type=int, default=50000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)

    play_search = Searcher()
    label_search = Searcher()

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for g in range(1, args.games + 1):
            board = chess.Board()

            for ply in range(args.max_plies):
                if board.is_game_over(claim_draw=True):
                    break

                # Sample this position for labeling
                if (ply % args.stride) == 0:
                    # Label with deeper search score from side-to-move
                    res = label_search.search(board, max_depth=args.label_depth, time_limit_ms=None)
                    if res.depth > 0:
                        f.write(json.dumps({"fen": board.fen(), "cp": int(res.score_cp)}) + "\n")
                        written += 1

                        if written % 500 == 0:
                            print(f"positions: {written} (game {g}/{args.games})")

                        if written >= args.max_positions:
                            print(f"hit max positions: {written}")
                            return

                # Pick a move (mostly best move, sometimes random)
                if rng.random() < args.epsilon:
                    move = rng.choice(list(board.legal_moves))
                else:
                    res = play_search.search(board, max_depth=args.play_depth, time_limit_ms=None)
                    move = res.best_move if res.best_move is not None else rng.choice(list(board.legal_moves))

                board.push(move)

    print(f"wrote {written} positions to {out}")


if __name__ == "__main__":
    main()
