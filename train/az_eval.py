from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
from dataclasses import dataclass
from pathlib import Path

import chess

# Allow running as a script without requiring `pip install -e .`
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nullmove.az_mcts import MCTS  # noqa: E402
from nullmove.az_net import AZNet  # noqa: E402


@dataclass
class EvalResult:
    games: int
    a_wins: int
    b_wins: int
    draws: int


def _random_opening_moves(seed: int, plies: int) -> list[chess.Move]:
    if plies <= 0:
        return []
    rng = random.Random(seed)
    b = chess.Board()
    moves: list[chess.Move] = []
    for _ in range(int(plies)):
        if b.is_game_over(claim_draw=True):
            break
        legal = list(b.legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        b.push(mv)
        moves.append(mv)
    return moves


def _result_to_wdl(result: str) -> tuple[int, int, int]:
    if result == "1-0":
        return (1, 0, 0)
    if result == "0-1":
        return (0, 1, 0)
    return (0, 0, 1)


def _play_game(
    net_white: AZNet,
    net_black: AZNet,
    sims: int,
    batch_size: int,
    c_puct: float,
    max_plies: int,
    seed: int,
    opening: list[chess.Move] | None = None,
) -> str:
    rng = random.Random(seed)
    board = chess.Board()

    # Apply the same randomized opening to both players (fairness).
    if opening:
        for mv in opening:
            if board.is_game_over(claim_draw=True):
                break
            if mv not in board.legal_moves:
                break
            board.push(mv)

    mcts_w = MCTS(net_white)
    mcts_b = MCTS(net_black)

    max_steps = int(max_plies)
    if max_steps <= 0:
        max_steps = 20000

    for _ply in range(max_steps):
        if board.is_game_over(claim_draw=True):
            break

        mcts = mcts_w if board.turn == chess.WHITE else mcts_b
        visits = mcts.run(
            board,
            simulations=sims,
            c_puct=c_puct,
            add_root_noise=False,
            batch_size=batch_size,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
        )
        if not visits:
            break

        # Deterministic: pick most visited.
        move = max(visits.items(), key=lambda kv: kv[1])[0]

        # Tiny tie-break to avoid pathological equal-count loops.
        if rng.random() < 0.0001:
            move = rng.choice(list(visits.keys()))

        board.push(move)

    return board.result(claim_draw=True)


def eval_weights(
    a_path: Path,
    b_path: Path,
    games: int,
    sims: int,
    batch_size: int,
    c_puct: float,
    max_plies: int,
    seed: int,
    opening_plies: int = 6,
    csv_out: Path | None = None,
    net_channels: int = 96,
    net_blocks: int = 8,
    net_amp: bool = True,
) -> EvalResult:
    a = AZNet(weights_path=str(a_path), channels=int(net_channels), blocks=int(net_blocks), use_amp=bool(net_amp))
    b = AZNet(weights_path=str(b_path), channels=int(net_channels), blocks=int(net_blocks), use_amp=bool(net_amp))

    a_wins = 0
    b_wins = 0
    draws = 0

    # Alternate colors to reduce bias.
    for g in range(games):
        opening = _random_opening_moves(seed + g * 104729, int(opening_plies))
        if g % 2 == 0:
            res = _play_game(a, b, sims, batch_size, c_puct, max_plies, seed + g, opening=opening)
            aw, bw, dr = _result_to_wdl(res)
        else:
            res = _play_game(b, a, sims, batch_size, c_puct, max_plies, seed + g, opening=opening)
            # If B was White, flip results back into A-vs-B frame.
            bw, aw, dr = _result_to_wdl(res)

        a_wins += aw
        b_wins += bw
        draws += dr

    out = EvalResult(games=games, a_wins=a_wins, b_wins=b_wins, draws=draws)

    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        new_file = not csv_out.exists()
        with csv_out.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    [
                        "utc_time",
                        "a_path",
                        "b_path",
                        "games",
                        "a_wins",
                        "b_wins",
                        "draws",
                        "sims",
                        "batch_size",
                        "c_puct",
                        "max_plies",
                        "opening_plies",
                        "seed",
                    ]
                )
            w.writerow(
                [
                    dt.datetime.utcnow().isoformat(),
                    str(a_path),
                    str(b_path),
                    games,
                    a_wins,
                    b_wins,
                    draws,
                    sims,
                    batch_size,
                    c_puct,
                    max_plies,
                    int(opening_plies),
                    seed,
                ]
            )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Weights A (typically current/new)")
    ap.add_argument("--b", required=True, help="Weights B (baseline/previous)")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--c_puct", type=float, default=1.5)
    ap.add_argument("--max-plies", type=int, default=250)
    ap.add_argument("--opening-plies", type=int, default=6, help="Random legal opening plies for more diverse eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv-out", default=str(Path("D:/NullMove/logs/az_eval.csv")))
    args = ap.parse_args()

    out = eval_weights(
        a_path=Path(args.a),
        b_path=Path(args.b),
        games=int(args.games),
        sims=int(args.sims),
        batch_size=int(args.batch_size),
        c_puct=float(args.c_puct),
        max_plies=int(args.max_plies),
        seed=int(args.seed),
        opening_plies=int(args.opening_plies),
        csv_out=(Path(args.csv_out) if args.csv_out else None),
    )

    print(f"eval A vs B over {out.games} games: A_wins={out.a_wins} B_wins={out.b_wins} draws={out.draws}")


if __name__ == "__main__":
    main()
