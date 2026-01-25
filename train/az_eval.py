from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

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
    elo_diff: float = 0.0  # Estimated Elo difference A vs B
    current_elo: float = 1000.0  # Running Elo estimate for current model


# ─────────────────────────────────────────────────────────────────────────────
# ELO TRACKING
# ─────────────────────────────────────────────────────────────────────────────

_STATE_FILE = Path("D:/NullMove/.az_state.json")


def _calculate_elo_diff(wins: int, losses: int, draws: int) -> float:
    """Calculate Elo difference from win/loss/draw record.
    
    Uses the standard Elo formula:
    score = (wins + 0.5*draws) / total_games
    elo_diff = -400 * log10(1/score - 1)
    
    Returns bounded estimate (-400 to +400 for reasonable values).
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0
    
    score = (wins + 0.5 * draws) / total
    
    # Clamp to avoid division by zero / infinite Elo
    score = max(0.01, min(0.99, score))
    
    elo_diff = -400 * math.log10(1 / score - 1)
    return round(elo_diff, 1)


def _load_current_elo() -> float:
    """Load current Elo from state file."""
    if _STATE_FILE.exists():
        try:
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
            return float(state.get("elo", 1000.0))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return 1000.0


def _save_current_elo(elo: float) -> None:
    """Save current Elo to state file."""
    state = {}
    if _STATE_FILE.exists():
        try:
            state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            pass
    state["elo"] = round(elo, 1)
    _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


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
) -> tuple[str, list[chess.Move]]:
    """Play a single evaluation game. Returns (result, moves_played)."""
    rng = random.Random(seed)
    board = chess.Board()
    moves_played: list[chess.Move] = []

    # Apply the same randomized opening to both players (fairness).
    if opening:
        for mv in opening:
            if board.is_game_over(claim_draw=True):
                break
            if mv not in board.legal_moves:
                break
            board.push(mv)
            moves_played.append(mv)

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
        moves_played.append(move)

    return board.result(claim_draw=True), moves_played


def _build_pgn_game(
    moves: list[chess.Move],
    result: str,
    white_name: str,
    black_name: str,
    game_index: int,
) -> chess.pgn.Game:
    """Build a PGN game object from a list of moves."""
    game = chess.pgn.Game()
    game.headers["Event"] = "NullMove AZ Eval"
    game.headers["Date"] = dt.date.today().isoformat()
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["PlyCount"] = str(len(moves))
    game.headers["GameIndex"] = str(game_index)
    
    node = game
    for mv in moves:
        node = node.add_variation(mv)
    
    return game


def _append_pgn(path: Path, game: chess.pgn.Game) -> None:
    """Append a PGN game to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(str(game))
        f.write("\n\n")


def _has_promotion(moves: list[chess.Move]) -> bool:
    """Check if any move in the game is a promotion."""
    for mv in moves:
        if mv.promotion is not None:
            return True
    return False


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
    pgn_out: Path | None = None,
    pgn_promos_out: Path | None = None,
    net_channels: int = 96,
    net_blocks: int = 8,
    net_amp: bool = True,
) -> EvalResult:
    a = AZNet(weights_path=str(a_path), channels=int(net_channels), blocks=int(net_blocks), use_amp=bool(net_amp))
    b = AZNet(weights_path=str(b_path), channels=int(net_channels), blocks=int(net_blocks), use_amp=bool(net_amp))

    a_wins = 0
    b_wins = 0
    draws = 0

    a_name = a_path.stem  # e.g., "az" from "az.pt"
    b_name = b_path.stem + "_prev"  # e.g., "az_prev" from "az.pt.prev"

    # Set up promotions PGN file (auto-derive from pgn_out if not specified)
    promos_out_eff: Path | None = pgn_promos_out
    if promos_out_eff is None and pgn_out is not None:
        promos_out_eff = pgn_out.with_name(pgn_out.stem + "_promos" + pgn_out.suffix)

    # Alternate colors to reduce bias.
    for g in range(games):
        opening = _random_opening_moves(seed + g * 104729, int(opening_plies))
        if g % 2 == 0:
            # A plays White, B plays Black
            res, moves = _play_game(a, b, sims, batch_size, c_puct, max_plies, seed + g, opening=opening)
            white_name, black_name = a_name, b_name
            aw, bw, dr = _result_to_wdl(res)
        else:
            # B plays White, A plays Black
            res, moves = _play_game(b, a, sims, batch_size, c_puct, max_plies, seed + g, opening=opening)
            white_name, black_name = b_name, a_name
            # If B was White, flip results back into A-vs-B frame.
            bw, aw, dr = _result_to_wdl(res)

        a_wins += aw
        b_wins += bw
        draws += dr

        # Save decisive games to PGN
        if pgn_out is not None and res in ("1-0", "0-1"):
            pgn_game = _build_pgn_game(moves, res, white_name, black_name, g)
            _append_pgn(pgn_out, pgn_game)
            print(f"  [EVAL] Decisive game {g}: {white_name} vs {black_name} = {res} (saved to PGN)")

        # Save games with promotions (milestone: AI learned pawn advancement)
        if promos_out_eff is not None and _has_promotion(moves):
            pgn_game = _build_pgn_game(moves, res, white_name, black_name, g)
            pgn_game.headers["Event"] = "NullMove AZ Eval (Promotions)"
            _append_pgn(promos_out_eff, pgn_game)
            print(f"  [EVAL] Promotion game {g}: {white_name} vs {black_name} = {res} (saved to promos PGN)")

    # Calculate Elo difference from this eval
    elo_diff = _calculate_elo_diff(a_wins, b_wins, draws)
    
    # Update running Elo estimate (Bayesian-style update with momentum)
    # K-factor scales how much each eval affects the running estimate
    current_elo = _load_current_elo()
    k_factor = 32  # Standard chess K-factor
    expected_score = 1 / (1 + 10 ** (-elo_diff / 400))
    actual_score = (a_wins + 0.5 * draws) / games if games > 0 else 0.5
    new_elo = current_elo + k_factor * (actual_score - 0.5)  # Gain if winning, lose if losing vs baseline
    
    # Also accumulate small bonus for beating baseline
    if elo_diff > 0:
        new_elo += elo_diff * 0.1  # Small permanent Elo gain for improvement
    
    _save_current_elo(new_elo)
    
    out = EvalResult(
        games=games, 
        a_wins=a_wins, 
        b_wins=b_wins, 
        draws=draws,
        elo_diff=elo_diff,
        current_elo=new_elo
    )

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
                        "elo_diff",
                        "current_elo",
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
                    elo_diff,
                    round(new_elo, 1),
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
