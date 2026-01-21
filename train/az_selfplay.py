from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from pathlib import Path

import chess
import chess.pgn

# Allow running as a script without requiring `pip install -e .`
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nullmove.az_features import move_to_action  # noqa: E402
from nullmove.az_mcts import MCTS  # noqa: E402
from nullmove.az_net import AZNet  # noqa: E402


_PIECE_VALUES: dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _material_from_pov(board: chess.Board) -> float:
    """Simple material score from side-to-move POV, normalized to [-1, 1]."""

    w = 0
    b = 0
    for pt, v in _PIECE_VALUES.items():
        w += len(board.pieces(pt, chess.WHITE)) * v
        b += len(board.pieces(pt, chess.BLACK)) * v

    diff = (w - b) if board.turn == chess.WHITE else (b - w)

    # Non-king material max is ~39 (8*1 + 2*3 + 2*3 + 2*5 + 1*9).
    return max(-1.0, min(1.0, float(diff) / 39.0))


def _append_pgn_game(path: Path, game: chess.pgn.Game) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as pf:
        pf.write(str(game))
        pf.write("\n\n")


def _append_pgn_text(path: Path, text: str) -> None:
    if not text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as pf:
        pf.write(text)


def _trim_pgn_games(path: Path, keep_last: int) -> None:
    if keep_last <= 0 or not path.exists():
        return
    games_list: list[chess.pgn.Game] = []
    with path.open("r", encoding="utf-8") as pf:
        while True:
            g = chess.pgn.read_game(pf)
            if g is None:
                break
            games_list.append(g)
    if len(games_list) <= keep_last:
        return
    games_list = games_list[-keep_last:]
    with path.open("w", encoding="utf-8") as pf:
        for g in games_list:
            pf.write(str(g))
            pf.write("\n\n")


def _append_line_locked(path: Path, line: str) -> None:
    """Append a single line to a shared log file safely across processes (Windows-friendly)."""

    if not line:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use a tiny byte-range lock so multiple workers can safely append.
    try:
        import msvcrt  # type: ignore

        with path.open("a+", encoding="utf-8") as f:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            except Exception:
                pass
            f.write(line.rstrip("\n") + "\n")
            f.flush()
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
    except Exception:
        # Fallback: best-effort append.
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
            f.flush()


def _ts_now() -> str:
    # Local time, stable and human-readable.
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _format_selfplay_line(*, game_index: int, result: str, moves_uci: list[str], elapsed_s: float | None = None) -> str:
    moves = " ".join(moves_uci)
    if elapsed_s is None:
        return f"[{_ts_now()}] selfplay game {game_index}: {result} moves {moves}"
    return f"[{_ts_now()}] selfplay game {game_index}: {result} moves {moves} (elapsed={elapsed_s:.1f}s)"


def result_to_value(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def generate_selfplay(
    out_path: Path,
    weights_path: Path,
    games: int,
    sims: int,
    batch_size: int,
    c_puct: float,
    max_plies: int,
    temp_plies: int,
    seed: int,
    pgn_out: Path | None = None,
    pgn_max_games: int = 0,
    pgn_every: int = 1,
    pgn_record_wins: bool = True,
    pgn_wins_out: Path | None = None,
    game_index_base: int = 0,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    temp_init: float = 1.25,
    temp_final: float = 0.5,
    sample_p_after_temp: float = 0.15,
    clear_tree_every_move: bool = False,
    adjudicate_draw: bool = False,
    draw_value_threshold: float = 0.12,
    draw_plies: int = 24,
    draw_min_ply: int = 120,
    resign_threshold: float | None = None,
    resign_plies: int = 8,
    print_moves: bool = False,
    draw_value: float = -1.0,
    repeat_penalty: float = 0.10,
    repeat_window: int = 12,
    workers: int = 1,
    net_channels: int = 96,
    net_blocks: int = 8,
    net_amp: bool = True,
    live_log: Path | None = None,
    live_log_master: Path | None = None,
    live_log_header: bool = True,
    emit_stdout: bool = True,
    pgn_every_per_worker: bool = False,
) -> int:
    wins_out_eff: Path | None = None
    if pgn_out is not None and bool(pgn_record_wins):
        wins_out_eff = pgn_wins_out
        if wins_out_eff is None:
            wins_out_eff = pgn_out.with_name(pgn_out.stem + "_wins" + pgn_out.suffix)

    if int(workers) > 1:
        return _generate_selfplay_parallel(
            out_path=out_path,
            weights_path=weights_path,
            games=games,
            sims=sims,
            batch_size=batch_size,
            c_puct=c_puct,
            max_plies=max_plies,
            temp_plies=temp_plies,
            seed=seed,
            pgn_out=pgn_out,
            pgn_max_games=pgn_max_games,
            pgn_every=pgn_every,
            pgn_record_wins=bool(pgn_record_wins),
            pgn_wins_out=wins_out_eff,
            pgn_every_per_worker=bool(pgn_every_per_worker),
            game_index_base=game_index_base,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            temp_init=temp_init,
            temp_final=temp_final,
            sample_p_after_temp=sample_p_after_temp,
            clear_tree_every_move=clear_tree_every_move,
            adjudicate_draw=adjudicate_draw,
            draw_value_threshold=draw_value_threshold,
            draw_plies=draw_plies,
            draw_min_ply=draw_min_ply,
            resign_threshold=resign_threshold,
            resign_plies=resign_plies,
            print_moves=print_moves,
            draw_value=draw_value,
            repeat_penalty=repeat_penalty,
            repeat_window=repeat_window,
            workers=workers,
            net_channels=int(net_channels),
            net_blocks=int(net_blocks),
            net_amp=bool(net_amp),
            live_log=live_log,
            live_log_master=live_log_master,
        )

    if live_log is not None:
        try:
            live_log.parent.mkdir(parents=True, exist_ok=True)
            live_log.touch(exist_ok=True)
            if bool(live_log_header):
                _append_line_locked(
                    live_log,
                    f"=== [{_ts_now()}] selfplay start seed={seed} games={games} sims={sims} batch={batch_size} ===",
                )
        except Exception:
            pass

    if live_log_master is not None and live_log_master != live_log:
        try:
            live_log_master.parent.mkdir(parents=True, exist_ok=True)
            live_log_master.touch(exist_ok=True)
            if bool(live_log_header):
                _append_line_locked(
                    live_log_master,
                    f"=== [{_ts_now()}] selfplay start seed={seed} games={games} sims={sims} batch={batch_size} ===",
                )
        except Exception:
            pass

    rng = random.Random(seed)
    net = AZNet(
        weights_path=str(weights_path),
        channels=int(net_channels),
        blocks=int(net_blocks),
        use_amp=bool(net_amp),
    )
    mcts = MCTS(net)

    # Early-stop MCTS when the root move is clearly decided.
    # This keeps self-play throughput high without changing the search algorithm.
    early_min = min(64, max(1, int(sims)))
    early_frac = 0.85
    early_lead = 32

    def _build_pgn_from_moves(moves: list[str]) -> chess.pgn.Game:
        gobj = chess.pgn.Game()
        node: chess.pgn.ChildNode | chess.pgn.Game = gobj
        b = chess.Board()
        for u in moves:
            try:
                mv = chess.Move.from_uci(u)
            except Exception:
                break
            if mv not in b.legal_moves:
                break
            b.push(mv)
            node = node.add_variation(mv)
        return gobj

    def _pick_move(
        items: list[tuple[chess.Move, float]],
        rng: random.Random,
        temperature: float,
    ) -> chess.Move:
        if not items:
            raise ValueError("no moves")
        if temperature <= 1e-6:
            return max(items, key=lambda x: x[1])[0]
        # Sample proportional to n^(1/T)
        inv_t = 1.0 / float(temperature)
        weights: list[float] = []
        for _mv, n in items:
            weights.append(max(0.0, float(n)) ** inv_t)
        s = sum(weights)
        if s <= 0:
            return items[0][0]
        r = rng.random() * s
        cum = 0.0
        for (mv, _n), w in zip(items, weights):
            cum += w
            if cum >= r:
                return mv
        return items[-1][0]

    written = 0
    t0_all = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for g in range(1, games + 1):
            t0_game = time.time()
            board = chess.Board()
            states: list[tuple[str, bool, dict[int, float], float]] = []
            moves_uci: list[str] = []

            rp = max(0.0, min(0.95, float(repeat_penalty)))
            rw = max(0, int(repeat_window))
            recent_counts: dict[object, int] = {}

            def _remember(key: object) -> None:
                if rw <= 0:
                    return
                recent_counts[key] = int(recent_counts.get(key, 0)) + 1

            def _forget(key: object) -> None:
                if rw <= 0:
                    return
                c = int(recent_counts.get(key, 0)) - 1
                if c <= 0:
                    recent_counts.pop(key, None)
                else:
                    recent_counts[key] = c

            recent_keys: list[object] = []
            if rw > 0:
                k0 = board._transposition_key()
                recent_keys.append(k0)
                _remember(k0)

            draw_streak = 0
            resign_streak = 0
            adjudicated_result: str | None = None

            write_pgn = False
            if pgn_out is not None:
                every = max(1, int(pgn_every))
                global_g = int(game_index_base) + g
                if bool(pgn_every_per_worker):
                    # In multi-worker mode, allow each worker to checkpoint every N games it plays.
                    # This makes PGN progress visible even when work is split across workers.
                    write_pgn = (g % every) == 0
                else:
                    # Global cadence (every N games overall).
                    write_pgn = (global_g % every) == 0

            pgn_game: chess.pgn.Game | None = None
            pgn_node: chess.pgn.ChildNode | chess.pgn.Game | None = None
            if write_pgn:
                pgn_game = chess.pgn.Game()
                pgn_node = pgn_game

            max_steps = int(max_plies)
            if max_steps <= 0:
                # "Unlimited" plies requested: rely on chess rules (mate/stalemate/claimable draws)
                # but still keep an extremely high safety bound.
                max_steps = 20000

            for ply in range(max_steps):
                if board.is_game_over(claim_draw=True):
                    break

                if clear_tree_every_move:
                    mcts.clear()

                visits = mcts.run(
                    board,
                    simulations=sims,
                    c_puct=c_puct,
                    add_root_noise=True,
                    batch_size=batch_size,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_eps=dirichlet_eps,
                    early_stop_min_sims=early_min,
                    early_stop_frac=early_frac,
                    early_stop_lead=early_lead,
                    # Speedups:
                    # - top-k expansion: only consider the highest-prior moves
                    # - progressive widening: gradually add moves as the node gets more visits
                    expand_topk=32,
                    pw_alpha=1.5,
                    pw_beta=0.5,
                )

                if not visits:
                    break

                # Mildly discourage repeating recent positions (keeps some shuffling possible).
                # We apply this both to move-choice and to the stored policy target.
                weighted: list[tuple[chess.Move, float]] = []
                for mv, n in visits.items():
                    w = float(n)
                    if rp > 0.0 and rw > 0:
                        board.push(mv)
                        k = board._transposition_key()
                        board.pop()
                        if k in recent_counts:
                            w *= (1.0 - rp)
                    weighted.append((mv, w))

                total_w = sum(w for _m, w in weighted)
                if total_w <= 0.0:
                    weighted = [(mv, float(n)) for mv, n in visits.items()]
                    total_w = sum(w for _m, w in weighted)

                pi: dict[int, float] = {}
                if total_w > 0.0:
                    for mv, w in weighted:
                        pi[move_to_action(mv)] = float(w) / float(total_w)

                states.append((board.fen(), board.turn == chess.WHITE, pi, _material_from_pov(board)))

                items = weighted
                if ply < temp_plies:
                    # Temperature schedule from temp_init -> temp_final over temp_plies.
                    if temp_plies <= 1:
                        t = float(temp_final)
                    else:
                        frac = float(ply) / float(max(1, temp_plies - 1))
                        t = (1.0 - frac) * float(temp_init) + frac * float(temp_final)
                    move = _pick_move(items, rng, temperature=t)
                else:
                    # After the opening, mostly pick best, but sometimes sample to avoid repeats.
                    if sample_p_after_temp > 0 and rng.random() < float(sample_p_after_temp):
                        move = _pick_move(items, rng, temperature=float(temp_final))
                    else:
                        move = max(items, key=lambda x: x[1])[0]

                board.push(move)
                moves_uci.append(move.uci())
                if pgn_node is not None:
                    pgn_node = pgn_node.add_variation(move)

                if rw > 0:
                    k = board._transposition_key()
                    recent_keys.append(k)
                    _remember(k)
                    if len(recent_keys) > rw:
                        old = recent_keys.pop(0)
                        _forget(old)

                # Optional adjudication to avoid long repetitive draws.
                # This does NOT change the training target format; it just ends games earlier.
                if adjudicate_draw and board.ply() >= int(draw_min_ply):
                    _logits, v_now = net.infer(board)
                    if abs(float(v_now)) <= float(draw_value_threshold):
                        draw_streak += 1
                    else:
                        draw_streak = 0
                    if draw_streak >= int(draw_plies):
                        adjudicated_result = "1/2-1/2"
                        break

                # Optional early resign (default disabled by leaving resign_threshold=None)
                if resign_threshold is not None and board.ply() >= int(draw_min_ply):
                    _logits, v_now = net.infer(board)
                    if float(v_now) <= float(resign_threshold):
                        resign_streak += 1
                    else:
                        resign_streak = 0
                    if resign_streak >= int(resign_plies):
                        # Side to move is losing heavily; treat as loss for side-to-move.
                        adjudicated_result = "0-1" if board.turn == chess.WHITE else "1-0"
                        break

            res = adjudicated_result or board.result(claim_draw=True)
            z_white = result_to_value(res)
            is_draw = res == "1/2-1/2"

            if print_moves:
                global_g = int(game_index_base) + g
                # Keep this compact so logs stay readable.
                line = _format_selfplay_line(game_index=global_g, result=res, moves_uci=moves_uci, elapsed_s=(time.time() - t0_game))
                if bool(emit_stdout):
                    print(line, flush=True)
                if live_log is not None:
                    _append_line_locked(live_log, line)
                if live_log_master is not None and live_log_master != live_log:
                    _append_line_locked(live_log_master, line)

            if pgn_game is not None and pgn_out is not None:
                pgn_game.headers["Event"] = "NullMove AZ Self-Play"
                pgn_game.headers["Date"] = dt.date.today().isoformat()
                pgn_game.headers["Result"] = res
                pgn_game.headers["PlyCount"] = str(board.ply())
                pgn_game.headers["GameIndex"] = str(int(game_index_base) + g)
                _append_pgn_game(pgn_out, pgn_game)
                _trim_pgn_games(pgn_out, pgn_max_games)

            # Always record decisive games (wins) into a dedicated PGN file.
            if wins_out_eff is not None and res in ("1-0", "0-1"):
                wg = pgn_game if pgn_game is not None else _build_pgn_from_moves(moves_uci)
                wg.headers["Event"] = "NullMove AZ Self-Play (Wins)"
                wg.headers["Date"] = dt.date.today().isoformat()
                wg.headers["Result"] = res
                wg.headers["PlyCount"] = str(board.ply())
                wg.headers["GameIndex"] = str(int(game_index_base) + g)
                _append_pgn_game(wins_out_eff, wg)
                if int(pgn_max_games) > 0:
                    _trim_pgn_games(wins_out_eff, int(pgn_max_games))

            for fen, stm_is_white, pi, mat in states:
                # If we want to punish draws, do it symmetrically: both sides see a negative target.
                if is_draw:
                    z = float(draw_value)
                else:
                    z = z_white if stm_is_white else -z_white
                policy_sparse = [[int(a), float(p)] for a, p in pi.items()]
                f.write(json.dumps({"fen": fen, "policy": policy_sparse, "z": float(z), "m": float(mat)}) + "\n")
                written += 1

            if bool(emit_stdout):
                print(f"[{_ts_now()}] game {g}/{games}: {res} examples {len(states)} total_examples={written}", flush=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="az_dataset.jsonl")
    ap.add_argument(
        "--pgn-out",
        default=str(Path("D:/NullMove/Self_Play_Games/az_progress.pgn")),
        help="PGN file to append checkpoint self-play games",
    )
    ap.add_argument("--pgn-max-games", type=int, default=0, help="Keep only last N PGN games (0 = keep all)")
    ap.add_argument("--pgn-every", type=int, default=100, help="Only append a PGN once every N games")
    ap.add_argument(
        "--pgn-every-per-worker",
        action="store_true",
        help="When --workers > 1, apply --pgn-every to each worker's local game counter",
    )
    ap.add_argument(
        "--live-log",
        default=str(Path("D:/NullMove/logs/selfplay_live.log")),
        help="Append per-game move lists here (safe across workers); empty disables",
    )
    ap.add_argument("--weights", default="az.pt")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=1, help="Run self-play games in multiple processes (merges outputs safely)")
    ap.add_argument("--c_puct", type=float, default=1.5)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--max-plies", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp-plies", type=int, default=30, help="Use temperature sampling for first N plies")
    ap.add_argument("--temp-init", type=float, default=1.25)
    ap.add_argument("--temp-final", type=float, default=0.5)
    ap.add_argument("--sample-p-after-temp", type=float, default=0.15, help="After temp_plies, sample with probability p")
    ap.add_argument("--clear-tree-every-move", action="store_true", help="Clear MCTS tree every move (more variety, slower/weaker)")
    ap.add_argument("--epsilon", type=float, default=0.0, help="Extra random move chance (debug)")

    # Keep the remaining args below this point unchanged.

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--adjudicate-draw", action="store_true", help="Enable early draw adjudication")
    g.add_argument("--no-adjudicate-draw", action="store_true", help="Disable early draw adjudication (legacy flag)")
    ap.add_argument("--draw-value-threshold", type=float, default=0.12)
    ap.add_argument("--draw-plies", type=int, default=24)
    ap.add_argument("--draw-min-ply", type=int, default=120)
    ap.add_argument("--draw-value", type=float, default=-0.2, help="Training target value to use for drawn games")

    ap.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.10,
        help="Multiply move probability by (1-penalty) if it repeats a recent position (0 disables)",
    )
    ap.add_argument(
        "--repeat-window",
        type=int,
        default=12,
        help="How many recent positions to consider for repeat-penalty (0 disables)",
    )

    ap.add_argument("--resign-threshold", type=float, default=999.0, help="Set < 0 to enable early resign (disabled by default)")
    ap.add_argument("--resign-plies", type=int, default=8)
    gpm = ap.add_mutually_exclusive_group()
    gpm.add_argument("--print-moves", dest="print_moves", action="store_true", help="Print one line per game with the move list")
    gpm.add_argument("--no-print-moves", dest="print_moves", action="store_false", help="Do not print per-game move lists")
    ap.set_defaults(print_moves=True)
    args = ap.parse_args()

    if args.epsilon != 0.0:
        raise SystemExit("--epsilon debug path removed; use --batch-size and --sims for strength")

    generate_selfplay(
        out_path=Path(args.out),
        weights_path=Path(args.weights),
        games=args.games,
        sims=args.sims,
        batch_size=args.batch_size,
        c_puct=args.c_puct,
        max_plies=args.max_plies,
        temp_plies=args.temp_plies,
        seed=args.seed,
        pgn_out=(Path(args.pgn_out) if args.pgn_out else None),
        pgn_max_games=args.pgn_max_games,
        pgn_every=args.pgn_every,
        pgn_every_per_worker=bool(getattr(args, "pgn_every_per_worker", False)),
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        temp_init=args.temp_init,
        temp_final=args.temp_final,
        sample_p_after_temp=args.sample_p_after_temp,
        clear_tree_every_move=bool(args.clear_tree_every_move),
        adjudicate_draw=bool(args.adjudicate_draw),
        draw_value_threshold=args.draw_value_threshold,
        draw_plies=args.draw_plies,
        draw_min_ply=args.draw_min_ply,
        resign_threshold=(None if args.resign_threshold >= 998.0 else float(args.resign_threshold)),
        resign_plies=args.resign_plies,
        print_moves=bool(args.print_moves),
        draw_value=float(args.draw_value),
        repeat_penalty=float(args.repeat_penalty),
        repeat_window=int(args.repeat_window),
        workers=int(getattr(args, "workers", 1)),
        live_log=(Path(args.live_log) if str(getattr(args, "live_log", "")).strip() else None),
    )


def _worker_entry(kwargs: dict) -> int:
    # Isolated entrypoint for multiprocessing on Windows.
    return generate_selfplay(
        workers=1,
        live_log_header=False,
        emit_stdout=False,
        **kwargs,
    )


def _generate_selfplay_parallel(
    *,
    out_path: Path,
    weights_path: Path,
    games: int,
    sims: int,
    batch_size: int,
    c_puct: float,
    max_plies: int,
    temp_plies: int,
    seed: int,
    pgn_out: Path | None,
    pgn_max_games: int,
    pgn_every: int,
    pgn_record_wins: bool,
    pgn_wins_out: Path | None,
    pgn_every_per_worker: bool,
    game_index_base: int,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    temp_init: float,
    temp_final: float,
    sample_p_after_temp: float,
    clear_tree_every_move: bool,
    adjudicate_draw: bool,
    draw_value_threshold: float,
    draw_plies: int,
    draw_min_ply: int,
    resign_threshold: float | None,
    resign_plies: int,
    print_moves: bool,
    draw_value: float,
    repeat_penalty: float,
    repeat_window: int,
    workers: int,
    net_channels: int,
    net_blocks: int,
    net_amp: bool,
    live_log: Path | None,
    live_log_master: Path | None,
) -> int:
    import multiprocessing as mp

    w = max(1, int(workers))
    if w <= 1 or int(games) <= 1:
        return generate_selfplay(
            out_path=out_path,
            weights_path=weights_path,
            games=games,
            sims=sims,
            batch_size=batch_size,
            c_puct=c_puct,
            max_plies=max_plies,
            temp_plies=temp_plies,
            seed=seed,
            pgn_out=pgn_out,
            pgn_max_games=pgn_max_games,
            pgn_every=pgn_every,
            pgn_record_wins=bool(pgn_record_wins),
            pgn_wins_out=pgn_wins_out,
            pgn_every_per_worker=bool(pgn_every_per_worker),
            game_index_base=game_index_base,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            temp_init=temp_init,
            temp_final=temp_final,
            sample_p_after_temp=sample_p_after_temp,
            clear_tree_every_move=clear_tree_every_move,
            adjudicate_draw=adjudicate_draw,
            draw_value_threshold=draw_value_threshold,
            draw_plies=draw_plies,
            draw_min_ply=draw_min_ply,
            resign_threshold=resign_threshold,
            resign_plies=resign_plies,
            print_moves=print_moves,
            draw_value=draw_value,
            repeat_penalty=repeat_penalty,
            repeat_window=repeat_window,
            workers=1,
            net_channels=net_channels,
            net_blocks=net_blocks,
            net_amp=net_amp,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "_selfplay_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # If a live_log is provided, treat it as the master combined log and also emit per-worker logs.
    live_log_master_eff = live_log_master
    live_log_eff = live_log
    if live_log_master_eff is None and live_log_eff is not None:
        live_log_master_eff = live_log_eff

    if live_log_master_eff is not None:
        try:
            live_log_master_eff.parent.mkdir(parents=True, exist_ok=True)
            live_log_master_eff.touch(exist_ok=True)
            _append_line_locked(
                live_log_master_eff,
                f"=== [{_ts_now()}] selfplay(parallel) start seed={seed} games={games} workers={w} sims={sims} batch={batch_size} ===",
            )
        except Exception:
            pass

    # Split games roughly evenly.
    splits: list[int] = [games // w] * w
    for i in range(games % w):
        splits[i] += 1

    jobs: list[dict] = []
    game_cursor = int(game_index_base)
    for wi, gcount in enumerate(splits):
        if gcount <= 0:
            continue
        tmp_out = tmp_dir / f"chunk_{wi}_{seed}.jsonl"
        # Avoid concurrent writes to the final PGN by having each worker write its own chunk.
        # We'll merge these chunks deterministically after all workers finish.
        this_pgn = (tmp_dir / f"chunk_{wi}_{seed}.pgn") if (pgn_out is not None) else None
        this_pgn_wins = None
        if pgn_wins_out is not None and bool(pgn_record_wins):
            this_pgn_wins = tmp_dir / f"chunk_{wi}_{seed}.wins.pgn"
        # Per-worker live log (optional) + a shared master log.
        worker_live_log: Path | None = None
        if live_log_eff is not None:
            # Example: selfplay_live_w0.log, selfplay_live_w1.log, ...
            suffix = live_log_eff.suffix or ".log"
            worker_live_log = live_log_eff.with_name(f"{live_log_eff.stem}_w{wi}{suffix}")
            try:
                worker_live_log.parent.mkdir(parents=True, exist_ok=True)
                worker_live_log.touch(exist_ok=True)
                _append_line_locked(
                    worker_live_log,
                    f"=== [{_ts_now()}] worker {wi} start seed={int(seed) + wi * 1337} games={gcount} ===",
                )
            except Exception:
                pass

        jobs.append(
            dict(
                out_path=tmp_out,
                weights_path=weights_path,
                games=gcount,
                sims=sims,
                batch_size=batch_size,
                c_puct=c_puct,
                max_plies=max_plies,
                temp_plies=temp_plies,
                seed=int(seed) + wi * 1337,
                pgn_out=this_pgn,
                pgn_wins_out=this_pgn_wins,
                pgn_max_games=pgn_max_games,
                pgn_every=pgn_every,
                pgn_record_wins=bool(pgn_record_wins),
                pgn_every_per_worker=bool(pgn_every_per_worker),
                game_index_base=game_cursor,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                temp_init=temp_init,
                temp_final=temp_final,
                sample_p_after_temp=sample_p_after_temp,
                clear_tree_every_move=clear_tree_every_move,
                adjudicate_draw=adjudicate_draw,
                draw_value_threshold=draw_value_threshold,
                draw_plies=draw_plies,
                draw_min_ply=draw_min_ply,
                resign_threshold=resign_threshold,
                resign_plies=resign_plies,
                print_moves=print_moves,
                draw_value=draw_value,
                repeat_penalty=repeat_penalty,
                repeat_window=repeat_window,
                net_channels=net_channels,
                net_blocks=net_blocks,
                net_amp=net_amp,
                live_log=worker_live_log,
                live_log_master=live_log_master_eff,
            )
        )
        game_cursor += int(gcount)

    ctx = mp.get_context("spawn")
    t0 = time.time()
    # For better visibility, stream new lines from the shared master log while workers run.
    log_path_for_stream: Path | None = live_log_master_eff
    log_offset = 0

    def _stream_new_log_lines() -> int:
        nonlocal log_offset
        if log_path_for_stream is None or not log_path_for_stream.exists():
            return 0
        try:
            with log_path_for_stream.open("r", encoding="utf-8", errors="replace") as lf:
                lf.seek(log_offset)
                data = lf.read()
                log_offset = lf.tell()
            if not data:
                return 0
            # Print to parent stdout so it shows up in az_loop stdout log / tail.
            for line in data.splitlines():
                if line.strip():
                    print(line, flush=True)
            return 1
        except Exception:
            return 0

    with ctx.Pool(processes=w) as pool:
        async_res = pool.map_async(_worker_entry, jobs)
        # Poll until workers finish.
        while not async_res.ready():
            _stream_new_log_lines()
            time.sleep(0.5)
        written_parts = async_res.get()
        # Flush any remaining lines.
        _stream_new_log_lines()

    # Print a short end-of-selfplay summary.
    print(f"=== [{_ts_now()}] selfplay(parallel) done games={games} workers={w} elapsed={time.time() - t0:.1f}s ===", flush=True)

    # Merge outputs deterministically.
    merged = 0
    with out_path.open("a", encoding="utf-8") as out_f:
        for j in jobs:
            p = Path(j["out_path"])
            if not p.exists():
                continue
            out_f.write(p.read_text(encoding="utf-8"))
            merged += int(p.stat().st_size)
            try:
                p.unlink()
            except Exception:
                pass

    # Merge PGN chunks deterministically (in game-index order).
    if pgn_out is not None:
        for j in jobs:
            pgn_chunk = j.get("pgn_out")
            if not pgn_chunk:
                continue
            pp = Path(pgn_chunk)
            if not pp.exists():
                continue
            try:
                _append_pgn_text(pgn_out, pp.read_text(encoding="utf-8"))
            finally:
                try:
                    pp.unlink()
                except Exception:
                    pass
        _trim_pgn_games(pgn_out, int(pgn_max_games))

    # Merge wins PGN chunks deterministically.
    if pgn_wins_out is not None and bool(pgn_record_wins):
        for j in jobs:
            pgn_chunk = j.get("pgn_wins_out")
            if not pgn_chunk:
                continue
            pp = Path(pgn_chunk)
            if not pp.exists():
                continue
            try:
                _append_pgn_text(pgn_wins_out, pp.read_text(encoding="utf-8"))
            finally:
                try:
                    pp.unlink()
                except Exception:
                    pass
        if int(pgn_max_games) > 0:
            _trim_pgn_games(pgn_wins_out, int(pgn_max_games))

    try:
        # Clean directory if empty.
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass

    return int(sum(written_parts))


if __name__ == "__main__":
    main()
