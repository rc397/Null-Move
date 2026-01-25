from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import signal
import time

import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from az_selfplay import generate_selfplay, generate_selfplay_streaming  # noqa: E402
from az_train import train_once  # noqa: E402
from az_eval import eval_weights  # noqa: E402


# Graceful shutdown support
_STOP_FILE = _ROOT / ".az_stop"
_SHUTDOWN_REQUESTED = False

# Persistent state file (survives restarts)
_STATE_FILE = _ROOT / ".az_state.json"


def _load_state() -> dict:
    """Load persistent training state from disk."""
    if _STATE_FILE.exists():
        try:
            with _STATE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"global_iteration": 0, "global_games": 0}


def _save_state(state: dict) -> None:
    """Save persistent training state to disk."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save state: {e}")


def _check_shutdown() -> bool:
    """Check if graceful shutdown was requested (via stop file or signal)."""
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        return True
    if _STOP_FILE.exists():
        _SHUTDOWN_REQUESTED = True
        try:
            _STOP_FILE.unlink()
        except Exception:
            pass
        return True
    return False


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM gracefully."""
    global _SHUTDOWN_REQUESTED
    print(f"\n[SHUTDOWN] Signal {signum} received - will exit after current iteration completes...")
    _SHUTDOWN_REQUESTED = True


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _acquire_single_instance_lock(lock_path: Path) -> None:
    """Best-effort single-instance lock via an exclusive lock file.

    Prevents accidental double-runs writing to the same dataset/weights.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    now = time.time()

    def _pid_is_running(other_pid: int) -> bool:
        if other_pid <= 0:
            return False
        try:
            # Works on Windows too; raises if process doesn't exist.
            os.kill(other_pid, 0)
            return True
        except Exception:
            return False

    if lock_path.exists():
        try:
            txt = lock_path.read_text(encoding="utf-8").strip()
            parts = txt.split()
            other_pid = int(parts[0]) if parts else 0
        except Exception:
            other_pid = 0

        if _pid_is_running(other_pid):
            raise SystemExit(
                f"Another training loop appears to be running (pid={other_pid}). "
                f"If it's a stale lock, delete {lock_path}."
            )
        else:
            # Stale lock.
            try:
                lock_path.unlink()
            except Exception:
                pass

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{pid} {now}\n")
    except FileExistsError:
        raise SystemExit(f"Another training loop is already running (lock exists at {lock_path}).")

    def _cleanup() -> None:
        try:
            if lock_path.exists():
                txt = lock_path.read_text(encoding="utf-8").strip().split()
                if txt and int(txt[0]) == pid:
                    lock_path.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)


def _tail_lines(path: Path, max_lines: int) -> None:
    """Keep only the last max_lines lines of a UTF-8 text file.

    Important: this must NOT read the whole file into memory, because the dataset
    can grow very large between trims.
    """

    if max_lines <= 0 or not path.exists():
        return

    # Fast path for small files.
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    if size <= 0:
        return

    # Read from the end in blocks until we have enough newlines.
    block = 1024 * 1024  # 1MB
    nl_needed = int(max_lines)
    nl_count = 0
    chunks: list[bytes] = []
    start_pos = size

    with path.open("rb") as f:
        while start_pos > 0 and nl_count <= nl_needed:
            read_size = block if start_pos >= block else start_pos
            start_pos -= read_size
            f.seek(start_pos)
            data = f.read(read_size)
            chunks.append(data)
            nl_count += data.count(b"\n")

        # If we still don't have enough newlines, file is already <= max_lines.
        if nl_count <= nl_needed and start_pos == 0:
            return

        data = b"".join(reversed(chunks))

        # Find the byte offset for the last max_lines lines.
        # If the file ends with a trailing newline (normal for JSONL), ignore that
        # terminal newline for counting, otherwise we drop one extra line.
        scan = data[:-1] if data.endswith(b"\n") else data

        idx = len(scan)
        nl_seen = 0
        while idx > 0 and nl_seen < nl_needed:
            idx = scan.rfind(b"\n", 0, idx)
            if idx == -1:
                idx = 0
                break
            nl_seen += 1

        # idx is at the newline before the kept region; keep after it.
        keep = data[idx + 1 :] if idx < len(data) else b""

    # Write back truncated file.
    # Dataset lines are ASCII/UTF-8 JSON; keep bytes to avoid decode overhead.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as out:
        out.write(keep)
    try:
        tmp.replace(path)
    except Exception:
        # Fallback: best-effort overwrite.
        path.write_bytes(keep)
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--dataset", default="az_dataset.jsonl")
    ap.add_argument("--weights", default="az.pt")

    # Network size ("neurons") knobs.
    ap.add_argument("--net-channels", type=int, default=96, help="ResNet trunk channels (wider = stronger/slower)")
    ap.add_argument("--net-blocks", type=int, default=8, help="ResNet residual blocks (deeper = stronger/slower)")
    gamp = ap.add_mutually_exclusive_group()
    gamp.add_argument("--net-amp", action="store_true", help="Enable mixed precision on CUDA")
    gamp.add_argument("--no-net-amp", action="store_true", help="Disable mixed precision")

    # Optional evaluation: play a few deterministic games of (new weights) vs (previous weights)
    # to detect regressions. Disabled by default.
    ap.add_argument("--eval-games", type=int, default=0, help="If >0, evaluate new vs previous weights")
    ap.add_argument("--eval-every", type=int, default=1, help="Run eval every N iterations (if enabled)")
    ap.add_argument("--eval-every-games", type=int, default=0, help="Run eval every N self-play games (if enabled)")
    ap.add_argument("--eval-sims", type=int, default=500)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--eval-cpuct", type=float, default=1.5)
    ap.add_argument("--eval-max-plies", type=int, default=250)
    ap.add_argument("--eval-opening-plies", type=int, default=6, help="Random legal opening plies for more diverse eval")
    ap.add_argument("--eval-csv", default=str(Path("D:/NullMove/logs/az_eval.csv")))

    ap.add_argument(
        "--pgn-out",
        default=str(Path("D:/NullMove/Self_Play_Games/az_progress.pgn")),
        help="PGN file to append checkpoint self-play games",
    )
    ap.add_argument("--pgn-every", type=int, default=100, help="Only append a PGN once every N self-play games")
    ap.add_argument(
        "--pgn-every-per-worker",
        action="store_true",
        help="When --selfplay-workers > 1, apply --pgn-every to each worker's local game counter (legacy behavior)",
    )
    ap.add_argument("--pgn-max-games", type=int, default=0, help="Keep only last N PGN games (0 = keep all)")

    ap.add_argument(
        "--selfplay-live-log",
        default=str(Path("D:/NullMove/logs/selfplay_live.log")),
        help="Append per-game move lists here (safe across workers); empty disables",
    )

    ap.add_argument("--games-per-iter", type=int, default=20)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--c_puct", type=float, default=1.5)
    ap.add_argument("--max-plies", type=int, default=250)
    ap.add_argument("--temp-plies", type=int, default=30)
    ap.add_argument("--temp-init", type=float, default=1.25)
    ap.add_argument("--temp-final", type=float, default=0.5)
    ap.add_argument("--sample-p-after-temp", type=float, default=0.15)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    ap.add_argument("--clear-tree-every-move", action="store_true")
    gpm = ap.add_mutually_exclusive_group()
    gpm.add_argument("--print-moves", dest="print_moves", action="store_true", help="Print one line per self-play game with the move list")
    gpm.add_argument("--no-print-moves", dest="print_moves", action="store_false", help="Do not print per-game move lists")
    ap.set_defaults(print_moves=True)
    ap.add_argument("--draw-value", type=float, default=-0.2, help="Training target value to use for drawn games")

    # Optional early draw adjudication (disabled by default; can force uniform game lengths when net is untrained).
    ap.add_argument("--adjudicate-draw", action="store_true", help="Enable early draw adjudication")
    ap.add_argument("--draw-value-threshold", type=float, default=0.12)
    ap.add_argument("--draw-plies", type=int, default=24)
    ap.add_argument("--draw-min-ply", type=int, default=120)

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

    ap.add_argument(
        "--capture-bonus-scale",
        type=float,
        default=5.0,
        help="Scale factor for capture/promotion/check bonus in policy priors (helps early training)",
    )

    ap.add_argument(
        "--book-path",
        type=str,
        default="",
        help="Path to Polyglot opening book (.bin) for diverse opening play",
    )
    ap.add_argument(
        "--book-plies",
        type=int,
        default=0,
        help="Number of plies to use book moves (0 = disabled)",
    )

    ap.add_argument("--selfplay-workers", type=int, default=1, help="Run self-play games in multiple processes")
    ap.add_argument(
        "--streaming-push",
        action="store_true",
        help="Use streaming parallel mode: workers play in parallel but push data sequentially (safer, allows continuous training)",
    )

    ap.add_argument("--train-epochs", type=int, default=1)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)

    # Training target packing: larger top-k uses more RAM and a bit more compute.
    ap.add_argument("--policy-topk", type=int, default=128, help="Store top-k policy actions per position in RAM")

    # Larger replay buffer => more RAM use (and typically better learning stability).
    ap.add_argument("--replay-max", type=int, default=2000000, help="Keep only last N examples in dataset (0=keep all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Enforce single-instance by default.
    _acquire_single_instance_lock(Path(args.weights).resolve().parent / ".az_loop.lock")

    use_amp = bool(args.net_amp) and (not bool(args.no_net_amp))

    dataset = Path(args.dataset)
    weights = Path(args.weights)
    pgn_out = Path(args.pgn_out) if args.pgn_out else None
    live_log = Path(args.selfplay_live_log) if str(getattr(args, "selfplay_live_log", "")).strip() else None

    prev_weights = weights.with_suffix(weights.suffix + ".prev")
    baseline_weights = weights.with_suffix(".baseline.pt")  # Fixed baseline for tracking progress

    # Load persistent state (survives restarts)
    state = _load_state()
    global_iter = state.get("global_iteration", 0)
    game_base = state.get("global_games", 0)

    # Create baseline if it doesn't exist (first run after this feature)
    if not baseline_weights.exists() and weights.exists():
        print(f"[BASELINE] Creating fixed baseline from current weights: {baseline_weights}")
        shutil.copyfile(weights, baseline_weights)

    next_eval_game = int(getattr(args, "eval_every_games", 0))

    # Print training configuration at startup
    print("\n" + "=" * 60)
    print("NullMove AlphaZero Training Loop")
    print("=" * 60)
    print(f"Neural Network: {args.net_channels} channels, {args.net_blocks} residual blocks")
    print(f"Self-Play: {args.games_per_iter} games/iter, {args.sims} MCTS simulations/move")
    print(f"Training: {args.train_epochs} epochs, batch={args.train_batch}, lr={args.lr}")
    print(f"Capture Bonus: {args.capture_bonus_scale}x (helps AI learn captures)")
    if args.book_path and int(args.book_plies) > 0:
        print(f"Opening Book: {args.book_path} ({args.book_plies} plies)")
    else:
        print(f"Opening Book: disabled")
    print(f"Evaluation: {args.eval_games} games every {args.eval_every} iteration(s)")
    print(f"  - Every 50 iterations: 100-game eval vs fixed baseline")
    print(f"  - Baseline: {baseline_weights}")
    print(f"Resuming from: iteration {global_iter}, games {game_base}")
    print("=" * 60)
    print("\nWhat happens each iteration:")
    print("  1. SELF-PLAY: AI plays games against itself using MCTS")
    print("  2. TRAINING: Neural network learns from the games")
    print("     - Policy head: learns which moves to consider")
    print("     - Value head: learns to evaluate positions")
    print("  3. EVALUATION: Tests if new weights beat old weights")
    print("=" * 60 + "\n")

    for it in range(1, args.iters + 1):
        global_iter += 1  # Increment persistent counter
        
        print(f"\n{'='*50}")
        print(f"=== ITERATION {it}/{args.iters} (global #{global_iter}) ===")
        print(f"{'='*50}")

        # Snapshot current weights before self-play+train so we can evaluate later.
        if weights.exists():
            try:
                shutil.copyfile(weights, prev_weights)
            except Exception:
                pass

        print(f"\n[STEP 1/3] SELF-PLAY: Generating {args.games_per_iter} training games...")
        print(f"  - Each move: {args.sims} MCTS simulations")
        print(f"  - Capture bonus scale: {args.capture_bonus_scale}x")
        if bool(getattr(args, "streaming_push", False)):
            print(f"  - Mode: STREAMING (workers play parallel, push sequential)")
        
        if bool(getattr(args, "streaming_push", False)) and int(args.selfplay_workers) > 1:
            # Use streaming parallel mode: play parallel, push sequential
            generate_selfplay_streaming(
                out_path=dataset,
                weights_path=weights,
                games=args.games_per_iter,
                sims=args.sims,
                batch_size=args.batch_size,
                c_puct=args.c_puct,
                max_plies=args.max_plies,
                temp_plies=args.temp_plies,
                seed=int(args.seed) + it * 100000,
                pgn_out=pgn_out,
                pgn_every=args.pgn_every,
                game_index_base=game_base,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                temp_init=args.temp_init,
                temp_final=args.temp_final,
                sample_p_after_temp=args.sample_p_after_temp,
                clear_tree_every_move=bool(args.clear_tree_every_move),
                adjudicate_draw=bool(args.adjudicate_draw),
                draw_value_threshold=float(args.draw_value_threshold),
                draw_plies=int(args.draw_plies),
                draw_min_ply=int(args.draw_min_ply),
                resign_threshold=None,
                resign_plies=8,
                draw_value=float(args.draw_value),
                repeat_penalty=float(args.repeat_penalty),
                repeat_window=int(args.repeat_window),
                workers=int(args.selfplay_workers),
                net_channels=int(args.net_channels),
                net_blocks=int(args.net_blocks),
                net_amp=use_amp,
                live_log=live_log,
                capture_bonus_scale=float(args.capture_bonus_scale),
                book_path=(Path(args.book_path) if args.book_path else None),
                book_plies=int(args.book_plies),
            )
        else:
            # Standard mode (batched parallel or single-worker)
            generate_selfplay(
                out_path=dataset,
                weights_path=weights,
                games=args.games_per_iter,
                sims=args.sims,
                batch_size=args.batch_size,
                c_puct=args.c_puct,
                max_plies=args.max_plies,
                temp_plies=args.temp_plies,
                seed=int(args.seed) + it * 100000,
                pgn_out=pgn_out,
                pgn_every=args.pgn_every,
                # Default to global cadence so `--pgn-every 100` means every 100 total games, not per-worker.
                pgn_every_per_worker=bool(getattr(args, "pgn_every_per_worker", False)),
                pgn_max_games=args.pgn_max_games,
                game_index_base=game_base,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_eps=args.dirichlet_eps,
                temp_init=args.temp_init,
                temp_final=args.temp_final,
                sample_p_after_temp=args.sample_p_after_temp,
                clear_tree_every_move=bool(args.clear_tree_every_move),
                print_moves=bool(args.print_moves),
                draw_value=float(args.draw_value),
                repeat_penalty=float(args.repeat_penalty),
                repeat_window=int(args.repeat_window),
                adjudicate_draw=bool(args.adjudicate_draw),
                draw_value_threshold=float(args.draw_value_threshold),
                draw_plies=int(args.draw_plies),
                draw_min_ply=int(args.draw_min_ply),
                workers=int(args.selfplay_workers),
                net_channels=int(args.net_channels),
                net_blocks=int(args.net_blocks),
                net_amp=use_amp,
                live_log=live_log,
                live_log_master=live_log,
                capture_bonus_scale=float(args.capture_bonus_scale),
                book_path=(Path(args.book_path) if args.book_path else None),
                book_plies=int(args.book_plies),
            )

        game_base += int(args.games_per_iter)

        _tail_lines(dataset, args.replay_max)

        print(f"\n[STEP 2/3] TRAINING: Learning from self-play games...")
        print(f"  - Epochs: {args.train_epochs} passes through the dataset")
        print(f"  - Analyzing: Move choices (policy) and position evaluations (value)")
        
        train_once(
            data_path=dataset,
            in_weights=weights,
            out_weights=weights,
            epochs=args.train_epochs,
            batch=args.train_batch,
            lr=args.lr,
            policy_topk=int(args.policy_topk),
            net_channels=int(args.net_channels),
            net_blocks=int(args.net_blocks),
            net_amp=use_amp,
        )

        # Optional strength sanity-check: new vs baseline (fixed reference point)
        do_eval = False
        if int(args.eval_games) > 0 and baseline_weights.exists():
            every_games = int(getattr(args, "eval_every_games", 0))
            if every_games > 0:
                if next_eval_game > 0 and int(game_base) >= int(next_eval_game):
                    do_eval = True
                    # Catch up schedule without running multiple evals per iteration.
                    while next_eval_game > 0 and int(game_base) >= int(next_eval_game):
                        next_eval_game += every_games
            else:
                if int(args.eval_every) > 0 and (global_iter % int(args.eval_every) == 0):
                    do_eval = True

        if do_eval:
            # Every 50 global iterations: run 100 games for better signal
            is_big_eval = (global_iter % 50 == 0)
            eval_game_count = 100 if is_big_eval else int(args.eval_games)
            
            print(f"\n[STEP 3/3] EVALUATION: Testing new weights vs BASELINE...")
            if is_big_eval:
                print(f"  *** BIG EVAL (iteration {global_iter} is multiple of 50) ***")
            print(f"  - Playing {eval_game_count} games with {args.eval_sims} sims/move")
            print(f"  - Baseline: {baseline_weights.name}")
            
            # Save decisive eval games to a dedicated PGN file
            eval_pgn = Path("D:/NullMove/Self_Play_Games/az_eval_wins.pgn")
            eval_pgn_promos = Path("D:/NullMove/Self_Play_Games/az_eval_promos.pgn")
            
            r = eval_weights(
                a_path=weights,
                b_path=baseline_weights,  # Use fixed baseline, not prev
                games=eval_game_count,
                sims=int(args.eval_sims),
                batch_size=int(args.eval_batch_size),
                c_puct=float(args.eval_cpuct),
                max_plies=int(args.eval_max_plies),
                opening_plies=int(args.eval_opening_plies),
                seed=int(args.seed) + global_iter * 99991,
                csv_out=(Path(args.eval_csv) if args.eval_csv else None),
                pgn_out=eval_pgn,
                pgn_promos_out=eval_pgn_promos,
                net_channels=int(args.net_channels),
                net_blocks=int(args.net_blocks),
                net_amp=use_amp,
            )
            score = (r.a_wins + 0.5 * r.draws) / r.games
            print(f"eval vs BASELINE: A_wins={r.a_wins} B_wins={r.b_wins} draws={r.draws} (games={r.games}) score={score:.3f}")

        # Save persistent state after each iteration
        _save_state({"global_iteration": global_iter, "global_games": game_base})

        # Check for graceful shutdown request at end of each iteration
        if _check_shutdown():
            print(f"\n[SHUTDOWN] Graceful shutdown requested - exiting after iteration {global_iter}")
            print(f"[SHUTDOWN] Weights saved to {weights}")
            print(f"[SHUTDOWN] State saved: iteration={global_iter}, games={game_base}")
            break

    print("done")


if __name__ == "__main__":
    main()
