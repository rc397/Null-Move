from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path
import shutil
import time

# Allow running as a script without requiring `pip install -e .`
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from az_selfplay import generate_selfplay  # noqa: E402
from az_train import train_once  # noqa: E402
from az_eval import eval_weights  # noqa: E402


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

    ap.add_argument("--selfplay-workers", type=int, default=1, help="Run self-play games in multiple processes")

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

    game_base = 0

    next_eval_game = int(getattr(args, "eval_every_games", 0))

    for it in range(1, args.iters + 1):
        print(f"\n=== iteration {it}/{args.iters} ===")

        # Snapshot current weights before self-play+train so we can evaluate later.
        if weights.exists():
            try:
                shutil.copyfile(weights, prev_weights)
            except Exception:
                pass

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
            pgn_every_per_worker=(int(args.selfplay_workers) > 1),
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
        )

        game_base += int(args.games_per_iter)

        _tail_lines(dataset, args.replay_max)

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

        # Optional strength sanity-check: new vs previous.
        do_eval = False
        if int(args.eval_games) > 0 and prev_weights.exists():
            every_games = int(getattr(args, "eval_every_games", 0))
            if every_games > 0:
                if next_eval_game > 0 and int(game_base) >= int(next_eval_game):
                    do_eval = True
                    # Catch up schedule without running multiple evals per iteration.
                    while next_eval_game > 0 and int(game_base) >= int(next_eval_game):
                        next_eval_game += every_games
            else:
                if int(args.eval_every) > 0 and (it % int(args.eval_every) == 0):
                    do_eval = True

        if do_eval:
            r = eval_weights(
                a_path=weights,
                b_path=prev_weights,
                games=int(args.eval_games),
                sims=int(args.eval_sims),
                batch_size=int(args.eval_batch_size),
                c_puct=float(args.eval_cpuct),
                max_plies=int(args.eval_max_plies),
                opening_plies=int(args.eval_opening_plies),
                seed=int(args.seed) + it * 99991,
                csv_out=(Path(args.eval_csv) if args.eval_csv else None),
                net_channels=int(args.net_channels),
                net_blocks=int(args.net_blocks),
                net_amp=use_amp,
            )
            print(f"eval vs prev: A_wins={r.a_wins} B_wins={r.b_wins} draws={r.draws} (games={r.games})")

    print("done")


if __name__ == "__main__":
    main()
