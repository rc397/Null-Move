# NullMove

A small, practical chess engine baseline that speaks **UCI** so you can run it in Arena / Cute Chess / BanksiaGUI.

## Quickstart (Windows)

1. Create venv + install deps (already done if you used the VS Code tasks):

   - `D:/NullMove/.venv/Scripts/python.exe -m pip install -e .`

2. Run the engine in a terminal:

   - `D:/NullMove/.venv/Scripts/python.exe -m nullmove.uci`

3. Or install the `nullmove` console script:

   - `D:/NullMove/.venv/Scripts/python.exe -m pip install -e .`
   - `nullmove`

## What you get

- UCI loop: `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`
- Iterative deepening + alpha-beta (negamax)
- Simple evaluation (material + piece-square tables)
- Quiescence search
- Transposition table + basic move ordering

## Next steps (high value)

- Add stronger move ordering: history heuristic + better SEE
- Add endgame scaling and pawn structure evaluation
- Plug in a small neural eval (optional) once the baseline is stable

## NN weights without copying other engines

If you want to keep it "original": you can generate training labels using **NullMove itself** via self-play + deeper search.

- Generate dataset (no Stockfish required):
   - `D:/NullMove/.venv/Scripts/python.exe train/generate_dataset_nullmove.py --out dataset.jsonl --games 200 --play-depth 2 --label-depth 7 --max-positions 50000`
- Train:
   - `D:/NullMove/.venv/Scripts/python.exe train/train_nn.py --data dataset.jsonl --out nn.pt --epochs 5 --batch 1024`

Then enable it in UCI:
- `setoption name UseNN value true`
- `setoption name NNWeights value nn.pt`

## AlphaZero-style loop (MCTS + policy/value net)

This is a minimal, "original" AlphaZero-like pipeline:
- Self-play uses **MCTS (PUCT)** guided by a policy/value net.
- Training uses (state, MCTS policy target, final game outcome).

## Auto-start training on Windows

You can run training automatically when you boot/login using Windows Task Scheduler.

- Start training manually:
   - `powershell -ExecutionPolicy Bypass -File D:/NullMove/scripts/start_training.ps1`

- Register an auto-start task (runs on logon):
   - `powershell -ExecutionPolicy Bypass -File D:/NullMove/scripts/register_startup_training_task.ps1`

- Remove the task:
   - `powershell -ExecutionPolicy Bypass -File D:/NullMove/scripts/register_startup_training_task.ps1 -Remove`

Outputs:
- Training dataset: `D:/NullMove/az_dataset.jsonl`
- Weights: `D:/NullMove/az.pt`
- Checkpoint PGNs (one every 100 games): `D:/NullMove/Self_Play_Games/az_progress.pgn`

Resource notes:
- The scripts can bias CPU usage by limiting threads and setting process affinity, but Windows does not provide a clean "give exactly 70% CPU and 24GB RAM" guarantee.
- GPU usage depends on how much work is queued (higher `--batch-size` and `--sims` generally increases GPU load).

## Daily schedule (7am–9pm)

You can run training on a schedule (start at 07:00, stop at 21:00) using Task Scheduler.

- Create scheduled tasks (tries to use `schtasks.exe`):
   - `powershell -ExecutionPolicy Bypass -File D:/NullMove/scripts/schedule_training.ps1 -StartTime 07:00 -StopTime 21:00`

- Remove tasks:
   - `powershell -ExecutionPolicy Bypass -File D:/NullMove/scripts/schedule_training.ps1 -Remove`

Notes:
- "Boot itself at 7am" from a powered-off state usually requires a BIOS/UEFI **RTC wake** setting.
- Waking from sleep requires enabling Windows wake timers and checking "Wake the computer to run this task" in Task Scheduler.
1) Self-play to generate training data:
- `D:/NullMove/.venv/Scripts/python.exe train/az_selfplay.py --out az_dataset.jsonl --weights az.pt --games 20 --sims 200`

2) Train the network (uses CUDA if available):
- `D:/NullMove/.venv/Scripts/python.exe train/az_train.py --data az_dataset.jsonl --in az.pt --out az.pt --epochs 1 --batch 256`

Repeat steps (1) and (2) to improve over time.

Optional (useful, but not required): strength evaluation
- The loop runner [train/az_loop.py](train/az_loop.py) can optionally evaluate "new weights vs previous weights" every iteration (or every N iterations).
- This is noisy, but it helps catch regressions and confirm you’re trending upward.
- Enable by passing `--eval-games` (example: `--eval-games 20 --eval-sims 100`).
- Results append to `D:/NullMove/logs/az_eval.csv`.

## Self-play / tournaments

- Self-play PGN generator:
   - `D:/NullMove/.venv/Scripts/python.exe scripts/selfplay.py --games 10 --depth 6 --out selfplay.pgn`

- Tiny round-robin tournament harness:
   - `D:/NullMove/.venv/Scripts/python.exe scripts/tournament.py --engine NullMove D:/NullMove/.venv/Scripts/python.exe -m nullmove.uci --engine NullMove2 D:/NullMove/.venv/Scripts/python.exe -m nullmove.uci --games 4 --depth 6`

## Optional C++ speedup (eval DLL)

This repo includes a tiny C++ DLL that accelerates the evaluation function. It’s optional: if `bin/nullmove_eval.dll` exists, the engine will use it automatically.

- Build (requires MSVC Build Tools / Developer PowerShell):
   - `powershell -NoProfile -ExecutionPolicy Bypass -File D:/NullMove/scripts/build_cpp_eval.ps1`

## Optional neural evaluation (scaffold)

This is an optional, lightweight path to train a small MLP eval from an external UCI evaluator (typically Stockfish). It’s meant for experimentation, not for “AlphaZero-scale” training.

- Install PyTorch (choose the correct CUDA build for your GPU):
   - `D:/NullMove/.venv/Scripts/python.exe -m pip install torch`

- Generate a tiny dataset (requires `stockfish` on PATH or an explicit command):
   - `D:/NullMove/.venv/Scripts/python.exe train/generate_dataset.py --stockfish stockfish --out dataset.jsonl --games 200 --plies 40 --depth 10`

- Train weights:
   - `D:/NullMove/.venv/Scripts/python.exe train/train_nn.py --data dataset.jsonl --out nn.pt --epochs 3`

- Enable in UCI:
   - `setoption name UseNN value true`
   - `setoption name NNWeights value nn.pt`
