from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess


def fen_to_tensor(board: chess.Board) -> list[float]:
    # 12x64 planes
    x = [0.0] * (12 * 64)

    def set_plane(plane: int, piece_type: chess.PieceType, color: chess.Color) -> None:
        for sq in board.pieces(piece_type, color):
            x[plane * 64 + sq] = 1.0

    set_plane(0, chess.PAWN, chess.WHITE)
    set_plane(1, chess.KNIGHT, chess.WHITE)
    set_plane(2, chess.BISHOP, chess.WHITE)
    set_plane(3, chess.ROOK, chess.WHITE)
    set_plane(4, chess.QUEEN, chess.WHITE)
    set_plane(5, chess.KING, chess.WHITE)

    set_plane(6, chess.PAWN, chess.BLACK)
    set_plane(7, chess.KNIGHT, chess.BLACK)
    set_plane(8, chess.BISHOP, chess.BLACK)
    set_plane(9, chess.ROOK, chess.BLACK)
    set_plane(10, chess.QUEEN, chess.BLACK)
    set_plane(11, chess.KING, chess.BLACK)

    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.jsonl")
    ap.add_argument("--out", default="nn.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs: list[list[float]] = []
    ys: list[float] = []

    with Path(args.data).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            board = chess.Board(obj["fen"])
            # Train on white-centric score (cp from side-to-move -> convert)
            cp_stm = int(obj["cp"])
            cp_white = cp_stm if board.turn == chess.WHITE else -cp_stm
            xs.append(fen_to_tensor(board))
            ys.append(cp_white / 100.0)  # scale

    X = torch.tensor(xs, dtype=torch.float32, device=device)
    Y = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12 * 64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n = X.shape[0]
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            xb = X[idx]
            yb = Y[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch}: loss {total:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
