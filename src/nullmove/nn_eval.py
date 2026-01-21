from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess


@dataclass
class NNConfig:
    enabled: bool = False
    weights_path: str = "nn.pt"


class NNEvaluator:
    """Optional Torch-based evaluator.

    This is intentionally lightweight scaffolding.
    - If torch isn't installed, this raises at construction.
    - The model is a tiny MLP over 12x64 piece planes.
    """

    def __init__(self, weights_path: str) -> None:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.model = MLP().to(self.device)
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"weights not found: {weights_path}")

        state = torch.load(str(p), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def evaluate(self, board: chess.Board) -> int:
        # 12 planes: WP WN WB WR WQ WK BP BN BB BR BQ BK
        planes = [[0.0] * 64 for _ in range(12)]

        def fill(idx: int, piece_type: chess.PieceType, color: chess.Color) -> None:
            for sq in board.pieces(piece_type, color):
                planes[idx][sq] = 1.0

        fill(0, chess.PAWN, chess.WHITE)
        fill(1, chess.KNIGHT, chess.WHITE)
        fill(2, chess.BISHOP, chess.WHITE)
        fill(3, chess.ROOK, chess.WHITE)
        fill(4, chess.QUEEN, chess.WHITE)
        fill(5, chess.KING, chess.WHITE)
        fill(6, chess.PAWN, chess.BLACK)
        fill(7, chess.KNIGHT, chess.BLACK)
        fill(8, chess.BISHOP, chess.BLACK)
        fill(9, chess.ROOK, chess.BLACK)
        fill(10, chess.QUEEN, chess.BLACK)
        fill(11, chess.KING, chess.BLACK)

        x = [v for pl in planes for v in pl]
        t = self.torch.tensor([x], dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            y = self.model(t).item()

        # y is "from white" in roughly centipawns after training; convert to side-to-move.
        cp = int(round(100.0 * y))
        return cp if board.turn == chess.WHITE else -cp


def maybe_load_nn(config: NNConfig) -> Optional[NNEvaluator]:
    if not config.enabled:
        return None

    try:
        return NNEvaluator(config.weights_path)
    except ModuleNotFoundError:
        # torch not installed
        return None
    except FileNotFoundError:
        return None
