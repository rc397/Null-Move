from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nn.pt", help="Output path for initialized weights")
    args = ap.parse_args()

    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

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

    model = MLP()
    out = Path(args.out)
    torch.save(model.state_dict(), str(out))
    print(f"wrote initialized weights: {out}")


if __name__ == "__main__":
    main()
