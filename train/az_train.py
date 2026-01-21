from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess

# Allow running as a script without requiring `pip install -e .`
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nullmove.az_features import ACTION_SIZE  # noqa: E402
from nullmove.az_features import board_to_planes  # noqa: E402
from nullmove.az_net import AZNet  # noqa: E402


def train_once(
    data_path: Path,
    in_weights: Path,
    out_weights: Path,
    epochs: int,
    batch: int,
    lr: float,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    policy_topk: int = 64,
    material_weight: float = 0.10,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    net_channels: int = 96,
    net_blocks: int = 8,
    net_amp: bool = True,
) -> None:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    net = AZNet(
        weights_path=str(in_weights),
        channels=int(net_channels),
        blocks=int(net_blocks),
        use_amp=bool(net_amp),
    )
    device = net.device
    net.model.train()

    amp_enabled = bool(use_amp) and (device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        autocast = lambda: torch.amp.autocast("cuda", enabled=amp_enabled)  # noqa: E731
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        autocast = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)  # noqa: E731

    # Load dataset in a compact form to avoid massive Python list overhead.
    # X/Z live on CPU; batches are moved to GPU for training.
    with data_path.open("r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    if n <= 0:
        print("no training data")
        return

    X_cpu = torch.empty((n, 13, 8, 8), dtype=torch.float32)
    Z_cpu = torch.empty((n, 1), dtype=torch.float32)
    M_cpu = torch.empty((n, 1), dtype=torch.float32)

    # Fixed Top-K packing for sparse policy targets (GPU-friendly).
    k = max(1, int(policy_topk))
    A_cpu = torch.full((n, k), 0, dtype=torch.int64)
    P_cpu = torch.zeros((n, k), dtype=torch.float32)

    with data_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            board = chess.Board(obj["fen"])
            x = board_to_planes(board)
            X_cpu[i].view(-1).copy_(torch.tensor(x, dtype=torch.float32))
            Z_cpu[i, 0] = float(obj["z"])
            if "m" in obj:
                M_cpu[i, 0] = float(obj.get("m", 0.0))
            else:
                M_cpu[i, 0] = float(obj.get("material", 0.0))

            pol = obj.get("policy") or []
            # policy is [[action, prob], ...] over (mostly) legal moves; keep top-k by prob.
            items: list[tuple[int, float]] = []
            for a, p in pol:
                a_i = int(a)
                p_f = float(p)
                if 0 <= a_i < ACTION_SIZE and p_f > 0.0:
                    items.append((a_i, p_f))
            if items:
                items.sort(key=lambda t: t[1], reverse=True)
                items = items[:k]
                s = float(sum(p for _a, p in items))
                if s > 0.0:
                    for j, (a_i, p_f) in enumerate(items):
                        A_cpu[i, j] = int(a_i)
                        P_cpu[i, j] = float(p_f) / s

    if device.type == "cuda":
        try:
            X_cpu = X_cpu.pin_memory()
            Z_cpu = Z_cpu.pin_memory()
            M_cpu = M_cpu.pin_memory()
            A_cpu = A_cpu.pin_memory()
            P_cpu = P_cpu.pin_memory()
        except Exception:
            pass

    opt = torch.optim.Adam(net.model.parameters(), lr=lr, weight_decay=float(weight_decay))
    log_softmax = nn.LogSoftmax(dim=1)
    mse = nn.MSELoss()

    n = int(X_cpu.shape[0])
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        total_policy = 0.0
        total_value = 0.0

        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            xb = X_cpu[idx].to(device, non_blocking=True)
            zb = Z_cpu[idx].to(device, non_blocking=True)
            mb = M_cpu[idx].to(device, non_blocking=True)
            ab = A_cpu[idx].to(device, non_blocking=True)
            pb = P_cpu[idx].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast():
                out = net.model(xb)
                if isinstance(out, tuple) and len(out) == 3:
                    logits, v, m_pred = out
                else:
                    logits, v = out
                    m_pred = None
                lp = log_softmax(logits)

                # Top-K sparse policy loss: -sum_j p_j * log pi(a_j)
                gathered = lp.gather(1, ab)  # [B,K]
                pol_loss_t = -(pb * gathered).sum(dim=1).mean()

                val_loss = mse(v, zb)
                if m_pred is None:
                    loss = policy_weight * pol_loss_t + value_weight * val_loss
                else:
                    mat_loss = mse(m_pred, mb)
                    loss = policy_weight * pol_loss_t + value_weight * val_loss + float(material_weight) * mat_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_policy += float(pol_loss_t.item())
            total_value += float(val_loss.item())

        print(f"epoch {epoch}: policy {total_policy:.4f} value {total_value:.4f}")

    torch.save(net.model.state_dict(), str(out_weights))
    print(f"wrote {out_weights}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="az_dataset.jsonl")
    ap.add_argument("--in", dest="in_weights", default="az.pt")
    ap.add_argument("--out", default="az.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--policy_weight", type=float, default=1.0)
    ap.add_argument("--value_weight", type=float, default=1.0)
    ap.add_argument("--policy-topk", type=int, default=64, help="Store only top-k policy actions per position")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = ap.parse_args()

    use_amp = bool(args.amp) and (not bool(args.no_amp))

    train_once(
        data_path=Path(args.data),
        in_weights=Path(args.in_weights),
        out_weights=Path(args.out),
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        policy_topk=int(args.policy_topk),
        weight_decay=float(args.weight_decay),
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
