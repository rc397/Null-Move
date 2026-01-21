from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Tuple

import chess

from .az_features import ACTION_SIZE, board_to_planes

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class AZConfig:
    weights_path: str = "az.pt"
    device: str | None = None  # None = auto
    channels: int = 96
    blocks: int = 8
    use_amp: bool = True


class AZNet:
    """Tiny policy+value net for AlphaZero-style training.

    Outputs:
    - policy logits over ACTION_SIZE
    - value in [-1, 1] from side-to-move perspective
    - (training only) optional material head in [-1, 1] from side-to-move perspective
    """

    def __init__(
        self,
        weights_path: str | None = None,
        device: str | None = None,
        *,
        channels: int = 96,
        blocks: int = 8,
        use_amp: bool = True,
    ) -> None:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore

        self.torch = torch

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_amp = bool(use_amp) and (self.device.type == "cuda")
        if self.device.type == "cuda":
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass

        class ResBlock(nn.Module):
            def __init__(self, channels: int) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(channels)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                return self.relu(out + x)

        class Net(nn.Module):
            def __init__(self, channels: int, blocks: int) -> None:
                super().__init__()
                self.in_conv = nn.Conv2d(13, channels, kernel_size=3, padding=1, bias=False)
                self.in_bn = nn.BatchNorm2d(channels)
                self.relu = nn.ReLU(inplace=True)
                self.res = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

                # Policy head
                self.p_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
                self.p_bn = nn.BatchNorm2d(32)
                self.p_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)

                # Value head
                self.v_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
                self.v_bn = nn.BatchNorm2d(32)
                self.v_fc1 = nn.Linear(32 * 8 * 8, 256)
                self.v_fc2 = nn.Linear(256, 1)
                self.tanh = nn.Tanh()

                # Material head (auxiliary training target)
                self.m_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
                self.m_bn = nn.BatchNorm2d(32)
                self.m_fc1 = nn.Linear(32 * 8 * 8, 64)
                self.m_fc2 = nn.Linear(64, 1)

            def forward(self, x):
                # x: [B, 13, 8, 8]
                h = self.relu(self.in_bn(self.in_conv(x)))
                h = self.res(h)

                p = self.relu(self.p_bn(self.p_conv(h)))
                p = p.view(p.size(0), -1)
                logits = self.p_fc(p)

                v = self.relu(self.v_bn(self.v_conv(h)))
                v = v.view(v.size(0), -1)
                v = self.relu(self.v_fc1(v))
                v = self.tanh(self.v_fc2(v))

                m = self.relu(self.m_bn(self.m_conv(h)))
                m = m.view(m.size(0), -1)
                m = self.relu(self.m_fc1(m))
                m = self.tanh(self.m_fc2(m))

                return logits, v, m

        ch = max(32, int(channels))
        bl = max(1, int(blocks))
        self.model = Net(channels=ch, blocks=bl).to(self.device)

        if weights_path is not None:
            p = Path(weights_path)
            if p.exists():
                state = torch.load(str(p), map_location=self.device)
                try:
                    self.model.load_state_dict(state, strict=False)
                except Exception:
                    # If the architecture changed (e.g., upgrading from MLP -> ResNet),
                    # start from fresh weights instead of crashing.
                    pass

        self.model.eval()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.model.state_dict(), path)

    def infer(self, board: chess.Board) -> Tuple["torch.Tensor", float]:
        """Returns (policy_logits[1,ACTION_SIZE], value_float)."""
        x = board_to_planes(board)
        t_cpu = self.torch.tensor([x], dtype=self.torch.float32)
        if self.device.type == "cuda":
            try:
                t_cpu = t_cpu.pin_memory()
            except Exception:
                pass
        t = t_cpu.to(self.device, non_blocking=True).view(1, 13, 8, 8)
        with self.torch.no_grad():
            if self.use_amp:
                with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                    out = self.model(t)
            else:
                out = self.model(t)
        if isinstance(out, tuple) and len(out) == 3:
            logits, v, _m = out
        else:
            logits, v = out
        return logits[0], float(v.item())

    def infer_many(self, boards: Iterable[chess.Board]) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Returns (policy_logits[B,ACTION_SIZE], values[B])."""
        xs = [board_to_planes(b) for b in boards]
        if not xs:
            t = self.torch.empty((0, 13, 8, 8), dtype=self.torch.float32, device=self.device)
            return (
                self.torch.empty((0, ACTION_SIZE), dtype=self.torch.float32, device=self.device),
                self.torch.empty((0,), dtype=self.torch.float32, device=self.device),
            )
        t_cpu = self.torch.tensor(xs, dtype=self.torch.float32)
        if self.device.type == "cuda":
            try:
                t_cpu = t_cpu.pin_memory()
            except Exception:
                pass
        t = t_cpu.to(self.device, non_blocking=True).view(len(xs), 13, 8, 8)
        with self.torch.no_grad():
            if self.use_amp:
                with self.torch.autocast(device_type="cuda", dtype=self.torch.float16):
                    out = self.model(t)
            else:
                out = self.model(t)
        if isinstance(out, tuple) and len(out) == 3:
            logits, v, _m = out
        else:
            logits, v = out
        return logits, v.squeeze(1)
