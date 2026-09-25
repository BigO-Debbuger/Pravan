"""
ml/models/cnn_baseline.py
--------------------------
EXP-005: Minimal Fully Convolutional Network for spatial anomaly prediction.

ARCHITECTURE
------------
Input : (B, 4, 125, 121)   — 3 lag anomaly channels + 1 climatology channel
Output: (B, 1, 125, 121)   — predicted anomaly map for month t

Layers:
    Conv2d(4  → 16, kernel=3, pad=1) + ReLU
    Conv2d(16 → 32, kernel=3, pad=1) + ReLU
    Conv2d(32 → 16, kernel=3, pad=1) + ReLU
    Conv2d(16 →  1, kernel=1)        (linear — regression output)

Design decisions:
  - padding=1 on all 3×3 convolutions → spatial dims 125×121 are preserved.
  - No pooling / stride / upsampling → output is pixel-aligned with input.
  - Final kernel=1 conv → pointwise projection, no spatial mixing at last step.
  - No BatchNorm in v1 (consistent with small batch_size=4 where BN is noisy).
  - No bias on final layer not needed (mean-zero target after normalization).
  - Parameter count: ~14,000 — deliberately tiny, VRAM budget: < 50 MB.

VRAM ESTIMATE (batch_size=4, float32)
--------------------------------------
  Weights      :  ~14k params × 4 bytes          ≈ 0.056 MB
  One batch X  :  4 × 4 × 125 × 121 × 4 bytes   ≈ 0.97  MB
  Activations  :  4 layers × ~4×H×W×max_chan     ≈ 15    MB (generous estimate)
  Gradients    :  same order as activations       ≈ 15    MB
  ─────────────────────────────────────────────────────────
  Total peak   :                                  ≈ 30-50 MB   << 4 GB ceiling
"""

import torch
import torch.nn as nn

__all__ = ["CNNBaseline", "count_parameters"]


class CNNBaseline(nn.Module):
    """
    Minimal fully convolutional network for EXP-005.

    Parameters
    ----------
    in_channels  : int — number of input channels (default 4: 3 lag + 1 climatology)
    base_filters : int — number of filters in first conv layer (default 16)
    """

    def __init__(self, in_channels: int = 4, base_filters: int = 16):
        super().__init__()
        f = base_filters  # 16 by default

        self.encoder = nn.Sequential(
            # Block 1: in_channels → f (16)
            nn.Conv2d(in_channels, f,      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # Block 2: f (16) → 2f (32)
            nn.Conv2d(f,          f * 2,   kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            # Block 3: 2f (32) → f (16)
            nn.Conv2d(f * 2,      f,       kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Output projection: f (16) → 1, pointwise — no spatial mixing
        self.output_conv = nn.Conv2d(f, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 4, 125, 121)  — normalized input channels
        returns : (B, 1, 125, 121) — normalized predicted anomaly
        """
        feat = self.encoder(x)          # (B, 16, 125, 121)
        out  = self.output_conv(feat)   # (B,  1, 125, 121)
        return out


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape verification (CPU)
    model = CNNBaseline(in_channels=4, base_filters=16)
    n_params = count_parameters(model)
    print(f"CNNBaseline parameters : {n_params:,}")

    # Dummy forward pass
    B, C, H, W = 4, 4, 125, 121
    x_dummy = torch.randn(B, C, H, W)
    y_dummy = model(x_dummy)
    print(f"Input  shape : {tuple(x_dummy.shape)}")
    print(f"Output shape : {tuple(y_dummy.shape)}")
    assert y_dummy.shape == (B, 1, H, W), \
        f"Output shape mismatch: expected ({B}, 1, {H}, {W}), got {tuple(y_dummy.shape)}"
    print("Shape check PASSED.")
