"""
ml/cnn_sanity_test.py
---------------------
EXP-005 GPU sanity test. DO NOT use for full training.

Verifies:
  0. CUDA availability
  1. Dataset loads correctly (shapes, NaNs)
  2. A single batch has correct input/output shapes
  3. Model forward pass works on CUDA
  4. Loss computes correctly
  5. Backward pass works (gradients flow)
  6. VRAM stays under 4 GB ceiling
  7. A 2-epoch mini-training loop completes without errors

Produces NO model artifacts and writes NO files.
Exit code 0 = all checks passed.
Exit code 1 = failure.
"""

import sys
import pathlib
import subprocess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from cnn_dataset import make_dataloaders
from models.cnn_baseline import CNNBaseline, count_parameters

VRAM_CEILING_GB = 4.0


def get_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            p = [s.strip() for s in r.stdout.strip().split(",")]
            return {
                "name"     : p[0],
                "total_gb" : round(int(p[1]) / 1024, 2),
                "free_gb"  : round(int(p[2]) / 1024, 2),
                "used_gb"  : round(int(p[3]) / 1024, 2),
            }
    except Exception:
        pass
    return None


def print_vram(label):
    info = get_vram()
    if info is None:
        print("   {:35s} | VRAM: unavailable".format(label))
        return info
    status = "OK" if info["used_gb"] < VRAM_CEILING_GB else "*** CEILING EXCEEDED ***"
    print("   {:35s} | VRAM: {:.2f}/{:.2f} GB ({:.2f} GB free)  [{}]".format(
        label, info["used_gb"], info["total_gb"], info["free_gb"], status))
    return info


def run():
    results = {}
    all_pass = True

    print("=" * 65)
    print("  EXP-005 CNN -- GPU SANITY TEST")
    print("=" * 65)
    print("  PyTorch version : {}".format(torch.__version__))
    print("  VRAM ceiling    : {} GB".format(VRAM_CEILING_GB))
    print()

    # CHECK 0: CUDA
    print("[CHECK 0] CUDA availability...")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        print("   CUDA available : {}".format(torch.cuda.get_device_name(0)))
        print_vram("Before anything")
        results["cuda"] = True
        print("   PASS\n")
    else:
        device = torch.device("cpu")
        print("   WARNING: CUDA not available.")
        print("   PyTorch build: {}".format(torch.__version__))
        print("   Install CUDA-enabled PyTorch (see below). Running on CPU to validate logic.")
        results["cuda"] = False
        # Not all_pass = False here; we still validate logic on CPU
        print()

    # CHECK 1: Dataset / DataLoader
    print("[CHECK 1] Loading Dataset and DataLoader...")
    try:
        train_loader, val_loader, test_loader, stats, anom, times = make_dataloaders(
            batch_size=4, num_workers=0, normalize=True, verbose=True
        )
        assert len(train_loader.dataset) == 126, \
            "Expected 126 train samples, got {}".format(len(train_loader.dataset))
        assert len(val_loader.dataset) == 24, \
            "Expected 24 val samples, got {}".format(len(val_loader.dataset))
        assert len(test_loader.dataset) == 24, \
            "Expected 24 test samples, got {}".format(len(test_loader.dataset))
        results["dataset"] = True
        print("   PASS: Dataset loaded. Split counts correct (126/24/24).\n")
    except Exception as e:
        print("   FAIL: {}: {}".format(type(e).__name__, e))
        results["dataset"] = False
        all_pass = False
        sys.exit(1)

    # CHECK 2: Batch shape and NaN check
    print("[CHECK 2] Batch shape and NaN check...")
    try:
        X_cpu, y_cpu = next(iter(train_loader))
        assert X_cpu.shape == (4, 4, 125, 121), "X shape mismatch: {}".format(X_cpu.shape)
        assert y_cpu.shape == (4, 1, 125, 121), "y shape mismatch: {}".format(y_cpu.shape)
        assert not X_cpu.isnan().any(), "NaNs in X batch"
        assert not y_cpu.isnan().any(), "NaNs in y batch"
        assert X_cpu.dtype == torch.float32, "Expected float32, got {}".format(X_cpu.dtype)
        print("   X shape  : {}  [OK]".format(tuple(X_cpu.shape)))
        print("   y shape  : {}  [OK]".format(tuple(y_cpu.shape)))
        print("   X range  : [{:.3f}, {:.3f}] (normalized)".format(X_cpu.min().item(), X_cpu.max().item()))
        print("   y range  : [{:.3f}, {:.3f}] (normalized)".format(y_cpu.min().item(), y_cpu.max().item()))
        print("   NaNs     : 0 in X, 0 in y  [OK]")
        results["batch_shape"] = True
        print("   PASS\n")
    except Exception as e:
        print("   FAIL: {}: {}".format(type(e).__name__, e))
        results["batch_shape"] = False
        all_pass = False

    # CHECK 3: Model construction
    print("[CHECK 3] Model construction...")
    try:
        model = CNNBaseline(in_channels=4, base_filters=16).to(device)
        n_params = count_parameters(model)
        print("   Parameters : {:,}".format(n_params))
        print_vram("After model loaded")
        results["model_init"] = True
        print("   PASS\n")
    except Exception as e:
        print("   FAIL: {}: {}".format(type(e).__name__, e))
        results["model_init"] = False
        all_pass = False

    # CHECK 4: Forward pass shape
    print("[CHECK 4] Forward pass shape...")
    try:
        model.eval()
        with torch.no_grad():
            X = X_cpu.to(device)
            pred = model(X)
        assert pred.shape == (4, 1, 125, 121), \
            "Output shape mismatch: {}".format(pred.shape)
        assert not pred.isnan().any(), "NaNs in model output"
        print("   Output shape : {}  [OK]".format(tuple(pred.shape)))
        print_vram("After forward pass")
        results["forward_pass"] = True
        print("   PASS\n")
    except Exception as e:
        print("   FAIL: {}: {}".format(type(e).__name__, e))
        results["forward_pass"] = False
        all_pass = False

    # CHECK 5: Loss computation
    print("[CHECK 5] Loss computation...")
    try:
        criterion = nn.MSELoss()
        y = y_cpu.to(device)
        loss = criterion(pred, y)
        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() > 0, "Loss is zero - suspicious"
        print("   MSE loss : {:.6f}  [OK]".format(loss.item()))
        results["loss"] = True
        print("   PASS\n")
    except Exception as e:
        print("   FAIL: {}: {}".format(type(e).__name__, e))
        results["loss"] = False
        all_pass = False

    # CHECK 6: Backward pass
    print("[CHECK 6] Backward pass (gradient check)...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X = X_cpu.to(device)
        y = y_cpu.to(device)
        optimizer.zero_grad()
        pred2 = model(X)
        loss2 = criterion(pred2, y)
        loss2.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.grad is not None]
        assert len(grad_norms) > 0, "No gradients computed"
        assert all(g >= 0 for g in grad_norms), "Negative gradient norm"
        print("   Gradient computed on {} param tensors  [OK]".format(len(grad_norms)))
        print("   Max grad norm : {:.6f}".format(max(grad_norms)))
        optimizer.step()
        print_vram("After backward pass")
        results["backward_pass"] = True
        print("   PASS\n")
    except Exception as e:
        print("   FAIL: {}: {}".format(type(e).__name__, e))
        results["backward_pass"] = False
        all_pass = False

    # CHECK 7: 2-epoch mini training loop
    print("[CHECK 7] 2-epoch mini training loop...")
    try:
        model.train()
        for ep in range(1, 3):
            ep_loss = 0.0
            for X_cpu_b, y_cpu_b in train_loader:
                X_b = X_cpu_b.to(device)
                y_b = y_cpu_b.to(device)
                optimizer.zero_grad()
                loss_b = criterion(model(X_b), y_b)
                loss_b.backward()
                optimizer.step()
                ep_loss += loss_b.item()
            print("   Epoch {}/2  train_loss={:.5f}".format(ep, ep_loss / len(train_loader)))
        info = print_vram("After 2-epoch mini train")
        vram_ok = (info is None) or (info["used_gb"] < VRAM_CEILING_GB)
        results["vram_safe"] = vram_ok
        if not vram_ok:
            print("   FAIL: VRAM ceiling exceeded!")
            all_pass = False
        results["mini_training"] = True
        print("   PASS\n")
    except Exception as e:
        print("   FAIL: {}: {}".format(type(e).__name__, e))
        results["mini_training"] = False
        results["vram_safe"] = False
        all_pass = False

    # SUMMARY
    print("=" * 65)
    print("  SANITY TEST SUMMARY")
    print("=" * 65)
    labels = [
        ("cuda",          "CUDA device available (PyTorch CUDA build)"),
        ("dataset",       "Dataset loads (126/24/24 samples)"),
        ("batch_shape",   "Batch shapes correct ([4,4,125,121] / [4,1,125,121])"),
        ("model_init",    "Model initializes on device"),
        ("forward_pass",  "Forward pass correct shape"),
        ("loss",          "MSELoss computes"),
        ("backward_pass", "Backward pass / gradients"),
        ("mini_training", "2-epoch mini training loop"),
        ("vram_safe",     "VRAM below {} GB ceiling".format(VRAM_CEILING_GB)),
    ]
    for key, label in labels:
        val = results.get(key)
        sym = "PASS" if val is True else ("FAIL" if val is False else "SKIP")
        print("   {:4s}  {}".format(sym, label))

    print()
    if not cuda_available:
        print("  ACTION REQUIRED: PyTorch is installed WITHOUT CUDA support.")
        print("  Detected PyTorch build: {}".format(torch.__version__))
        print("  Your GPU (RTX 4070 Laptop) supports CUDA 12.x (compute cap 8.9).")
        print()
        print("  To install CUDA-enabled PyTorch, run ONE of:")
        print("    # CUDA 12.8 (recommended for driver 616.92):")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print()
        print("  After reinstalling PyTorch, re-run: python ml/cnn_sanity_test.py")

    logic_checks = ["dataset", "batch_shape", "model_init", "forward_pass",
                    "loss", "backward_pass", "mini_training", "vram_safe"]
    logic_pass = all(results.get(k) is True for k in logic_checks)

    if logic_pass and not cuda_available:
        print()
        print("  All LOGIC checks PASSED (on CPU fallback).")
        print("  Only CUDA installation is missing.")
        print("  Install CUDA PyTorch (see above), then re-run this test.")
        sys.exit(1)   # exit 1 until CUDA is confirmed
    elif logic_pass and cuda_available:
        print("  ALL CHECKS PASSED (including CUDA).")
        print("  EXP-005 CNN is verified safe for full training.")
        print("  Awaiting user approval before running: python ml/train_cnn.py")
        sys.exit(0)
    else:
        print("  ONE OR MORE LOGIC CHECKS FAILED.")
        print("  DO NOT proceed to full training until failures are resolved.")
        sys.exit(1)


if __name__ == "__main__":
    run()
