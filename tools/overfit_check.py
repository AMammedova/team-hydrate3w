"""Member 4 -- tiny-subset overfit sanity check.

Run this before any expensive GPU sweep. If TCN/GRU cannot memorize ~20
windows with dropout disabled, stop and fix the model/data pipeline first.

    python -m tools.overfit_check --model both --device cuda
"""

from __future__ import annotations

import argparse
import tempfile

import torch
import torch.nn as nn

from src.data.make_fake_data import make_fake_dataset
from src.models.dataset import WindowDataset
from src.models.gru import GRUClassifier
from src.models.tcn import TCN
from src.models.train_loop import seed_everything


def run(
    model_name: str,
    n_windows: int = 20,
    max_steps: int = 500,
    lr: float = 1e-3,
    device: str = "cuda",
) -> float:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    seed_everything(0)
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_dataset(
            tmp,
            n_wells=4,
            n_instances=6,
            n_channels=8,
            window_size=60,
            windows_per_instance=10,
            seed=0,
        )
        ds = WindowDataset.from_directory(tmp)

        if len(ds) < n_windows:
            raise RuntimeError(f"fake dataset produced {len(ds)} windows, need {n_windows}")
        sub = ds.subset(list(range(n_windows)))

        X = torch.stack([sub[i][0] for i in range(len(sub))]).to(device_obj)
        mask = torch.stack([sub[i][1] for i in range(len(sub))]).to(device_obj)
        y = torch.stack([sub[i][2] for i in range(len(sub))]).to(device_obj)

    if model_name == "tcn":
        model = TCN(n_channels=sub.n_channels, num_classes=3, dropout=0.0)
        assert model.receptive_field() >= sub.window_size
    elif model_name == "gru":
        model = GRUClassifier(n_channels=sub.n_channels, num_classes=3, dropout=0.0)
    else:
        raise ValueError(f"unknown model {model_name!r}")

    model = model.to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss().to(device_obj)

    final_loss = float("inf")
    model.train()
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X, mask)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if final_loss < 1e-3:
            print(f"[{model_name}] converged at step {step}, loss={final_loss:.6f}")
            return final_loss

    print(f"[{model_name}] did NOT converge after {max_steps} steps, loss={final_loss:.6f}")
    return final_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["tcn", "gru", "both"], default="both")
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    models = ["tcn", "gru"] if args.model == "both" else [args.model]
    failed = []
    for name in models:
        loss = run(name, args.n_windows, args.max_steps, args.lr, args.device)
        if loss >= 1e-3:
            failed.append(name)
    if failed:
        raise SystemExit(f"OVERFIT CHECK FAILED for {failed}")
    print("All models overfit the tiny subset successfully.")


if __name__ == "__main__":
    main()
