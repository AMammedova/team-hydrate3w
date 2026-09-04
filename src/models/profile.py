"""Member 4 -- GPU profiling utilities for TCN and GRU.

Reports:
- trainable parameter count;
- inference latency per window;
- peak CUDA memory during inference;
- training epoch time and peak CUDA memory during a representative epoch.

GPU numbers are meaningful only for the hardware/configuration used, so record
GPU name, batch size, channel count, window size and AMP setting with results.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_inference(
    model: nn.Module,
    sample_input: tuple[torch.Tensor, torch.Tensor],
    device: str | torch.device = "cuda",
    n_warmup: int = 20,
    n_repeats: int = 100,
) -> dict:
    device = torch.device(device)
    model = model.to(device).eval()
    x, mask = (t.to(device) for t in sample_input)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, mask)
        _sync(device)
        start = time.perf_counter()
        for _ in range(n_repeats):
            model(x, mask)
        _sync(device)
        elapsed = time.perf_counter() - start

    peak = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else float("nan")
    )
    batch_size = int(x.shape[0])
    return {
        "n_parameters": count_parameters(model),
        "inference_peak_vram_mb": peak,
        "latency_ms_per_window": elapsed * 1000.0 / (n_repeats * batch_size),
    }


def profile_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str | torch.device = "cuda",
    amp: bool = True,
    max_batches: int | None = None,
) -> dict:
    """Profile one representative training pass without checkpointing.

    Use a fresh model/optimizer (or reload weights afterwards) because this
    function performs optimizer updates.
    """
    device = torch.device(device)
    model = model.to(device).train()
    loss_fn = loss_fn.to(device)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    n_batches = 0
    n_windows = 0

    for x, mask, y in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x, mask)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_batches += 1
        n_windows += int(x.shape[0])

    _sync(device)
    elapsed = time.perf_counter() - start
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else float("nan")
    )
    return {
        "training_elapsed_s": elapsed,
        "training_batches_profiled": n_batches,
        "training_windows_profiled": n_windows,
        "training_ms_per_batch": elapsed * 1000.0 / max(n_batches, 1),
        "training_peak_vram_mb": peak,
    }


# Backwards-compatible name used by the earlier Member 4 draft.
def profile_model(*args, **kwargs) -> dict:
    return profile_inference(*args, **kwargs)
