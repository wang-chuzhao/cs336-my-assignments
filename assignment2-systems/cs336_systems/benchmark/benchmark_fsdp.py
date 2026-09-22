"""Benchmark peak memory and per-iteration time with/without FSDP.

Answers fsdp_accounting (a) and (b). For (b), run under Nsight Systems to see
whether the weight all-gathers overlap with compute:

    nsys profile -o fsdp_xl --trace=cuda,nvtx \
        uv run python -m cs336_systems.benchmark.bench_fsdp

Usage:
    uv run python -m cs336_systems.benchmark.bench_fsdp
"""

import os
import time

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.parallel.ddp import DDP
from cs336_systems.parallel.fsdp import FSDP

# xl model size from the assignment handout
XL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 128,
    "d_model": 1600,
    "num_layers": 48,
    "num_heads": 25,
    "d_ff": 6400,
}

BATCH_SIZE = 4
WARMUP_STEPS = 2
MEASURE_STEPS = 5


def _mb(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def _setup(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29512")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _run(rank: int, world_size: int, mode: str):
    device = _setup(rank, world_size)
    torch.manual_seed(42)

    base = BasicsTransformerLM(**XL_CONFIG).to(device)
    n_params = sum(p.numel() for p in base.parameters())

    if mode == "fsdp":
        model = FSDP(base)
    elif mode == "ddp":
        model = DDP(base)
    else:
        model = base

    torch.cuda.synchronize()
    mem_after_init = torch.cuda.max_memory_allocated()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    vocab_size = XL_CONFIG["vocab_size"]
    ctx = XL_CONFIG["context_length"]
    inputs = torch.randint(0, vocab_size, (BATCH_SIZE, ctx), device=device)
    targets = torch.randint(0, vocab_size, (BATCH_SIZE, ctx), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    mem_before_step = 0
    mem_after_step = 0
    step_times = []

    for step in range(WARMUP_STEPS + MEASURE_STEPS):
        if step == WARMUP_STEPS:
            # Reset after warmup so reported peaks reflect steady state.
            torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        with nvtx.range(f"forward_step{step}"):
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))

        with nvtx.range(f"backward_step{step}"):
            loss.backward()

        if mode in ("fsdp", "ddp"):
            with nvtx.range(f"grad_sync_step{step}"):
                model.finish_gradient_synchronization()

        torch.cuda.synchronize()
        peak_before = torch.cuda.max_memory_allocated()

        with nvtx.range(f"optim_step{step}"):
            optimizer.step()

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        peak_after = torch.cuda.max_memory_allocated()

        if step >= WARMUP_STEPS:
            mem_before_step = max(mem_before_step, peak_before)
            mem_after_step = max(mem_after_step, peak_after)

    if rank == 0:
        measured = sorted(step_times[WARMUP_STEPS:])
        median_ms = measured[len(measured) // 2] * 1e3
        print(f"\n=== {mode} (world_size={world_size}) ===")
        print(f"peak memory after model init:   {_mb(mem_after_init):8.1f} MiB")
        print(f"peak memory before optim step:  {_mb(mem_before_step):8.1f} MiB")
        print(f"peak memory after optim step:   {_mb(mem_after_step):8.1f} MiB")
        print(f"median time per iteration:      {median_ms:8.1f} ms")

        # Theoretical per-rank breakdown, for comparison with the measurements.
        divisor = world_size if mode == "fsdp" else 1
        print(f"\nparameter count: {n_params / 1e6:.1f}M")
        print(f"params (fp32):          {_mb(n_params * 4 / divisor):8.1f} MiB")
        print(f"grads (fp32):           {_mb(n_params * 4 / divisor):8.1f} MiB")
        print(f"AdamW state (2x fp32):  {_mb(n_params * 8 / divisor):8.1f} MiB")

    dist.destroy_process_group()


def main():
    world_size = 2
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"need {world_size} GPUs, found {torch.cuda.device_count()}")
    for mode in ("ddp", "fsdp"):
        mp.spawn(_run, args=(world_size, mode), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
