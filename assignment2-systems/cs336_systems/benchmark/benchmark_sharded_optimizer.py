"""Benchmark peak memory and per-iteration time with/without optimizer state sharding.

Answers optimizer_state_sharding_accounting (a) and (b).

Usage:
    uv run python -m cs336_systems.benchmark.bench_sharded_optimizer
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.parallel.sharded_optimizer import ShardedOptimizer

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
    os.environ.setdefault("MASTER_PORT", "29511")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _run(rank: int, world_size: int, sharded: bool):
    device = _setup(rank, world_size)
    torch.manual_seed(42)

    model = BasicsTransformerLM(**XL_CONFIG).to(device)
    torch.cuda.synchronize()
    mem_after_init = torch.cuda.max_memory_allocated()

    if sharded:
        optimizer = ShardedOptimizer(
            model.parameters(), torch.optim.AdamW, lr=1e-4, weight_decay=0.01
        )
    else:
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
            # Only reset the peak counter once warmup allocations have settled,
            # so the reported peaks reflect steady-state training.
            torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()

        torch.cuda.synchronize()
        peak_before = torch.cuda.max_memory_allocated()

        optimizer.step()

        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        peak_after = torch.cuda.max_memory_allocated()

        if step >= WARMUP_STEPS:
            mem_before_step = max(mem_before_step, peak_before)
            mem_after_step = max(mem_after_step, peak_after)

    if rank == 0:
        label = "sharded" if sharded else "unsharded"
        median_ms = sorted(step_times[WARMUP_STEPS:])[len(step_times[WARMUP_STEPS:]) // 2] * 1e3
        print(f"\n=== optimizer state sharding: {label} (world_size={world_size}) ===")
        print(f"peak memory after model init:   {_mb(mem_after_init):8.1f} MiB")
        print(f"peak memory before optim step:  {_mb(mem_before_step):8.1f} MiB")
        print(f"peak memory after optim step:   {_mb(mem_after_step):8.1f} MiB")
        print(f"median time per iteration:      {median_ms:8.1f} ms")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nparameter count: {n_params / 1e6:.1f}M")
        print(f"params (fp32):          {_mb(n_params * 4):8.1f} MiB")
        print(f"grads (fp32):           {_mb(n_params * 4):8.1f} MiB")
        adam_state = n_params * 4 * 2
        if sharded:
            adam_state //= world_size
        print(f"AdamW state (2x fp32):  {_mb(adam_state):8.1f} MiB")

    dist.destroy_process_group()


def main():
    world_size = 2
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"need {world_size} GPUs, found {torch.cuda.device_count()}"
        )
    for sharded in (False, True):
        mp.spawn(_run, args=(world_size, sharded), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
