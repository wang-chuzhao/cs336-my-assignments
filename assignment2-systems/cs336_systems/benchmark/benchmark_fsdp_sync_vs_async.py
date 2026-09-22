"""Compare blocking FSDP vs the prefetch/async implementation.

Usage (2 GPUs):
    python -m cs336_systems.benchmark.benchmark_fsdp_sync_vs_async
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from cs336_basics.model import BasicsTransformerLM, Embedding, Linear
from cs336_systems.parallel.fsdp import FSDP


GPU_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 128,
    "d_model": 1600,
    "num_layers": 8,
    "num_heads": 25,
    "d_ff": 6400,
}

# Fat-but-shallow so all-gather is visible without a 2-GPU quota.
CPU_CONFIG = {
    "vocab_size": 4096,
    "context_length": 64,
    "d_model": 768,
    "num_layers": 4,
    "num_heads": 12,
    "d_ff": 3072,
}

BATCH_SIZE = 4
WARMUP_STEPS = 2
MEASURE_STEPS = 5


class FSDPSync(nn.Module):
    """Original blocking FSDP: sync all-gather / reduce-scatter, no prefetch."""

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        self.mod_params = []
        self.grad_handles = []
        self.origin_dtype = None
        self.fp32_shard = {}
        hooked = set()

        for name, mod in self.module.named_modules():
            if isinstance(mod, (Embedding, Linear)):
                p = mod.weight
                if self.origin_dtype is None:
                    self.origin_dtype = p.dtype
                if p.requires_grad:
                    part_length = p.shape[0] // self.world_size
                    start = self.rank * part_length
                    self.mod_params.append(
                        {"name": name + ".weight", "full_shape": p.shape, "param": p}
                    )
                    p.data = p.data[start : start + part_length].clone()
                    mod.register_forward_pre_hook(self.fwd_pre)
                    mod.register_forward_hook(self.fwd_post)
                    if isinstance(mod, Linear):
                        mod.register_full_backward_pre_hook(self.bwd_pre)
                    if id(p) not in hooked:
                        p.register_post_accumulate_grad_hook(self.sharded_grad_hook)
                        hooked.add(id(p))

        for mod in self.module.modules():
            for p in mod.parameters(recurse=False):
                if p.requires_grad and id(p) not in hooked:
                    p.register_post_accumulate_grad_hook(self.full_grad_hook)

    def _unshard(self, module):
        shard = module.weight.data
        self.fp32_shard[id(module.weight)] = shard.clone()
        if self.compute_dtype is not None:
            shard = shard.to(self.compute_dtype)
        full = torch.empty(
            (shard.shape[0] * self.world_size, shard.shape[1]),
            dtype=shard.dtype,
            device=shard.device,
        )
        dist.all_gather_into_tensor(full, shard.contiguous())
        module.weight.data = full

    def fwd_pre(self, module, *args):
        self._unshard(module)

    def fwd_post(self, module, *args):
        module.weight.data = self.fp32_shard[id(module.weight)]

    def bwd_pre(self, module, *args):
        self._unshard(module)

    def sharded_grad_hook(self, param):
        full_shape = None
        for item in self.mod_params:
            if item["param"] is param:
                full_shape = item["full_shape"]
                break
        part_length = full_shape[0] // self.world_size
        if param.data.shape[0] == full_shape[0]:
            param.data = self.fp32_shard.pop(id(param))
        full_grad = param.grad.data.to(self.origin_dtype) / self.world_size
        output = torch.empty(
            (part_length, full_shape[1]),
            dtype=full_grad.dtype,
            device=full_grad.device,
        )
        dist.reduce_scatter_tensor(output, full_grad.contiguous(), op=dist.ReduceOp.SUM)
        param.grad = output

    def full_grad_hook(self, param):
        param.grad.data /= self.world_size
        handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
        self.grad_handles.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.grad_handles:
            handle.wait()
        self.grad_handles.clear()


def _setup(rank: int, world_size: int, backend: str, port: str):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def _median_ms(times):
    s = sorted(times)
    return s[len(s) // 2] * 1e3


def _run(rank: int, world_size: int, mode: str, backend: str, port: str, config: dict):
    device = _setup(rank, world_size, backend, port)
    torch.manual_seed(42)

    base = BasicsTransformerLM(**config).to(device)
    model = FSDP(base) if mode == "async" else FSDPSync(base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    vocab = config["vocab_size"]
    ctx = config["context_length"]
    inputs = torch.randint(0, vocab, (BATCH_SIZE, ctx), device=device)
    targets = torch.randint(0, vocab, (BATCH_SIZE, ctx), device=device)

    step_times = []
    for step in range(WARMUP_STEPS + MEASURE_STEPS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits.reshape(-1, vocab), targets.reshape(-1))
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= WARMUP_STEPS:
            step_times.append(time.perf_counter() - t0)

    if rank == 0:
        mean_ms = sum(step_times) / len(step_times) * 1e3
        print(f"backend={backend}  config={config}", flush=True)
        print(
            f"{mode:6s}  median={_median_ms(step_times):8.1f} ms  "
            f"mean={mean_ms:8.1f} ms  "
            f"steps={[round(t * 1e3, 1) for t in step_times]}",
            flush=True,
        )

    dist.destroy_process_group()


def main():
    world_size = 2
    if torch.cuda.device_count() >= world_size:
        backend = "nccl"
    else:
        backend = "gloo"
        print(f"no {world_size} GPUs, falling back to gloo/cpu", flush=True)

    config = GPU_CONFIG if backend == "nccl" else CPU_CONFIG
    print(f"world_size={world_size} backend={backend} config={config}", flush=True)
    for i, mode in enumerate(("sync", "async")):
        port = str(29611 + i)
        mp.spawn(
            _run,
            args=(world_size, mode, backend, port, config),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
